"""Tests for eval/gen_ppl.py.

We keep GPT-2 out of the test environment (no network, no several-hundred-MB
download) by mocking the scorer with a tiny handcrafted LM + tokenizer that
still exposes the HF-compatible interface ``score_texts_with_lm`` expects
(``__call__(batch_text_list, return_tensors=..., padding=..., ...) -> dict``
and ``model(input_ids=..., attention_mask=...).logits``).
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import gen_ppl  # noqa: E402

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Character-level tokenizer with a closed vocab — enough for score_texts_with_lm."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.bos_token_id = 1

    def __call__(
        self,
        texts: Sequence[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ):
        ids = []
        for t in texts:
            row = [ord(c) % (self.vocab_size - 3) + 3 for c in t]  # reserve 0..2
            row = row[:max_length]
            if not row:
                row = [3]
            ids.append(row)
        max_len = max(len(r) for r in ids)
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        input_ids, attn = [], []
        for row in ids:
            pad = max_len - len(row)
            input_ids.append(row + [pad_id] * pad)
            attn.append([1] * len(row) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class _FakeLM(torch.nn.Module):
    """Vocab-sized linear head over input embeddings — cheap and deterministic."""

    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.head = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return SimpleNamespace(logits=logits)


# ---------------------------------------------------------------------------
# score_texts_with_lm
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_lm():
    torch.manual_seed(0)
    return _FakeLM(vocab_size=64), _FakeTokenizer(vocab_size=64)


def test_score_texts_returns_one_row_per_input(fake_lm):
    model, tok = fake_lm
    texts = ["hello world", "foo bar baz", "x"]
    rows = gen_ppl.score_texts_with_lm(
        texts,
        scorer_model=model,
        scorer_tokenizer=tok,
        device="cpu",
        batch_size=2,
        min_tokens=2,
    )
    assert len(rows) == 3
    for t, r in zip(texts, rows, strict=True):
        assert r["text"] == t


def test_score_texts_produces_finite_ppl_for_normal_inputs(fake_lm):
    model, tok = fake_lm
    rows = gen_ppl.score_texts_with_lm(
        ["hello world", "abcdefg"],
        scorer_model=model,
        scorer_tokenizer=tok,
        device="cpu",
        batch_size=4,
        min_tokens=2,
    )
    for r in rows:
        assert not r["skipped"]
        assert r["ppl"] > 0
        assert r["ppl"] < float("inf")
        assert r["n_tokens"] >= 2


def test_score_texts_marks_short_inputs_as_skipped(fake_lm):
    model, tok = fake_lm
    # Single-char text → 1 tokenized char + BOS = 2 positions → shifted length = 1 → min_tokens=5 skips it.
    rows = gen_ppl.score_texts_with_lm(
        ["x"],
        scorer_model=model,
        scorer_tokenizer=tok,
        device="cpu",
        batch_size=1,
        min_tokens=5,
    )
    assert rows[0]["skipped"] is True
    assert rows[0]["ppl"] is None


def test_score_texts_sets_pad_token_when_missing(fake_lm):
    model, tok = fake_lm
    assert tok.pad_token is None
    gen_ppl.score_texts_with_lm(
        ["hi there"],
        scorer_model=model,
        scorer_tokenizer=tok,
        device="cpu",
        batch_size=1,
    )
    assert tok.pad_token == tok.eos_token
    assert tok.pad_token_id is not None


def test_score_texts_prepends_bos_when_requested(fake_lm):
    model, tok = fake_lm
    # Same text, same model — enabling BOS should change the scored token count by +1.
    rows_bos = gen_ppl.score_texts_with_lm(
        ["hello"],
        scorer_model=model,
        scorer_tokenizer=tok,
        device="cpu",
        add_bos=True,
        min_tokens=1,
    )
    rows_no_bos = gen_ppl.score_texts_with_lm(
        ["hello"],
        scorer_model=model,
        scorer_tokenizer=tok,
        device="cpu",
        add_bos=False,
        min_tokens=1,
    )
    assert rows_bos[0]["n_tokens"] == rows_no_bos[0]["n_tokens"] + 1


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def test_corpus_ppl_ignores_skipped_rows():
    rows = [
        {"skipped": True, "ppl": None, "nll": None, "n_tokens": 0},
        {"skipped": False, "ppl": 10.0, "nll": 2.302585 * 3, "n_tokens": 3},
        {"skipped": False, "ppl": 10.0, "nll": 2.302585 * 3, "n_tokens": 3},
    ]
    # total_nll / total_tok = 2*2.302585*3 / 6 = 2.302585 -> exp ≈ 10
    assert gen_ppl._corpus_ppl(rows) == pytest.approx(10.0, rel=1e-3)


def test_corpus_ppl_returns_nan_when_everything_skipped():
    import math as _m

    rows = [{"skipped": True, "ppl": None, "nll": None, "n_tokens": 0}]
    assert _m.isnan(gen_ppl._corpus_ppl(rows))


def test_mean_and_median_ignore_non_finite():
    import math as _m

    rows = [
        {"skipped": False, "ppl": 5.0, "nll": 1.0, "n_tokens": 1, "mean_nll": 1.0},
        {"skipped": False, "ppl": float("inf"), "nll": 1e10, "n_tokens": 1, "mean_nll": 1e10},
        {"skipped": False, "ppl": 15.0, "nll": 1.0, "n_tokens": 1, "mean_nll": 1.0},
    ]
    assert gen_ppl._mean_sample_ppl(rows) == pytest.approx(10.0)
    assert gen_ppl._median_sample_ppl(rows) == 10.0

    # All infinite/nan → nan, not a crash.
    only_bad = [{"skipped": False, "ppl": float("inf"), "nll": 1.0, "n_tokens": 1}]
    assert _m.isnan(gen_ppl._mean_sample_ppl(only_bad))


def test_aggregate_produces_populated_report():
    rows = [
        {"skipped": False, "ppl": 20.0, "nll": 1.0, "n_tokens": 10, "mean_nll": 0.1},
        {"skipped": False, "ppl": 40.0, "nll": 2.0, "n_tokens": 20, "mean_nll": 0.1},
        {"skipped": True, "ppl": None, "nll": None, "n_tokens": 0},
    ]
    report = gen_ppl.aggregate(rows, scorer="fake")
    assert report.n_samples == 3
    assert report.n_scored == 2
    assert report.n_skipped == 1
    assert report.mean_tokens == 15.0
    assert "corpus-PPL" in report.summary_line()
    assert "scorer=fake" in report.summary_line()


# ---------------------------------------------------------------------------
# compute_gen_ppl end-to-end (with monkey-patched HF loaders)
# ---------------------------------------------------------------------------


def test_compute_gen_ppl_uses_injected_scorer(monkeypatch, fake_lm):
    model, tok = fake_lm

    class _FakeAuto:
        @staticmethod
        def from_pretrained(name):
            return {"tokenizer": tok, "model": model}[name]

    import transformers

    # Patch both AutoTokenizer and AutoModelForCausalLM so the loader sees our fakes.
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda n: tok),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda n: model),
    )

    report, per_sample = gen_ppl.compute_gen_ppl(
        ["hello world", "abcd efgh"],
        scorer_name="fake",
        device="cpu",
        batch_size=2,
    )
    assert report.n_samples == 2
    assert report.scorer == "fake"
    assert len(per_sample) == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_returns_1_on_empty_jsonl(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert gen_ppl.main(["--samples", str(p)]) == 1


def test_cli_returns_1_when_all_texts_empty(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text(json.dumps({"sample_id": 0, "text": "   "}) + "\n")
    assert gen_ppl.main(["--samples", str(p)]) == 1


def test_cli_writes_json_with_injected_scorer(tmp_path, monkeypatch, fake_lm, capsys):
    model, tok = fake_lm

    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda n: tok),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda n: model),
    )

    samples = tmp_path / "samples.jsonl"
    with samples.open("w") as f:
        for i in range(3):
            f.write(json.dumps({"sample_id": i, "text": f"hello world {i}"}) + "\n")
    out = tmp_path / "gen_ppl.json"
    rc = gen_ppl.main(
        [
            "--samples",
            str(samples),
            "--scorer",
            "fake",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--json-out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["scorer"] == "fake"
    assert payload["report"]["n_samples"] == 3
    assert len(payload["per_sample"]) == 3
    assert "text" not in payload["per_sample"][0]  # stripped for compactness

    assert "corpus-PPL" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
