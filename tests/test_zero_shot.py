"""Tests for eval/zero_shot.py.

We stay offline here: the HF Hub download path is stubbed with a tiny
in-memory :class:`datasets.Dataset`, and the model forward is replaced with
a deterministic loss so we can assert on exact bookkeeping without loading
a real checkpoint. The goal is to lock down (a) the table/registry shape,
(b) failure-isolation between datasets, and (c) the tokenization plumbing
(``join_docs`` in particular, since it's the easiest thing to get wrong).
"""

from __future__ import annotations

import pathlib
import sys
from collections import namedtuple

import datasets as hf_datasets
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import zero_shot as zs  # noqa: E402

# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_contains_all_default_datasets():
    """Every name in DEFAULT_DATASETS must resolve in DATASET_REGISTRY."""
    for name in zs.DEFAULT_DATASETS:
        assert name in zs.DATASET_REGISTRY, f"{name} missing from registry"
        spec = zs.DATASET_REGISTRY[name]
        assert spec.name == name
        assert spec.hf_name
        assert spec.split


def test_registry_entries_have_plausible_fields():
    """Catch copy-paste bugs: every entry has a non-empty text column and split."""
    for spec in zs.DATASET_REGISTRY.values():
        assert spec.text_column, f"{spec.name}: empty text_column"
        assert spec.split in {
            "test",
            "validation",
            "train",
        }, f"{spec.name}: suspicious split {spec.split!r}"


def test_run_zero_shot_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown datasets"):
        zs.run_zero_shot("dummy.ckpt", datasets=["does_not_exist"])


# ---------------------------------------------------------------------------
# Table / row formatting
# ---------------------------------------------------------------------------


def _fake_entry(name="wikitext103", ppl=123.45, nll=2.34, err=None):
    return zs.ZeroShotEntry(
        dataset=name,
        hf_name="wikitext",
        hf_subset="wikitext-103-raw-v1",
        split="test",
        ppl=ppl,
        per_token_nll=nll,
        n_sequences=100,
        tokens_per_pass=25600,
        wall_seconds=12.3,
        error=err,
    )


def test_format_table_renders_ok_and_failure_rows():
    table = zs._format_table(
        [
            _fake_entry(),
            _fake_entry(name="ptb", ppl=None, nll=None, err="RuntimeError: boom"),
        ]
    )
    assert "wikitext103" in table
    assert "123.45" in table
    assert "(fail)" in table  # sentinel for failed run
    # Status column is truncated to 8 chars so we only see the prefix.
    assert "Runtime" in table
    # Markdown table shape: 4 lines (header, sep, row, row).
    assert len(table.splitlines()) == 4


def test_row_from_error_captures_exception_type_and_message():
    spec = zs.DATASET_REGISTRY["ptb"]
    row = zs._row_from_error(spec, RuntimeError("network down"), wall=5.0)
    assert row.error is not None
    assert "RuntimeError" in row.error
    assert "network down" in row.error
    assert row.ppl is None
    assert row.wall_seconds == 5.0


# ---------------------------------------------------------------------------
# Dataloader construction (offline, stubbed _load_split)
# ---------------------------------------------------------------------------


class _MockTok:
    """Deterministic tokenizer stub that matches the HF call sites we use."""

    def __init__(self, seq_len: int = 16):
        self.seq_len = seq_len

    def __call__(self, texts, max_length, truncation, padding, return_attention_mask):
        # Each 'word' -> id len(word), pad to max_length with 0.
        input_ids = []
        attn = []
        for t in texts:
            toks = [len(w) % 31 + 1 for w in t.split()][:max_length]
            mask = [1] * len(toks)
            pad = max_length - len(toks)
            input_ids.append(toks + [0] * pad)
            attn.append(mask + [0] * pad)
        return {"input_ids": input_ids, "attention_mask": attn}


def test_build_test_loader_join_docs_packs_short_rows(monkeypatch):
    """join_docs=True should merge sentence-level rows into longer strings."""
    # Each sentence must pass the >10-char filter but also be small enough
    # that several fit in the seq_len*4=64-char packing budget; "short snippet N"
    # is ~15 chars, so 3-4 pack per joined string.
    short_rows = [{"sentence": f"short snippet number{i:02d}"} for i in range(20)]
    fake = hf_datasets.Dataset.from_list(short_rows)
    monkeypatch.setattr(zs, "_load_split", lambda spec: fake)

    spec = zs.DATASET_REGISTRY["ptb"]
    assert spec.join_docs, "sanity: ptb spec is expected to be join_docs"

    loader = zs._build_test_loader(
        spec, tokenizer=_MockTok(), seq_len=16, batch_size=2, num_workers=0
    )
    batches = list(loader)
    # We should have strictly fewer batches than naive rows/batch_size because
    # join_docs packed sentences together. 20 rows / 2 = 10 batches naively;
    # with packing it must be < 10.
    assert 1 <= len(batches) < 10, f"packing didn't reduce batch count: {len(batches)}"
    for b in batches:
        assert b["input_ids"].shape == (2, 16)
        assert b["attention_mask"].shape == (2, 16)


def test_build_test_loader_filters_empty_rows(monkeypatch):
    """Empty/blank rows must be dropped (they'd score as pure pad otherwise)."""
    rows = [
        {"text": ""},
        {"text": "   "},  # whitespace-only
        {"text": "x"},  # too short (<=10 after strip)
        {"text": "this is a valid training sentence with content"},
        {"text": "another well-formed input for perplexity scoring"},
    ]
    fake = hf_datasets.Dataset.from_list(rows)
    monkeypatch.setattr(zs, "_load_split", lambda spec: fake)

    spec = zs.DATASET_REGISTRY["wikitext103"]  # join_docs=False, text column
    loader = zs._build_test_loader(
        spec, tokenizer=_MockTok(), seq_len=16, batch_size=1, num_workers=0
    )
    batches = list(loader)
    # Only 2 rows survive the length filter; batch_size=1 & drop_last=False → 2 batches.
    # Note: DataLoader is created with drop_last=True in our code, so depending on
    # batch_size the count may be 2 (1+1) with drop_last=True too.
    assert len(batches) == 2


# ---------------------------------------------------------------------------
# End-to-end harness with mocked model+loader
# ---------------------------------------------------------------------------


MockLoss = namedtuple("MockLoss", ["loss", "nlls", "token_mask"])


class _FakeModel(torch.nn.Module):
    """A model stand-in with just enough surface for compute_perplexity."""

    def __init__(self, block_size=4):
        super().__init__()
        self.block_size = block_size

    def _loss(self, x, mask):
        # Deterministic NLL proportional to the token id so tests can reason
        # about aggregation exactly.
        nlls = (x.float() * 0.01) * mask.float()
        token_mask = mask.float()
        loss = nlls.sum() / token_mask.sum().clamp_min(1.0)
        return MockLoss(loss=loss, nlls=nlls, token_mask=token_mask)


def test_run_zero_shot_isolates_per_dataset_failures(monkeypatch):
    """A failure on one dataset must not poison the others.

    We stub load_abd3_from_checkpoint to return a _FakeModel, then make
    ptb raise at dataloader build time while wikitext103 succeeds.
    """
    fake_model = _FakeModel()

    class _Cfg:
        class _Data:
            name = "wikitext"

        class _Model:
            length = 16

        class _Loader:
            eval_batch_size = 2

        data = _Data()
        model = _Model()
        loader = _Loader()

    def fake_load(*args, **kwargs):
        fake_model.config = _Cfg()
        return fake_model, _Cfg(), _MockTok(seq_len=16)

    def fake_build_loader(spec, tokenizer, seq_len, batch_size, num_workers=0):
        if spec.name == "ptb":
            raise RuntimeError("synthetic network failure")
        rows = [
            {"text": "well-formed evaluation row number one"},
            {"text": "well-formed evaluation row number two"},
        ]
        ds = hf_datasets.Dataset.from_list(rows)

        def _tok(batch):
            return tokenizer(
                batch["text"],
                max_length=seq_len,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )

        tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
        tokenized.set_format("torch")
        return torch.utils.data.DataLoader(
            tokenized, batch_size=batch_size, shuffle=False, drop_last=True
        )

    monkeypatch.setattr(zs, "load_abd3_from_checkpoint", fake_load)
    monkeypatch.setattr(zs, "_build_test_loader", fake_build_loader)

    entries = zs.run_zero_shot(
        checkpoint="dummy.ckpt",
        datasets=["wikitext103", "ptb"],
        n_samples=1,
        device="cpu",
        use_ema=False,
        progress=False,
    )

    assert len(entries) == 2
    by_name = {e.dataset: e for e in entries}
    assert by_name["wikitext103"].error is None
    assert by_name["wikitext103"].ppl is not None
    assert by_name["ptb"].error is not None
    assert "RuntimeError" in by_name["ptb"].error


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
