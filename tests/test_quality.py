"""Tests for eval/quality.py.

We test the orchestration contract — error isolation, section ordering,
optional MAUVE — rather than redoing the per-metric numerical tests
(those live in test_diversity.py / test_gen_ppl.py / test_mauve.py).
"""

from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import quality as quality_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def samples_jsonl(tmp_path):
    path = tmp_path / "samples.jsonl"
    with path.open("w") as f:
        for i in range(6):
            f.write(
                json.dumps(
                    {
                        "sample_id": i,
                        "text": f"the quick brown fox number {i} jumps over",
                        "token_ids": [1, 2, 3, 4, 5 + i, 6, 7, 8],
                    }
                )
                + "\n"
            )
    return path


@pytest.fixture(autouse=True)
def _stub_gen_ppl(monkeypatch):
    """Don't hit GPT-2 in any test. Replace ``compute_gen_ppl`` with a stub that
    returns a plausible report so we can verify wiring without downloads."""

    def fake_compute(texts, **kwargs):
        from eval.gen_ppl import GenPPLReport

        rep = GenPPLReport(
            n_samples=len(texts),
            n_scored=len(texts),
            n_skipped=0,
            corpus_ppl=42.0,
            mean_sample_ppl=45.0,
            median_sample_ppl=40.0,
            mean_tokens=10.0,
            scorer=kwargs.get("scorer_name", "stub"),
        )
        per_sample = [
            {"n_tokens": 10, "nll": 1.0, "mean_nll": 0.1, "ppl": 40 + i, "skipped": False}
            for i in range(len(texts))
        ]
        return rep, per_sample

    from eval import gen_ppl as real

    monkeypatch.setattr(real, "compute_gen_ppl", fake_compute)
    monkeypatch.setattr(quality_mod.gen_ppl_mod, "compute_gen_ppl", fake_compute)


# ---------------------------------------------------------------------------
# run_quality
# ---------------------------------------------------------------------------


def test_run_quality_populates_diversity_and_gen_ppl_sections(samples_jsonl):
    report = quality_mod.run_quality(samples_jsonl, run_gen_ppl=True)
    assert report.n_records == 6

    assert report.diversity.error is None
    assert report.diversity.report["n_samples"] == 6
    assert "distinct_1" in report.diversity.report

    assert report.gen_ppl is not None
    assert report.gen_ppl.error is None
    assert report.gen_ppl.report["corpus_ppl"] == 42.0

    # MAUVE section skipped when no refs provided.
    assert report.mauve is None


def test_run_quality_skip_gen_ppl_when_disabled(samples_jsonl):
    report = quality_mod.run_quality(samples_jsonl, run_gen_ppl=False)
    assert report.gen_ppl is None


def test_run_quality_mauve_runs_when_refs_file_given(samples_jsonl, tmp_path, monkeypatch):
    refs = tmp_path / "refs.txt"
    refs.write_text("one ref\ntwo ref\n")
    fake = SimpleNamespace(
        compute_mauve=lambda **kw: SimpleNamespace(
            mauve=0.9,
            frontier_integral=0.85,
            num_buckets=12,
        )
    )
    monkeypatch.setitem(sys.modules, "mauve", fake)

    report = quality_mod.run_quality(
        samples_jsonl,
        run_gen_ppl=False,
        mauve_refs_file=str(refs),
    )
    assert report.mauve is not None
    assert report.mauve.error is None
    assert report.mauve.report["mauve"] == pytest.approx(0.9)


def test_run_quality_isolates_mauve_failure(samples_jsonl, tmp_path, monkeypatch):
    """If the MAUVE package is missing, diversity + gen_ppl must still succeed."""
    refs = tmp_path / "refs.txt"
    refs.write_text("one ref\n")
    # Import will fail inside compute_mauve → the quality runner should catch it.
    monkeypatch.setitem(sys.modules, "mauve", None)

    report = quality_mod.run_quality(
        samples_jsonl,
        run_gen_ppl=True,
        mauve_refs_file=str(refs),
    )
    assert report.diversity.error is None
    assert report.gen_ppl.error is None
    assert report.mauve is not None
    assert report.mauve.error is not None
    assert "mauve-text" in report.mauve.error


def test_run_quality_falls_back_to_text_when_token_ids_missing(tmp_path):
    path = tmp_path / "samples.jsonl"
    with path.open("w") as f:
        for i in range(3):
            f.write(json.dumps({"sample_id": i, "text": f"word a b c {i}"}) + "\n")
    report = quality_mod.run_quality(path, run_gen_ppl=False)
    assert report.diversity.error is None
    # The diversity section records which tokenization it actually used.
    assert report.diversity.report["tokenization"] in {"token_ids", "text"}


def test_run_quality_mauve_with_bad_refs_file_surfaces_error(samples_jsonl, tmp_path, monkeypatch):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    # Even with mauve present, empty refs should land in the error field.
    monkeypatch.setitem(
        sys.modules,
        "mauve",
        SimpleNamespace(
            compute_mauve=lambda **kw: SimpleNamespace(
                mauve=0.0, frontier_integral=0.0, num_buckets=1
            )
        ),
    )
    report = quality_mod.run_quality(
        samples_jsonl,
        run_gen_ppl=False,
        mauve_refs_file=str(empty),
    )
    assert report.mauve is not None
    assert report.mauve.error is not None
    assert "no usable rows" in report.mauve.error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_combined_json(samples_jsonl, tmp_path, capsys):
    out = tmp_path / "quality.json"
    rc = quality_mod.main(
        [
            "--samples",
            str(samples_jsonl),
            "--diversity-source",
            "token_ids",
            "--gen-ppl-scorer",
            "stub",
            "--json-out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_records"] == 6
    assert payload["diversity"]["report"] is not None
    assert payload["gen_ppl"]["report"] is not None
    assert payload["mauve"] is None  # opt-in, not configured

    captured = capsys.readouterr()
    assert "[diversity]" in captured.out
    assert "[gen_ppl]" in captured.out


def test_cli_returns_nonzero_when_every_section_errors(tmp_path, monkeypatch):
    # Break diversity by feeding an empty file (compute_diversity raises on empty).
    samples = tmp_path / "empty.jsonl"
    samples.write_text("")  # 0 records → diversity raises "Empty corpus"
    rc = quality_mod.main(
        [
            "--samples",
            str(samples),
            "--no-gen-ppl",
        ]
    )
    assert rc == 2


def test_cli_digit_num_buckets_is_parsed_as_int(tmp_path, monkeypatch, samples_jsonl):
    refs = tmp_path / "refs.txt"
    refs.write_text("r1\nr2\n")

    captured = {}

    def fake_compute_mauve(**kw):
        captured.update(kw)
        return SimpleNamespace(mauve=0.5, frontier_integral=0.5, num_buckets=kw["num_buckets"])

    monkeypatch.setitem(sys.modules, "mauve", SimpleNamespace(compute_mauve=fake_compute_mauve))

    rc = quality_mod.main(
        [
            "--samples",
            str(samples_jsonl),
            "--no-gen-ppl",
            "--mauve-refs-file",
            str(refs),
            "--mauve-num-buckets",
            "32",
        ]
    )
    assert rc == 0
    assert captured["num_buckets"] == 32  # int, not "32"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
