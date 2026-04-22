"""Tests for eval/mauve.py.

MAUVE itself is an optional dependency with a large GPU footprint, so we
mock the ``mauve.compute_mauve`` function and exercise only the wrapper,
argument plumbing, and reference loaders. This is enough to prove the
contract: given a mocked MAUVE, does our code pass the right kwargs and
translate the result back into a ``MauveReport``?
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

from eval import mauve as mauve_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------


def test_load_refs_from_file_strips_blank_lines(tmp_path):
    p = tmp_path / "refs.txt"
    p.write_text("first line\n\n  second line  \n\n\nthird line\n")
    refs = mauve_mod.load_refs_from_file(p, max_refs=None)
    assert refs == ["first line", "second line", "third line"]


def test_load_refs_from_file_respects_max_refs(tmp_path):
    p = tmp_path / "refs.txt"
    p.write_text("\n".join(f"row{i}" for i in range(10)))
    refs = mauve_mod.load_refs_from_file(p, max_refs=3)
    assert refs == ["row0", "row1", "row2"]


def test_load_refs_from_hf_rejects_malformed_spec():
    with pytest.raises(ValueError, match="dataset|subset|split|column"):
        mauve_mod.load_refs_from_hf("just_dataset_name", max_refs=10)


def test_load_refs_from_hf_parses_empty_subset(monkeypatch):
    captured = {}

    class FakeRow(dict):
        pass

    def fake_load_dataset(name, subset, split, streaming):
        captured["name"] = name
        captured["subset"] = subset
        captured["split"] = split
        return iter(
            [
                FakeRow({"text": "alpha"}),
                FakeRow({"text": ""}),  # skipped: empty
                FakeRow({"other": "ignored"}),  # skipped: no text column
                FakeRow({"text": "beta"}),
            ]
        )

    fake_datasets = SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    refs = mauve_mod.load_refs_from_hf("wikitext|-|validation|text", max_refs=10)
    assert refs == ["alpha", "beta"]
    assert captured["name"] == "wikitext"
    assert captured["subset"] is None
    assert captured["split"] == "validation"


def test_load_refs_from_hf_caps_at_max_refs(monkeypatch):
    def fake_load_dataset(name, subset, split, streaming):
        return iter([{"text": f"t{i}"} for i in range(100)])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    refs = mauve_mod.load_refs_from_hf("ds|-|train|text", max_refs=4)
    assert len(refs) == 4


# ---------------------------------------------------------------------------
# compute_mauve wrapper
# ---------------------------------------------------------------------------


def test_compute_mauve_raises_clear_error_when_package_missing(monkeypatch):
    # Remove `mauve` from sys.modules and block imports.
    monkeypatch.setitem(sys.modules, "mauve", None)
    with pytest.raises(RuntimeError, match="mauve-text"):
        mauve_mod.compute_mauve(["hi"], ["hello"])


def test_compute_mauve_rejects_empty_inputs(monkeypatch):
    fake = SimpleNamespace(
        compute_mauve=lambda **kw: SimpleNamespace(
            mauve=0.5,
            frontier_integral=0.5,
            num_buckets=10,
        )
    )
    monkeypatch.setitem(sys.modules, "mauve", fake)
    with pytest.raises(ValueError, match="gen_texts"):
        mauve_mod.compute_mauve([], ["ref"])
    with pytest.raises(ValueError, match="ref_texts"):
        mauve_mod.compute_mauve(["hi"], [])


def test_compute_mauve_passes_through_kwargs(monkeypatch):
    captured = {}

    def fake_compute_mauve(**kw):
        captured.update(kw)
        return SimpleNamespace(mauve=0.92, frontier_integral=0.88, num_buckets=17)

    monkeypatch.setitem(sys.modules, "mauve", SimpleNamespace(compute_mauve=fake_compute_mauve))

    report = mauve_mod.compute_mauve(
        ["gen1", "gen2"],
        ["ref1", "ref2", "ref3"],
        featurize_model="gpt2",
        device_id=-1,
        max_text_length=256,
        scaling_factor=7.0,
        num_buckets=16,
        seed=123,
        verbose=False,
    )

    assert report.mauve == pytest.approx(0.92)
    assert report.frontier_integral == pytest.approx(0.88)
    assert report.cluster_k == 17
    assert report.n_samples == 2
    assert report.n_refs == 3
    assert report.seed == 123
    assert report.featurize_model == "gpt2"

    assert captured["featurize_model_name"] == "gpt2"
    assert captured["device_id"] == -1
    assert captured["max_text_length"] == 256
    assert captured["num_buckets"] == 16
    # newer mauve key; the wrapper always emits it
    assert captured["mauve_scaling_factor"] == 7.0


def test_report_summary_line_contains_scalars(monkeypatch):
    fake = SimpleNamespace(
        compute_mauve=lambda **kw: SimpleNamespace(
            mauve=0.5,
            frontier_integral=0.44,
            num_buckets=10,
        )
    )
    monkeypatch.setitem(sys.modules, "mauve", fake)
    report = mauve_mod.compute_mauve(["a"], ["b"])
    line = report.summary_line()
    assert "MAUVE=" in line
    assert "FI=" in line
    assert "n_gen=1" in line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_samples(path: pathlib.Path, n: int = 3) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"sample_id": i, "text": f"sample text {i}"}) + "\n")


def test_cli_returns_1_on_empty_samples(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("")
    refs = tmp_path / "refs.txt"
    refs.write_text("one\ntwo\n")
    rc = mauve_mod.main(["--samples", str(samples), "--refs-file", str(refs)])
    assert rc == 1


def test_cli_returns_1_when_all_gens_empty(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": 0, "text": ""}) + "\n")
    refs = tmp_path / "refs.txt"
    refs.write_text("one\ntwo\n")
    rc = mauve_mod.main(["--samples", str(samples), "--refs-file", str(refs)])
    assert rc == 1


def test_cli_returns_1_when_refs_empty(tmp_path):
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples)
    refs = tmp_path / "empty_refs.txt"
    refs.write_text("")
    rc = mauve_mod.main(["--samples", str(samples), "--refs-file", str(refs)])
    assert rc == 1


def test_cli_happy_path_writes_json(tmp_path, monkeypatch, capsys):
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples, n=5)
    refs = tmp_path / "refs.txt"
    refs.write_text("ref one\nref two\nref three\n")

    fake = SimpleNamespace(
        compute_mauve=lambda **kw: SimpleNamespace(
            mauve=0.77,
            frontier_integral=0.66,
            num_buckets=8,
        )
    )
    monkeypatch.setitem(sys.modules, "mauve", fake)

    out = tmp_path / "mauve.json"
    rc = mauve_mod.main(
        [
            "--samples",
            str(samples),
            "--refs-file",
            str(refs),
            "--featurize-model",
            "gpt2",
            "--device",
            "cpu",
            "--max-samples",
            "10",
            "--max-refs",
            "10",
            "--num-buckets",
            "8",
            "--json-out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["refs"].startswith("file:")
    assert payload["report"]["mauve"] == pytest.approx(0.77)
    assert "MAUVE=" in capsys.readouterr().out


def test_cli_requires_one_of_refs_sources(tmp_path):
    samples = tmp_path / "samples.jsonl"
    _write_samples(samples)
    # No refs source → argparse should fail.
    with pytest.raises(SystemExit):
        mauve_mod.main(["--samples", str(samples)])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
