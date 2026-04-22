"""Tests for eval/efficiency.py.

We mock the model's ``sample()`` method so the benchmark's measurement
logic is exercised without any heavy inference. The mock returns a
deterministic (wall-time, NFE, per-block NFE) triple controlled by the
config, which lets us write exact assertions about timing aggregation,
NFE bookkeeping, and markdown rendering.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import efficiency  # noqa: E402

# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_parse_configs_produces_cartesian_product():
    points = efficiency.parse_configs(
        "adaptive=off,on ; block_size=4,8 ; batch_size=1 ; num_steps=10"
    )
    # 2 × 2 × 1 × 1 = 4
    assert len(points) == 4
    labels = [p.label() for p in points]
    assert any("astop=off bs=4" in label for label in labels)
    assert any("astop=on bs=8" in label for label in labels)


def test_parse_configs_rejects_missing_axis():
    with pytest.raises(ValueError, match="missing axes"):
        efficiency.parse_configs("adaptive=on ; block_size=4 ; batch_size=1")


def test_parse_configs_rejects_unknown_axis():
    with pytest.raises(ValueError, match="Unknown axis"):
        efficiency.parse_configs(
            "adaptive=on ; block_size=4 ; batch_size=1 ; num_steps=10 ; foo=bar"
        )


def test_parse_on_off_accepts_common_synonyms():
    assert efficiency._parse_on_off("on,off,true,false,1,0") == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_parse_on_off_rejects_garbage():
    with pytest.raises(ValueError):
        efficiency._parse_on_off("maybe")


# ---------------------------------------------------------------------------
# Benchmark measurement
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for ABD3Diffusion with a deterministic cost model.

    ``sample()`` sleeps for ``(num_steps × 0.001)`` seconds per call and
    returns ``(dummy_tokens, total_nfe, per_block_nfe)``. ``adaptive_stopping``
    toggling halves the per-block NFE, matching real behavior qualitatively
    without running any torch ops.
    """

    def __init__(self, num_tokens: int = 16) -> None:
        self.num_tokens = num_tokens
        self.adaptive_stopping = False
        self.last_kwargs: dict = {}

    def sample(self, *, n_samples, num_steps, block_size, track_nfe_per_block, progress):
        self.last_kwargs = dict(
            n_samples=n_samples,
            num_steps=num_steps,
            block_size=block_size,
            track_nfe_per_block=track_nfe_per_block,
            progress=progress,
        )
        # Deterministic tiny sleep → measurable, non-flaky wall-clock.
        time.sleep(0.002)
        num_blocks = self.num_tokens // block_size
        per_block = num_steps // 2 if self.adaptive_stopping else num_steps
        per_block_nfe = [per_block] * num_blocks
        total_nfe = sum(per_block_nfe)
        import torch

        dummy = torch.zeros(n_samples, self.num_tokens, dtype=torch.long)
        return dummy, total_nfe, per_block_nfe

    def eval(self):  # Lightning compatibility
        return self


@pytest.fixture
def fake_model():
    return _FakeModel(num_tokens=16)


def test_run_single_passes_point_through_to_sample(fake_model):
    point = efficiency.SweepPoint(
        adaptive_stopping=True,
        block_size=4,
        batch_size=2,
        num_steps=10,
    )
    wall, total_nfe, per_block, peak = efficiency.run_single(
        fake_model,
        point,
        device="cpu",
        seed=0,
    )
    assert fake_model.last_kwargs["n_samples"] == 2
    assert fake_model.last_kwargs["num_steps"] == 10
    assert fake_model.last_kwargs["block_size"] == 4
    assert fake_model.last_kwargs["track_nfe_per_block"] is True

    # adaptive on → each block uses 5 NFE; 16/4 = 4 blocks.
    assert per_block == [5, 5, 5, 5]
    assert total_nfe == 20
    assert wall > 0
    # CPU → peak memory is reported as 0.
    assert peak == 0


def test_run_single_restores_adaptive_flag(fake_model):
    fake_model.adaptive_stopping = False
    point = efficiency.SweepPoint(
        adaptive_stopping=True,
        block_size=4,
        batch_size=1,
        num_steps=4,
    )
    efficiency.run_single(fake_model, point, device="cpu")
    assert fake_model.adaptive_stopping is False  # restored


def test_benchmark_point_aggregates_over_repeats(fake_model):
    point = efficiency.SweepPoint(
        adaptive_stopping=False,
        block_size=4,
        batch_size=1,
        num_steps=10,
    )
    result = efficiency.benchmark_point(
        fake_model,
        point,
        device="cpu",
        repeat=3,
        warmup=1,
    )
    assert result.repeat == 3
    assert result.median_wall_seconds > 0
    assert (
        result.min_wall_seconds <= result.median_wall_seconds <= result.mean_wall_seconds
        or result.min_wall_seconds <= result.median_wall_seconds
    )
    # adaptive=False so no savings.
    assert result.total_nfe == 40  # 4 blocks × 10 steps
    assert result.theoretical_nfe == 40
    assert result.nfe_savings == 0.0
    assert result.seq_len == 16
    assert result.device == "cpu"


def test_benchmark_point_reports_nfe_savings_when_adaptive(fake_model):
    point = efficiency.SweepPoint(
        adaptive_stopping=True,
        block_size=4,
        batch_size=1,
        num_steps=10,
    )
    result = efficiency.benchmark_point(
        fake_model,
        point,
        device="cpu",
        repeat=2,
        warmup=0,
    )
    # adaptive halves per-block NFE → savings ≈ 0.5
    assert result.total_nfe == 20
    assert result.theoretical_nfe == 40
    assert result.nfe_savings == pytest.approx(0.5)
    assert result.per_block_nfe_mean == 5.0


def test_benchmark_point_throughput_is_positive(fake_model):
    point = efficiency.SweepPoint(
        adaptive_stopping=False,
        block_size=4,
        batch_size=8,
        num_steps=4,
    )
    r = efficiency.benchmark_point(
        fake_model,
        point,
        device="cpu",
        repeat=1,
        warmup=0,
    )
    assert r.tokens_per_second > 0


# ---------------------------------------------------------------------------
# Sweep + on_result callback
# ---------------------------------------------------------------------------


def test_run_sweep_invokes_callback_per_point(fake_model):
    pts = efficiency.parse_configs("adaptive=off,on ; block_size=4 ; batch_size=1 ; num_steps=4")
    seen: list = []
    results = efficiency.run_sweep(
        fake_model,
        pts,
        device="cpu",
        repeat=1,
        warmup=0,
        on_result=lambda r: seen.append(r),
    )
    assert len(results) == 2
    assert len(seen) == 2
    assert seen[0].adaptive_stopping is False
    assert seen[1].adaptive_stopping is True


# ---------------------------------------------------------------------------
# Rendering / IO
# ---------------------------------------------------------------------------


def test_render_markdown_contains_header_and_rows(fake_model):
    pts = efficiency.parse_configs("adaptive=off,on ; block_size=4 ; batch_size=1 ; num_steps=4")
    results = efficiency.run_sweep(fake_model, pts, device="cpu", repeat=1, warmup=0)
    md = efficiency.render_markdown(results)
    assert "| adaptive | block |" in md
    assert "astop=" not in md  # labels are per-column
    assert md.count("\n") >= 3  # header + separator + 2 rows


def test_render_markdown_handles_empty():
    assert "(no results)" in efficiency.render_markdown([])


def test_write_jsonl_roundtrip(tmp_path, fake_model):
    pts = efficiency.parse_configs("adaptive=off ; block_size=4 ; batch_size=1 ; num_steps=4")
    results = efficiency.run_sweep(fake_model, pts, device="cpu", repeat=1, warmup=0)
    out = tmp_path / "sweep.jsonl"
    efficiency.write_jsonl(out, results)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["adaptive_stopping"] is False
    assert rows[0]["block_size"] == 4


# ---------------------------------------------------------------------------
# CLI (mocking the checkpoint loader)
# ---------------------------------------------------------------------------


def test_cli_end_to_end_with_mocked_loader(tmp_path, fake_model, monkeypatch, capsys):
    def fake_loader(checkpoint, *, config_name, overrides, device, use_ema):
        return fake_model, SimpleNamespace(), SimpleNamespace()

    # The CLI imports lazily; patch in the right module.
    import eval.perplexity

    monkeypatch.setattr(eval.perplexity, "load_abd3_from_checkpoint", fake_loader)

    jsonl = tmp_path / "sweep.jsonl"
    md = tmp_path / "sweep.md"
    rc = efficiency.main(
        [
            "--checkpoint",
            "fake.ckpt",
            "--device",
            "cpu",
            "--configs",
            "adaptive=off,on ; block_size=4 ; batch_size=1 ; num_steps=4",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--jsonl-out",
            str(jsonl),
            "--markdown-out",
            str(md),
        ]
    )
    assert rc == 0
    assert jsonl.exists()
    assert md.exists()
    captured = capsys.readouterr()
    assert "sweep points" in captured.out
    assert "[done]" in captured.out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
