"""Sampler efficiency benchmark: wall-clock / peak memory / NFE.

What we're measuring
--------------------
ABD3 claims two efficiency wins vs vanilla block-diffusion:

1. **Adaptive stopping** — cuts NFEs (number of function evaluations) by
   short-circuiting per-block denoising once the argmax stops changing.
2. **Two-stream DiT + RDR** — the added machinery shouldn't meaningfully
   increase per-step cost; we need numbers to defend that claim.

This module sweeps ``sample()`` across axes we care about — adaptive
on/off, block size, batch size, per-block step budget — and for each
config records:

    * wall-clock seconds (median of ``repeat`` runs, excluding warmup)
    * peak CUDA memory (``torch.cuda.max_memory_allocated``)
    * total NFE (sum over per-block NFE counters)
    * mean + std of per-block NFE
    * tokens/sec throughput (batch × seq_len / wall-clock)

Output is a pandas-compatible JSONL, plus a rendered Markdown table for
the paper appendix.

CLI::

    python -m eval.efficiency \\
        --checkpoint checkpoints/feasibility/epoch=43-step=4000.ckpt \\
        --device cuda \\
        --configs "adaptive=off,on ; block_size=4,8 ; batch_size=4,16 ; num_steps=10,20" \\
        --repeat 3 --warmup 1 \\
        --jsonl-out report/efficiency/feasibility_step4000.jsonl \\
        --markdown-out report/efficiency/feasibility_step4000.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
import time
from collections.abc import Iterable

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SweepPoint:
    """One row of the sweep grid — exactly the axes we vary."""

    adaptive_stopping: bool
    block_size: int
    batch_size: int
    num_steps: int

    def label(self) -> str:
        ast = "on" if self.adaptive_stopping else "off"
        return f"astop={ast} bs={self.block_size} n={self.batch_size} T={self.num_steps}"


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_on_off(s: str) -> list[bool]:
    out: list[bool] = []
    for part in s.split(","):
        v = part.strip().lower()
        if v in {"on", "true", "1", "yes"}:
            out.append(True)
        elif v in {"off", "false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse on/off from {part!r}; use on|off")
    return out


def parse_configs(spec: str) -> list[SweepPoint]:
    """Parse ``adaptive=...; block_size=...; batch_size=...; num_steps=...``.

    Axes are separated by ``;``; values within an axis by ``,``. The Cartesian
    product is returned in axis order (adaptive × block_size × batch_size ×
    num_steps), which happens to be a sensible group-by for tables.
    """
    axes: dict[str, list] = {}
    for raw in spec.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Bad axis {raw!r}; expected key=values")
        key, values = raw.split("=", 1)
        key = key.strip()
        if key == "adaptive":
            axes[key] = _parse_on_off(values)
        elif key in {"block_size", "batch_size", "num_steps"}:
            axes[key] = _parse_int_list(values)
        else:
            raise ValueError(f"Unknown axis {key!r}")
    # Require all four; defaults aren't ambiguous enough to be safe silently.
    required = {"adaptive", "block_size", "batch_size", "num_steps"}
    missing = required - set(axes.keys())
    if missing:
        raise ValueError(f"--configs missing axes: {sorted(missing)}")

    points: list[SweepPoint] = []
    for adaptive in axes["adaptive"]:
        for bs in axes["block_size"]:
            for b in axes["batch_size"]:
                for t in axes["num_steps"]:
                    points.append(
                        SweepPoint(
                            adaptive_stopping=bool(adaptive),
                            block_size=int(bs),
                            batch_size=int(b),
                            num_steps=int(t),
                        )
                    )
    return points


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


def _sync(device: str | torch.device) -> None:
    if isinstance(device, torch.device):
        is_cuda = device.type == "cuda"
    else:
        is_cuda = str(device).startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory_stats(device: str | torch.device) -> None:
    if torch.cuda.is_available() and (
        isinstance(device, torch.device) and device.type == "cuda" or str(device).startswith("cuda")
    ):
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_bytes(device: str | torch.device) -> int:
    if torch.cuda.is_available() and (
        isinstance(device, torch.device) and device.type == "cuda" or str(device).startswith("cuda")
    ):
        return int(torch.cuda.max_memory_allocated())
    return 0


@dataclasses.dataclass
class SweepResult:
    """Everything we record for one SweepPoint (post-aggregation over repeats)."""

    adaptive_stopping: bool
    block_size: int
    batch_size: int
    num_steps: int
    seq_len: int
    repeat: int
    median_wall_seconds: float
    min_wall_seconds: float
    mean_wall_seconds: float
    std_wall_seconds: float
    total_nfe: int  # from the last repeat; NFE is deterministic given inputs
    theoretical_nfe: int
    nfe_savings: float
    per_block_nfe_mean: float
    per_block_nfe_std: float
    peak_memory_mib: float
    tokens_per_second: float
    device: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@torch.no_grad()
def run_single(
    model,
    point: SweepPoint,
    *,
    device: str | torch.device,
    seed: int = 42,
) -> tuple[float, int, list[int], int]:
    """Run one ``sample()`` call at ``point`` and return measurements.

    Returns ``(wall_seconds, total_nfe, per_block_nfe, peak_memory_bytes)``.

    Toggling ``adaptive_stopping`` is done on the model in-place because it's
    the officially supported lever (matches how training configs wire it).
    """
    prev_astop = model.adaptive_stopping
    try:
        model.adaptive_stopping = point.adaptive_stopping
        torch.manual_seed(seed)

        _reset_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        _, total_nfe, per_block_nfe = model.sample(
            n_samples=point.batch_size,
            num_steps=point.num_steps,
            block_size=point.block_size,
            track_nfe_per_block=True,
            progress=False,
        )
        _sync(device)
        wall = time.perf_counter() - t0
        peak = _peak_memory_bytes(device)
        return wall, int(total_nfe), list(per_block_nfe), peak
    finally:
        model.adaptive_stopping = prev_astop


def benchmark_point(
    model,
    point: SweepPoint,
    *,
    device: str | torch.device,
    repeat: int = 3,
    warmup: int = 1,
    seed: int = 42,
) -> SweepResult:
    """Run ``warmup + repeat`` times; aggregate wall-clock / NFE / memory."""
    # Warmup (JIT, cuDNN selection, allocator warm-up, etc.).
    for _ in range(warmup):
        run_single(model, point, device=device, seed=seed)

    walls: list[float] = []
    peak_bytes = 0
    total_nfe = 0
    per_block_nfe: list[int] = []
    for i in range(repeat):
        wall, nfe, pb, peak = run_single(model, point, device=device, seed=seed + i)
        walls.append(wall)
        total_nfe = nfe
        per_block_nfe = pb
        peak_bytes = max(peak_bytes, peak)

    seq_len = int(getattr(model, "num_tokens", point.block_size * (len(per_block_nfe) or 1)))
    num_blocks = max(1, seq_len // point.block_size)
    theoretical = num_blocks * point.num_steps

    # per_block_nfe may be empty if batch_size=0 etc.; guard.
    pb_mean = statistics.fmean(per_block_nfe) if per_block_nfe else 0.0
    pb_std = statistics.pstdev(per_block_nfe) if len(per_block_nfe) > 1 else 0.0

    median_wall = statistics.median(walls)
    tokens = point.batch_size * seq_len
    tps = tokens / median_wall if median_wall > 0 else 0.0

    return SweepResult(
        adaptive_stopping=point.adaptive_stopping,
        block_size=point.block_size,
        batch_size=point.batch_size,
        num_steps=point.num_steps,
        seq_len=seq_len,
        repeat=repeat,
        median_wall_seconds=median_wall,
        min_wall_seconds=min(walls),
        mean_wall_seconds=statistics.fmean(walls),
        std_wall_seconds=(statistics.pstdev(walls) if len(walls) > 1 else 0.0),
        total_nfe=int(total_nfe),
        theoretical_nfe=int(theoretical),
        nfe_savings=(1.0 - total_nfe / theoretical) if theoretical > 0 else 0.0,
        per_block_nfe_mean=float(pb_mean),
        per_block_nfe_std=float(pb_std),
        peak_memory_mib=peak_bytes / (1024 * 1024),
        tokens_per_second=float(tps),
        device=str(device),
    )


def run_sweep(
    model,
    points: Iterable[SweepPoint],
    *,
    device: str | torch.device,
    repeat: int = 3,
    warmup: int = 1,
    seed: int = 42,
    on_result=None,
) -> list[SweepResult]:
    """Run every ``SweepPoint`` sequentially; optionally stream results."""
    results: list[SweepResult] = []
    for p in points:
        res = benchmark_point(model, p, device=device, repeat=repeat, warmup=warmup, seed=seed)
        results.append(res)
        if on_result is not None:
            on_result(res)
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(results: list[SweepResult]) -> str:
    """Pretty Markdown table, grouped by adaptive on/off for easy before/after."""
    if not results:
        return "_(no results)_\n"
    header = (
        "| adaptive | block | batch | T | wall-s (med) | tok/s | NFE | NFE/theo | peak-MiB |\n"
        "|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in results:
        ast = "on" if r.adaptive_stopping else "off"
        ratio = (r.total_nfe / r.theoretical_nfe) if r.theoretical_nfe > 0 else 0.0
        lines.append(
            f"| {ast} | {r.block_size} | {r.batch_size} | {r.num_steps} | "
            f"{r.median_wall_seconds:.3f} | {r.tokens_per_second:.1f} | "
            f"{r.total_nfe} | {ratio * 100:.1f}% | {r.peak_memory_mib:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_jsonl(path: str | pathlib.Path, results: list[SweepResult]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in results:
            f.write(json.dumps(r.as_dict()) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Benchmark ABD3 sampler across configs (wall-clock / NFE / memory).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="feasibility")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    ap.add_argument(
        "--configs",
        default="adaptive=off,on ; block_size=4 ; batch_size=4 ; num_steps=10,20",
        help="Axes spec; see module docstring.",
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jsonl-out", default=None)
    ap.add_argument("--markdown-out", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    points = parse_configs(args.configs)
    print(f"[efficiency] parsed {len(points)} sweep points:")
    for p in points:
        print(f"  - {p.label()}")

    # Lazy import so unit tests can import this module without the full stack.
    from eval.perplexity import load_abd3_from_checkpoint

    print(f"[efficiency] loading {args.checkpoint}")
    model, _cfg, _tok = load_abd3_from_checkpoint(
        args.checkpoint,
        config_name=args.config_name,
        overrides=args.overrides,
        device=args.device,
        use_ema=args.use_ema,
    )
    model.eval()

    def _on_result(r: SweepResult) -> None:
        ast = "on " if r.adaptive_stopping else "off"
        print(
            f"  [done] astop={ast} bs={r.block_size:>3} n={r.batch_size:>3} "
            f"T={r.num_steps:>3}  "
            f"wall={r.median_wall_seconds:.3f}s  NFE={r.total_nfe}  "
            f"peak={r.peak_memory_mib:.1f}MiB  tok/s={r.tokens_per_second:.1f}"
        )

    results = run_sweep(
        model,
        points,
        device=args.device,
        repeat=args.repeat,
        warmup=args.warmup,
        seed=args.seed,
        on_result=_on_result,
    )

    md = render_markdown(results)
    print("\n" + md)

    if args.jsonl_out:
        write_jsonl(args.jsonl_out, results)
        print(f"[efficiency] wrote {args.jsonl_out}")
    if args.markdown_out:
        p = pathlib.Path(args.markdown_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md)
        print(f"[efficiency] wrote {args.markdown_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
