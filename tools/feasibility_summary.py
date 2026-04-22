"""Summarise the feasibility + ablation runs.

Reads the Lightning CSVLogger `metrics.csv` from `logs/version_*/`, matches
each to an experiment via `hparams.yaml:algo.name`, and prints a side-by-side
table of validation-loss trajectory + minimum. Also flags which `version_*`
directory belongs to which experiment so downstream plotting / reports stay
unambiguous.

Usage:
    python tools/feasibility_summary.py [--logs-dir logs]

No external deps beyond the standard library. Designed to be rerunnable as
experiments complete — each new `version_N` directory is picked up
automatically.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys
from collections import defaultdict

EXPERIMENT_ORDER = ["baseline", "rdr_only", "abd3"]


def parse_hparams_algo(path: pathlib.Path) -> str | None:
    """Return the value of `algo.name` from a Lightning hparams.yaml."""
    if not path.is_file():
        return None
    in_algo = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("algo:"):
            in_algo = True
            continue
        if in_algo:
            if stripped.startswith("name:"):
                return stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if line and not line.startswith(" ") and not line.startswith("\t"):
                in_algo = False
    return None


def load_curve(path: pathlib.Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (train_curve, val_curve) as lists of (step, loss)."""
    trains: list[tuple[int, float]] = []
    vals: list[tuple[int, float]] = []
    with path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                step = int(row["step"]) if row.get("step") else -1
            except ValueError:
                step = -1
            tl = row.get("train/loss")
            vl = row.get("val/loss")
            if tl:
                trains.append((step, float(tl)))
            if vl:
                vals.append((step, float(vl)))
    return trains, vals


def bucket(points: list[tuple[int, float]], width: int = 500) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for step, val in points:
        if step < 0:
            continue
        buckets[round(step / width) * width].append(val)
    return {b: statistics.mean(vs) for b, vs in buckets.items()}


def fmt(val: float | None, width: int = 7) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'—':>{width}}"
    return f"{val:>{width}.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    args = ap.parse_args()

    logs_dir = pathlib.Path(args.logs_dir)
    found: dict[str, tuple[pathlib.Path, list, list]] = {}

    for vdir in sorted(logs_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[1])):
        metrics = vdir / "metrics.csv"
        hparams = vdir / "hparams.yaml"
        if not metrics.is_file():
            print(f"[skip] {vdir.name}: no metrics.csv", file=sys.stderr)
            continue
        algo = parse_hparams_algo(hparams) or "unknown"
        tr, va = load_curve(metrics)
        if algo in found:
            print(
                f"[warn] {algo} already seen ({found[algo][0].name}); ignoring {vdir.name}",
                file=sys.stderr,
            )
            continue
        found[algo] = (vdir, tr, va)

    if not found:
        print("No completed runs found under", logs_dir)
        return 1

    # Ordered: baseline, rdr_only, abd3 — missing ones are shown as '(pending)'.
    ordered = [(algo, found.get(algo)) for algo in EXPERIMENT_ORDER if True]

    print()
    print("=" * 88)
    print(f"{'ABD3 feasibility suite — validation loss trajectory':^88}")
    print("=" * 88)
    print()
    print(
        f"{'run':<28}{'version':<12}{'#train':>8}{'#val':>8}{'tr@5k':>9}{'val@1k':>9}{'val@3k':>9}{'min_val':>10}"
    )
    print("-" * 88)

    for algo, entry in ordered:
        label = {
            "abd3": "feasibility (full ABD3)",
            "baseline": "ablation_baseline (BD3-LMs)",
            "rdr_only": "ablation_rdr_only (RDR only)",
        }.get(algo, algo)
        if entry is None:
            print(f"{label:<28}{'(pending)':<12}")
            continue
        vdir, tr, va = entry

        def nearest(points, target):
            if not points:
                return None
            return min(points, key=lambda x: abs(x[0] - target))[1]

        min_val = min(v for _, v in va) if va else None
        print(
            f"{label:<28}{vdir.name:<12}{len(tr):>8}{len(va):>8}"
            f"{fmt(nearest(tr, 5000), 9)}{fmt(nearest(va, 1000), 9)}"
            f"{fmt(nearest(va, 3000), 9)}{fmt(min_val, 10)}"
        )

    print()
    print("-" * 88)
    print("val/loss trajectory (bucketed per 500 steps)")
    print("-" * 88)
    for algo, entry in ordered:
        if entry is None:
            continue
        vdir, tr, va = entry
        print(f"\n  {algo}  ({vdir.name}):")
        buckets = bucket(va, 500)
        for step in sorted(buckets):
            v = buckets[step]
            ppl = math.exp(v)
            print(f"    step~{step:>5}   val_loss={v:.3f}   ppl≈{ppl:>7.1f}")

    # Headline comparison
    if "abd3" in found and "baseline" in found:
        _, _, va_a = found["abd3"]
        _, _, va_b = found["baseline"]
        min_a = min(v for _, v in va_a)
        min_b = min(v for _, v in va_b)
        print()
        print("-" * 88)
        print(f"  ABD3 min val loss     : {min_a:.3f}  (ppl≈{math.exp(min_a):.1f})")
        print(f"  baseline min val loss : {min_b:.3f}  (ppl≈{math.exp(min_b):.1f})")
        print(f"  delta (ABD3 - base)   : {min_a - min_b:+.3f} nats")
        if "rdr_only" in found:
            _, _, va_r = found["rdr_only"]
            min_r = min(v for _, v in va_r)
            print(f"  rdr_only min val loss : {min_r:.3f}  (ppl≈{math.exp(min_r):.1f})")
            print(f"  delta (rdr - base)    : {min_r - min_b:+.3f} nats")
            print(f"  delta (abd3 - rdr)    : {min_a - min_r:+.3f} nats")
        else:
            print("  rdr_only min val loss : (pending)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
