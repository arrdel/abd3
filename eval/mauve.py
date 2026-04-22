"""MAUVE divergence between generated and reference texts.

MAUVE (Pillutla et al., NeurIPS 2021) measures the gap between two text
distributions — typically *model generations* vs *human references* — by:

1. Embedding each text with a strong LM (default: ``gpt2-large``).
2. Clustering the joint embedding space into K centroids.
3. Computing the area between the KL(P‖M) and KL(Q‖M) divergence curves
   as cluster counts are swept.

The result is a single scalar in ``[0, 1]`` where higher = more similar.
It's the most standard "holistic sample quality" metric used by
diffusion-LM papers (SSD-LM, DiffuSeq, BD3-LMs), so it belongs in our
eval suite.

Implementation strategy
-----------------------
We *wrap* the excellent ``mauve-text`` package rather than re-implement
the clustering and divergence integration. That package is optional:
we import it lazily inside :func:`compute_mauve` and emit a clear
install hint if it's missing, so the rest of the eval suite doesn't
break on hosts that skipped ``pip install mauve-text``.

Input
-----
A generated JSONL (from :mod:`eval.generate`) **and** a reference text
source — either a file of one text per line, or a HF ``datasets``-style
name (e.g. ``wikitext|wikitext-103-raw-v1|validation|text``). The
reference is lazily materialised; only the first ``--max-refs`` non-empty
rows are used.

CLI::

    python -m eval.mauve \\
        --samples report/samples/feasibility_step4000.jsonl \\
        --refs-hf wikitext|wikitext-103-raw-v1|validation|text \\
        --max-samples 500 --max-refs 500 \\
        --featurize-model gpt2 --device cuda \\
        --json-out report/mauve/feasibility_step4000.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections.abc import Sequence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_MAUVE_INSTALL_HINT = (
    "MAUVE requires the `mauve-text` package. Install with:\n"
    "    pip install mauve-text\n"
    "Then re-run this command."
)


# ---------------------------------------------------------------------------
# Reference-loading helpers
# ---------------------------------------------------------------------------


def load_refs_from_file(path: str | pathlib.Path, *, max_refs: int | None) -> list[str]:
    """One reference per line; blank lines are skipped.

    We purposefully don't apply any normalisation here — MAUVE's featurizer
    will tokenize its way, and caller-provided refs already embody whatever
    cleaning policy the user wants.
    """
    refs: list[str] = []
    with pathlib.Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            refs.append(line)
            if max_refs is not None and len(refs) >= max_refs:
                break
    return refs


def load_refs_from_hf(spec: str, *, max_refs: int | None = None) -> list[str]:
    """Parse a ``dataset|subset|split|column`` spec and return ``max_refs`` rows.

    Use ``-`` to skip ``subset``:

        wikitext|wikitext-103-raw-v1|validation|text
        pg19|-|test|text

    We stream rows and bail as soon as ``max_refs`` non-empty ones are
    collected, so you don't pay the cost of loading the whole split.
    """
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError(f"--refs-hf must be 'dataset|subset|split|column'; got {spec!r}")
    name, subset, split, column = parts
    if subset in {"", "-", "none", "None"}:
        subset = None

    try:
        import datasets
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Loading HF refs requires `datasets`. Install with `pip install datasets`."
        ) from e

    ds = datasets.load_dataset(name, subset, split=split, streaming=True)
    refs: list[str] = []
    for row in ds:
        txt = row.get(column)
        if not isinstance(txt, str) or not txt.strip():
            continue
        refs.append(txt)
        if max_refs is not None and len(refs) >= max_refs:
            break
    return refs


# ---------------------------------------------------------------------------
# MAUVE wrapper
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MauveReport:
    mauve: float  # the scalar in [0, 1] — higher = closer to ref.
    frontier_integral: float  # the "area" scalar (intermediate, sometimes reported).
    featurize_model: str
    n_samples: int
    n_refs: int
    max_text_length: int
    scaling_factor: float
    cluster_k: int
    seed: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        return (
            f"MAUVE={self.mauve:.4f}  "
            f"FI={self.frontier_integral:.4f}  "
            f"(n_gen={self.n_samples} n_ref={self.n_refs}, "
            f"feat={self.featurize_model}, K={self.cluster_k})"
        )


def compute_mauve(
    gen_texts: Sequence[str],
    ref_texts: Sequence[str],
    *,
    featurize_model: str = "gpt2-large",
    device_id: int = -1,  # -1 = CPU (mauve's convention).
    max_text_length: int = 1024,
    scaling_factor: float = 5.0,
    mauve_scaling_factor: float | None = None,
    num_buckets: int | str = "auto",
    seed: int = 42,
    verbose: bool = False,
) -> MauveReport:
    """Thin wrapper around ``mauve.compute_mauve`` that returns our report shape.

    Lazy-imports ``mauve`` so absence doesn't crash module import. We
    preserve every parameter the user might want to tune for reproducing
    published numbers; defaults follow the Pillutla et al. paper:
    GPT-2 Large features, scaling factor 5, auto-selected K.
    """
    try:
        import mauve  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(_MAUVE_INSTALL_HINT) from e

    if not gen_texts:
        raise ValueError("`gen_texts` is empty")
    if not ref_texts:
        raise ValueError("`ref_texts` is empty")

    # mauve_scaling_factor is an alias for scaling_factor in newer versions;
    # pass through whichever the caller supplied.
    kwargs = dict(
        p_text=list(gen_texts),
        q_text=list(ref_texts),
        device_id=device_id,
        max_text_length=max_text_length,
        num_buckets=num_buckets,
        seed=seed,
        verbose=verbose,
        featurize_model_name=featurize_model,
    )
    # Newer mauve uses `mauve_scaling_factor`; older uses `scaling_factor`.
    if mauve_scaling_factor is not None:
        kwargs["mauve_scaling_factor"] = mauve_scaling_factor
    else:
        kwargs["mauve_scaling_factor"] = scaling_factor

    out = mauve.compute_mauve(**kwargs)

    return MauveReport(
        mauve=float(out.mauve),
        frontier_integral=float(out.frontier_integral),
        featurize_model=featurize_model,
        n_samples=len(gen_texts),
        n_refs=len(ref_texts),
        max_text_length=int(max_text_length),
        scaling_factor=float(scaling_factor),
        cluster_k=int(getattr(out, "num_buckets", -1)),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Samples loader (re-export for uniform imports across eval/)
# ---------------------------------------------------------------------------


def load_samples(path: str | pathlib.Path) -> list[dict]:
    from eval.diversity import load_samples as _load

    return _load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MAUVE divergence between generated samples and a reference set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--samples", required=True, help="Path to a SampleRecord JSONL.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--refs-file", default=None, help="Plaintext file, one reference per line.")
    grp.add_argument(
        "--refs-hf",
        default=None,
        help="HF spec 'dataset|subset|split|column' (use '-' for no subset).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Cap on generated samples actually fed to MAUVE.",
    )
    ap.add_argument("--max-refs", type=int, default=1000, help="Cap on reference samples.")
    ap.add_argument(
        "--featurize-model",
        default="gpt2-large",
        help="HF hub id for the featurizer LM (gpt2 / gpt2-large / etc.).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="CPU is slow; use cuda when available.",
    )
    ap.add_argument("--max-text-length", type=int, default=1024)
    ap.add_argument("--scaling-factor", type=float, default=5.0)
    ap.add_argument(
        "--num-buckets", default="auto", help="Cluster count K; 'auto' lets MAUVE pick sqrt(N)."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    records = load_samples(args.samples)
    if not records:
        print(f"[mauve] {args.samples}: empty; nothing to score", file=sys.stderr)
        return 1

    gen_texts = [r.get("text", "") for r in records if r.get("text", "").strip()]
    if args.max_samples and len(gen_texts) > args.max_samples:
        gen_texts = gen_texts[: args.max_samples]
    if not gen_texts:
        print("[mauve] all generated texts are empty", file=sys.stderr)
        return 1

    if args.refs_file:
        refs = load_refs_from_file(args.refs_file, max_refs=args.max_refs)
    else:
        refs = load_refs_from_hf(args.refs_hf, max_refs=args.max_refs)
    if not refs:
        print("[mauve] no usable references found", file=sys.stderr)
        return 1

    # mauve uses GPU id convention: -1 for CPU, else device ordinal.
    import os

    if args.device == "cuda":
        device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    else:
        device_id = -1

    num_buckets: int | str
    try:
        num_buckets = int(args.num_buckets)
    except ValueError:
        num_buckets = args.num_buckets  # 'auto'

    print(
        f"[mauve] scoring {len(gen_texts)} gens vs {len(refs)} refs " f"with {args.featurize_model}"
    )
    report = compute_mauve(
        gen_texts,
        refs,
        featurize_model=args.featurize_model,
        device_id=device_id,
        max_text_length=args.max_text_length,
        scaling_factor=args.scaling_factor,
        num_buckets=num_buckets,
        seed=args.seed,
    )

    print("=" * 100)
    print(report.summary_line())
    print("=" * 100)

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "samples_path": str(args.samples),
                    "refs": (f"file:{args.refs_file}" if args.refs_file else f"hf:{args.refs_hf}"),
                    "report": report.as_dict(),
                },
                indent=2,
            )
        )
        print(f"[mauve] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
