"""Zero-shot MC-ELBO perplexity runner across held-out corpora.

Loads an ABD3 checkpoint once, then evaluates its importance-weighted
MC-ELBO perplexity on each dataset in the registry using the trained
tokenizer. This is the Phase 2.4 counterpart to :mod:`eval.perplexity`,
which only runs on the training corpus.

The registry is a flat table mapping short dataset ids (``wikitext103``,
``ptb``, ``lambada``, ``pg19``, ``arxiv``) to the HF Hub coordinates plus
the text column to read and the split to use for evaluation. Anything
that can be coerced into ``{"input_ids", "attention_mask"}`` with the
model's tokenizer works — adding a new corpus is a one-line entry here,
not a new Hydra yaml.

CLI (see ``python -m eval.zero_shot --help``)::

    python -m eval.zero_shot \\
        --checkpoint checkpoints/feasibility/epoch=43-step=4000.ckpt \\
        --datasets wikitext103 ptb lambada \\
        --n-samples 4 \\
        --device cuda \\
        --json-out report/zero_shot/feasibility.json

Library usage::

    from eval.zero_shot import run_zero_shot
    results = run_zero_shot(ckpt="ckpt.ckpt", datasets=["ptb"], n_samples=4)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from collections.abc import Sequence

import datasets as hf_datasets
import torch
import transformers

# Make ``from abd3 ...`` / ``from eval ...`` imports work when the module is
# invoked as a script (python eval/zero_shot.py) instead of via python -m.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.perplexity import (  # noqa: E402
    PPLResult,
    compute_perplexity,
    load_abd3_from_checkpoint,
    restore_live_params,
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """How to pull the *test* (or nearest equivalent) split from HF Hub.

    ``join_docs`` controls whether consecutive short rows (e.g. WT103 sentences)
    are concatenated into longer strings before tokenization — set True when
    the corpus is sentence-level and individual rows would mostly become
    padding after truncation.
    """

    name: str  # short registry id
    hf_name: str
    hf_subset: str | None
    split: str
    text_column: str = "text"
    join_docs: bool = False


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    # WikiText-103 raw-v1: the canonical long-context LM benchmark.
    "wikitext103": DatasetSpec(
        name="wikitext103",
        hf_name="wikitext",
        hf_subset="wikitext-103-raw-v1",
        split="test",
        text_column="text",
    ),
    # WikiText-2 raw-v1: small, mostly sanity-check.
    "wikitext2": DatasetSpec(
        name="wikitext2",
        hf_name="wikitext",
        hf_subset="wikitext-2-raw-v1",
        split="test",
        text_column="text",
    ),
    # Penn Treebank (text-only mirror of the classic WSJ split).
    "ptb": DatasetSpec(
        name="ptb",
        hf_name="ptb_text_only",
        hf_subset=None,
        split="test",
        text_column="sentence",
        join_docs=True,  # sentences are ~23 tokens; pack them.
    ),
    # LAMBADA-OpenAI: last-word cloze; for PPL we just score the full passage.
    "lambada": DatasetSpec(
        name="lambada",
        hf_name="EleutherAI/lambada_openai",
        hf_subset="default",
        split="test",
        text_column="text",
    ),
    # PG19: long-form book-length contexts.
    "pg19": DatasetSpec(
        name="pg19",
        hf_name="pg19",
        hf_subset=None,
        split="test",
        text_column="text",
    ),
    # SciPapers / arxiv: abstract+article; we use the abstract column as a
    # compromise between compute and coverage.
    "arxiv": DatasetSpec(
        name="arxiv",
        hf_name="scientific_papers",
        hf_subset="arxiv",
        split="test",
        text_column="abstract",
    ),
}


DEFAULT_DATASETS = ["wikitext103", "ptb", "lambada", "pg19", "arxiv"]


# ---------------------------------------------------------------------------
# Dataloader construction
# ---------------------------------------------------------------------------


def _load_split(spec: DatasetSpec) -> hf_datasets.Dataset:
    """Fetch a single split. Encapsulates the HF call so it's easy to stub."""
    kwargs: dict = {"trust_remote_code": True, "split": spec.split}
    if spec.hf_subset is not None:
        return hf_datasets.load_dataset(spec.hf_name, spec.hf_subset, **kwargs)
    return hf_datasets.load_dataset(spec.hf_name, **kwargs)


def _build_test_loader(
    spec: DatasetSpec,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Tokenize and wrap the test split in a DataLoader.

    The design mirrors ``abd3.dataloader.get_dataloaders`` (same filtering,
    same ``max_length`` / ``padding="max_length"`` choice) so the NLL is
    computed on exactly the same geometry the model trained under.
    """
    ds = _load_split(spec)

    # Drop rows whose raw text is essentially empty after stripping — matches
    # the training dataloader's policy so we're not scoring blank sequences.
    ds = ds.filter(lambda ex: len((ex[spec.text_column] or "").strip()) > 10)

    if spec.join_docs:
        # Pack short sentences into seq_len-sized strings. Keeps the same
        # number of training-time positions per row even for sentence-level
        # corpora like PTB.
        joined: list[str] = []
        buf: list[str] = []
        approx_budget = seq_len * 4  # rough chars/token; final truncation handles overrun
        running_chars = 0
        for row in ds:
            text = (row[spec.text_column] or "").strip()
            if not text:
                continue
            if running_chars + len(text) > approx_budget and buf:
                joined.append(" ".join(buf))
                buf, running_chars = [], 0
            buf.append(text)
            running_chars += len(text) + 1
        if buf:
            joined.append(" ".join(buf))
        ds = hf_datasets.Dataset.from_dict({"text": joined})
        text_column = "text"
    else:
        text_column = spec.text_column

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[text_column],
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

    tokenized = ds.map(
        _tokenize,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=1,
    )
    tokenized.set_format("torch")

    return torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ZeroShotEntry:
    """One (dataset, ppl) row inside the results table."""

    dataset: str
    hf_name: str
    hf_subset: str | None
    split: str
    ppl: float | None
    per_token_nll: float | None
    n_sequences: int
    tokens_per_pass: int
    wall_seconds: float
    error: str | None = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _row_from_result(spec: DatasetSpec, result: PPLResult) -> ZeroShotEntry:
    return ZeroShotEntry(
        dataset=spec.name,
        hf_name=spec.hf_name,
        hf_subset=spec.hf_subset,
        split=spec.split,
        ppl=result.ppl,
        per_token_nll=result.per_token_nll,
        n_sequences=result.n_sequences,
        tokens_per_pass=result.tokens_per_pass,
        wall_seconds=result.wall_seconds,
        error=None,
    )


def _row_from_error(spec: DatasetSpec, err: BaseException, wall: float) -> ZeroShotEntry:
    return ZeroShotEntry(
        dataset=spec.name,
        hf_name=spec.hf_name,
        hf_subset=spec.hf_subset,
        split=spec.split,
        ppl=None,
        per_token_nll=None,
        n_sequences=0,
        tokens_per_pass=0,
        wall_seconds=wall,
        error=f"{type(err).__name__}: {err}",
    )


def _format_table(entries: Sequence[ZeroShotEntry]) -> str:
    """Render the results as a pipe-delimited markdown table."""
    header = (
        "| dataset        | split | seqs | tokens/pass | ppl       | nll   | wall(s) | status   |"
    )
    sep = "|----------------|-------|------|-------------|-----------|-------|---------|----------|"
    rows = [header, sep]
    for e in entries:
        ppl_s = f"{e.ppl:9.2f}" if e.ppl is not None else "   (fail)"
        nll_s = f"{e.per_token_nll:5.3f}" if e.per_token_nll is not None else " -   "
        status = "ok" if e.error is None else e.error.split(":", 1)[0][:8]
        rows.append(
            f"| {e.dataset:<14} | {e.split:<5} | {e.n_sequences:>4} | "
            f"{e.tokens_per_pass:>11} | {ppl_s} | {nll_s} | {e.wall_seconds:7.1f} | {status:<8} |"
        )
    return "\n".join(rows)


def run_zero_shot(
    checkpoint: str,
    datasets: Sequence[str] | None = None,
    *,
    config_name: str = "feasibility",
    n_samples: int = 4,
    device: str = "cpu",
    use_ema: bool = True,
    max_batches: int | None = None,
    eval_batch_size: int | None = None,
    seed: int = 42,
    progress: bool = True,
) -> list[ZeroShotEntry]:
    """Evaluate ``checkpoint`` on each dataset in ``datasets``.

    A failure on one dataset (most commonly: HF Hub download hiccup or a
    dataset that got removed/renamed) is captured in the corresponding
    :class:`ZeroShotEntry.error` field and does NOT abort the run — the
    remaining datasets still get evaluated. Rationale: zero-shot sweeps
    are long, and losing 4/5 results because PG19 404'd is the worst UX.
    """
    datasets = list(datasets or DEFAULT_DATASETS)
    unknown = [d for d in datasets if d not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; known: {sorted(DATASET_REGISTRY)}")

    print(f"[zero-shot] loading checkpoint: {checkpoint}")
    model, cfg, tok = load_abd3_from_checkpoint(
        checkpoint,
        config_name=config_name,
        device=device,
        use_ema=use_ema,
    )
    model.config = cfg

    seq_len = int(cfg.model.length)
    bs = int(eval_batch_size if eval_batch_size is not None else cfg.loader.eval_batch_size)
    print(
        f"[zero-shot] seq_len={seq_len}  eval_batch_size={bs}  "
        f"n_samples={n_samples}  ema={'on' if use_ema else 'off'}  device={device}"
    )

    entries: list[ZeroShotEntry] = []
    try:
        for idx, dname in enumerate(datasets, 1):
            spec = DATASET_REGISTRY[dname]
            print(
                f"\n[zero-shot] ({idx}/{len(datasets)}) {spec.name}: {spec.hf_name}"
                + (f"/{spec.hf_subset}" if spec.hf_subset else "")
                + f"  split={spec.split}"
            )
            t0 = time.time()
            try:
                loader = _build_test_loader(spec, tok, seq_len, bs, num_workers=0)
                result = compute_perplexity(
                    model,
                    loader,
                    n_samples=n_samples,
                    device=device,
                    seed=seed,
                    max_batches=max_batches,
                    progress=progress,
                )
                result.checkpoint = checkpoint
                entries.append(_row_from_result(spec, result))
                print(
                    f"[zero-shot] {spec.name}: ppl={result.ppl:.2f}  "
                    f"nll={result.per_token_nll:.3f}  "
                    f"seqs={result.n_sequences}  "
                    f"[{result.wall_seconds:.1f}s]"
                )
            except Exception as exc:
                wall = time.time() - t0
                entries.append(_row_from_error(spec, exc, wall))
                print(f"[zero-shot] {spec.name}: FAILED after {wall:.1f}s: {exc}")
    finally:
        # Make sure any EMA swap-in is reversed even if we blew up mid-loop.
        restore_live_params(model)

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Zero-shot MC-ELBO PPL sweep for an ABD3 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file.")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=f"Subset to evaluate. Known: {sorted(DATASET_REGISTRY)}",
    )
    ap.add_argument(
        "--config-name",
        default="feasibility",
        help="Hydra config used to compose model/tokenizer (must match training).",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="MC-ELBO draws per dataset (zero-shot uses fewer than training PPL).",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        default=True,
        help="Swap EMA shadow params in for eval (the train-time default).",
    )
    ap.add_argument(
        "--no-ema", dest="use_ema", action="store_false", help="Evaluate the live (non-EMA) params."
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override cfg.loader.eval_batch_size (helpful on the cluster).",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap test batches per MC pass per dataset (smoke runs).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json-out", default=None, help="Optional path to dump the full results table as JSON."
    )
    ap.add_argument(
        "--markdown-out",
        default=None,
        help="Optional path to dump the results as a markdown table.",
    )
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    entries = run_zero_shot(
        checkpoint=args.checkpoint,
        datasets=args.datasets,
        config_name=args.config_name,
        n_samples=args.n_samples,
        device=args.device,
        use_ema=args.use_ema,
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        progress=args.progress,
    )

    print()
    print("=" * 100)
    print(_format_table(entries))
    print("=" * 100)

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "config_name": args.config_name,
            "n_samples": args.n_samples,
            "device": args.device,
            "use_ema": args.use_ema,
            "results": [e.as_dict() for e in entries],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"[zero-shot] wrote {out}")

    if args.markdown_out:
        out = pathlib.Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_format_table(entries) + "\n")
        print(f"[zero-shot] wrote {out}")

    # Non-zero exit if EVERYTHING failed (so CI/bash can notice).
    if entries and all(e.error is not None for e in entries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
