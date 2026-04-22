"""Block-autoregressive generation with per-block NFE logging.

Thin CLI + library wrapper around ``ABD3Diffusion.sample``. Emits a JSONL file
where each line is a single generation record.

Both the semi-AR (``block_size < model.length``) and one-shot diffusion
(``block_size == model.length``) paths of ``ABD3Diffusion.sample`` are
exercised; regression coverage lives in ``tests/test_sampler.py``.

    {
      "sample_id":       0,
      "text":            "<decoded sample>",
      "token_ids":       [...],
      "block_size":      4,
      "num_steps":       10,
      "total_nfe":       23,             # sum of per_block_nfe
      "per_block_nfe":   [10, 4, 3, ...],
      "theoretical_nfe": 40,              # num_blocks * num_steps (no early stop)
      "nfe_savings":     0.425,           # 1 - total / theoretical
      "elapsed_seconds": 12.4,
      "seed":            42,
      "checkpoint":      "checkpoints/feasibility/...",
      "algo":            "abd3",
      "sampler":         "ddpm_self_cond"
    }

The per-block NFE + savings columns are what make the efficiency claims
("RDR cuts NFEs by N%") concrete. Downstream ``eval/quality.py`` (next) will
consume these JSONL files to compute MAUVE / Gen-PPL / distinct-n / self-BLEU.

CLI:

    python -m eval.generate \\
        --checkpoint checkpoints/feasibility/epoch=43-step=4000.ckpt \\
        --n-samples 4 --num-steps 10 --device cpu \\
        --out report/samples/feasibility_step4000.jsonl

Library:

    from eval.generate import generate
    records = generate(model, tokenizer, n_samples=4, num_steps=10, seed=42)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
import time
from typing import Optional

import torch
import transformers

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abd3.diffusion import ABD3Diffusion  # noqa: E402
from eval.perplexity import load_abd3_from_checkpoint  # noqa: E402


@dataclasses.dataclass
class SampleRecord:
    sample_id: int
    text: str
    token_ids: list[int]
    block_size: int
    num_steps: int
    total_nfe: int
    per_block_nfe: list[int]
    theoretical_nfe: int
    nfe_savings: float
    elapsed_seconds: float
    seed: int
    checkpoint: str
    algo: str
    sampler: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@torch.no_grad()
def generate(
    model: ABD3Diffusion,
    tokenizer: transformers.PreTrainedTokenizer,
    *,
    n_samples: int = 4,
    num_steps: Optional[int] = None,
    block_size: Optional[int] = None,
    seed: int = 42,
    progress: bool = False,
    checkpoint_path: str = "",
    algo_name: str = "",
) -> list[SampleRecord]:
    """Generate ``n_samples`` sequences and return structured records.

    Per-block NFE is captured so efficiency comparisons (adaptive stopping
    on/off) are apples-to-apples. Theoretical NFE assumes no early exit
    (``num_blocks * num_steps``); actual NFE sums ``per_block_nfe``.
    """
    torch.manual_seed(seed)

    effective_num_steps = num_steps if num_steps is not None else model.T
    effective_block_size = block_size if block_size is not None else model.block_size
    num_blocks = model.num_tokens // effective_block_size
    theoretical = num_blocks * effective_num_steps

    t0 = time.time()
    x, total_nfe, per_block_nfe = model.sample(
        n_samples=n_samples,
        num_steps=effective_num_steps,
        block_size=effective_block_size,
        track_nfe_per_block=True,
        progress=progress,
    )
    elapsed = time.time() - t0

    texts = tokenizer.batch_decode(x.cpu().tolist(), skip_special_tokens=False)

    # Per-block NFE is aggregated across the whole batch (the sampler doesn't
    # branch per-sequence), so every record in this batch shares the same
    # NFE trace. We still emit it per-record for downstream joins.
    sampler_name = (
        "ddpm_self_cond"
        if getattr(model, "self_conditioning", False)
        else "ddpm"
    )
    if getattr(model, "adaptive_stopping", False):
        sampler_name += "+adaptive_stop"

    records: list[SampleRecord] = []
    for i, (text, token_row) in enumerate(zip(texts, x.cpu().tolist())):
        records.append(SampleRecord(
            sample_id=i,
            text=text,
            token_ids=list(token_row),
            block_size=int(effective_block_size),
            num_steps=int(effective_num_steps),
            total_nfe=int(sum(per_block_nfe)),
            per_block_nfe=list(per_block_nfe),
            theoretical_nfe=int(theoretical),
            nfe_savings=(
                1.0 - sum(per_block_nfe) / theoretical if theoretical > 0 else 0.0
            ),
            elapsed_seconds=elapsed,
            seed=seed,
            checkpoint=checkpoint_path,
            algo=algo_name,
            sampler=sampler_name,
        ))

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate samples from an ABD3 checkpoint with per-block NFE logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", default="feasibility")
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="Hydra-style overrides, e.g. algo=baseline block_size=8.")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=None,
                    help="Denoising steps per block (default: model.T from config).")
    ap.add_argument("--block-size", type=int, default=None,
                    help="Block size for generation (default: model.block_size).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Output JSONL path. If omitted, only prints to stdout.")
    ap.add_argument("--print-samples", action="store_true",
                    help="Print each decoded sample (truncated) to stdout.")
    return ap


def _format_summary(records: list[SampleRecord]) -> str:
    if not records:
        return "(no records)"
    totals = [r.total_nfe for r in records]
    theo = records[0].theoretical_nfe
    return (
        f"n_samples={len(records)}  block_size={records[0].block_size}  "
        f"num_steps={records[0].num_steps}  "
        f"total_nfe={sum(totals) // len(totals)} (mean)  "
        f"theoretical={theo}  "
        f"savings={records[0].nfe_savings * 100:.1f}%  "
        f"elapsed={records[0].elapsed_seconds:.1f}s  "
        f"sampler={records[0].sampler}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    print(f"[gen] loading checkpoint: {args.checkpoint}")
    model, cfg, tok = load_abd3_from_checkpoint(
        args.checkpoint,
        config_name=args.config_name,
        overrides=args.overrides,
        device=args.device,
        use_ema=args.use_ema,
    )
    algo_name = getattr(cfg.algo, "name", "unknown") if hasattr(cfg, "algo") else "unknown"
    print(
        f"[gen] algo={algo_name}  block_size={cfg.block_size}  "
        f"T={model.T}  adaptive_stop={model.adaptive_stopping}  "
        f"self_cond={model.self_conditioning}  device={args.device}"
    )

    records = generate(
        model, tok,
        n_samples=args.n_samples,
        num_steps=args.num_steps,
        block_size=args.block_size,
        seed=args.seed,
        progress=True,
        checkpoint_path=args.checkpoint,
        algo_name=algo_name,
    )

    print("\n" + "=" * 100)
    print(_format_summary(records))
    print("=" * 100)

    if args.print_samples:
        for r in records:
            preview = r.text.replace("\n", " ")[:180]
            print(f"\n--- sample {r.sample_id}  NFE={r.total_nfe}/"
                  f"{r.theoretical_nfe} ({r.nfe_savings * 100:.0f}% saved) ---")
            print(preview + ("…" if len(r.text) > 180 else ""))

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in records:
                f.write(json.dumps(r.as_dict()) + "\n")
        print(f"\n[gen] wrote {len(records)} records to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
