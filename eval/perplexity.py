"""Importance-weighted MC-ELBO perplexity for ABD3 checkpoints.

Matched to the BD3-LMs eval protocol so every number is directly comparable
across the abd3 / rdr_only / baseline ablation arms and against published
BD3-LMs results:

  per_token_nll =  (1 / n_samples) * sum_i [ sum_{seq,tok} nll_i / tokens_i ]
  ppl            =  exp(per_token_nll)

Each of the ``n_samples`` Monte-Carlo passes draws a fresh (antithetic) t-grid
per sequence, so per-sequence variance is controlled the same way the training
loss is; averaging across passes gives the low-variance ELBO estimator used in
the BD3-LMs paper.

CLI (see ``python -m eval.perplexity --help``):

    python -m eval.perplexity \\
        --checkpoint checkpoints/feasibility/epoch=43-step=4000.ckpt \\
        --n-samples 8 \\
        --device cpu

Library usage:

    from eval.perplexity import load_abd3_from_checkpoint, compute_perplexity
    model, cfg, tok = load_abd3_from_checkpoint("ckpt.ckpt", device="cpu")
    _, val_loader = abd3.dataloader.get_dataloaders(cfg, tok)
    res = compute_perplexity(model, val_loader, n_samples=8, device="cpu")
    print(res.ppl)
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import pathlib
import sys
import time
from typing import Iterable, Iterator, Optional

import torch
import transformers
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

# Make `from abd3 ...` imports work regardless of invocation path (module vs
# script). Keep this block before any project-local imports.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abd3 import dataloader as abd3_dataloader  # noqa: E402
from abd3.diffusion import ABD3Diffusion  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint / config plumbing
# ---------------------------------------------------------------------------


def _compose_config(
    config_name: str = "feasibility",
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Re-compose the Hydra config that was used to train the checkpoint.

    We do it from scratch rather than relying on the checkpoint's pickled
    hparams because (a) it lets the caller override fields (e.g. swap the
    dataset to PTB for zero-shot PPL) and (b) it survives config-schema
    evolution — as long as keys we read here still exist.
    """
    config_dir = str(_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def _build_tokenizer(cfg: DictConfig) -> transformers.PreTrainedTokenizer:
    tok = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_abd3_from_checkpoint(
    checkpoint_path: str,
    config_name: str = "feasibility",
    overrides: Optional[list[str]] = None,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[ABD3Diffusion, DictConfig, transformers.PreTrainedTokenizer]:
    """Load a trained ABD3Diffusion Lightning checkpoint ready for eval.

    Returns (model, resolved_cfg, tokenizer). Model is in ``.eval()`` on the
    requested device. If ``use_ema`` is True and the checkpoint has EMA state,
    the EMA shadow params replace the live params for the duration of eval
    (call ``restore_live_params`` if you need to revert).
    """
    cfg = _compose_config(config_name, overrides)
    tok = _build_tokenizer(cfg)

    # ``weights_only=False`` is required because Lightning stashes the full
    # config/tokenizer in hparams; ``strict=False`` gives us graceful handling
    # of optional buffers the forward-compat branch may add later.
    model = ABD3Diffusion.load_from_checkpoint(
        checkpoint_path,
        config=cfg,
        tokenizer=tok,
        map_location=device,
        strict=False,
        weights_only=False,
    )
    model = model.to(device)
    model.eval()

    # If the checkpoint predates EMA-state serialization, reseed shadows from
    # the now-loaded live weights. Must happen AFTER load_from_checkpoint
    # returns (i.e. after state_dict was applied): the `on_load_checkpoint`
    # hook itself runs before state_dict load and cannot do this safely.
    if getattr(model, "_ema_needs_live_reseed", False):
        model.sync_ema_from_live()

    if use_ema and getattr(model, "ema", None) is not None:
        # `store` snapshots live params; `copy_to` swaps EMA shadow into live.
        model.ema.store(model._get_parameters())
        model.ema.copy_to(model._get_parameters())
        model._ema_swapped = True
    else:
        model._ema_swapped = False

    return model, cfg, tok


def restore_live_params(model: ABD3Diffusion) -> None:
    """Undo a prior EMA swap-in so the model matches the train-time state."""
    if getattr(model, "_ema_swapped", False) and model.ema is not None:
        model.ema.restore(model._get_parameters())
        model._ema_swapped = False


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PPLResult:
    """One evaluation's numbers, serialisable as JSON for dashboards."""

    checkpoint: str
    dataset: str
    block_size: int
    n_samples: int
    n_sequences: int
    tokens_per_pass: int
    per_token_nll: float
    ppl: float
    per_pass_ppl: list[float]
    wall_seconds: float
    device: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        ckpt = pathlib.Path(self.checkpoint).name
        spread = (
            f"  per-pass ppl range: "
            f"{min(self.per_pass_ppl):.2f}..{max(self.per_pass_ppl):.2f}"
            if len(self.per_pass_ppl) > 1
            else ""
        )
        return (
            f"{ckpt}  "
            f"dataset={self.dataset}  B={self.block_size}  "
            f"n_samples={self.n_samples}  seqs={self.n_sequences}  "
            f"tokens/pass={self.tokens_per_pass}  "
            f"nll={self.per_token_nll:.3f}  ppl={self.ppl:.2f}"
            f"{spread}  [{self.wall_seconds:.1f}s, {self.device}]"
        )


@torch.no_grad()
def compute_perplexity(
    model: ABD3Diffusion,
    val_loader: Iterable[dict],
    *,
    n_samples: int = 8,
    device: str = "cpu",
    seed: int = 42,
    max_batches: Optional[int] = None,
    progress: bool = True,
) -> PPLResult:
    """Return the MC-ELBO perplexity of ``model`` on ``val_loader``.

    Parameters
    ----------
    model
        An ABD3Diffusion module in eval mode (EMA already swapped if desired).
    val_loader
        Yields dicts with ``input_ids`` and ``attention_mask`` (Long / Long).
    n_samples
        Monte-Carlo ELBO draws (BD3-LMs default = 8). Each draw reseeds so the
        passes are both reproducible and decorrelated.
    seed
        Base seed; pass ``s`` uses ``seed + s``.
    max_batches
        Optional cap used by unit tests / fast smoke runs.
    progress
        Print a running NLL line while iterating.
    """

    per_pass_ppl: list[float] = []
    total_nll_sum = 0.0
    total_token_sum = 0
    n_sequences = 0
    tokens_per_pass = 0

    t0 = time.time()
    for mc in range(n_samples):
        torch.manual_seed(seed + mc)

        pass_nll = 0.0
        pass_tokens = 0
        batches_seen = 0

        for batch in val_loader:
            if max_batches is not None and batches_seen >= max_batches:
                break

            x = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)

            # Model._loss returns a Loss(loss=token_nll, nlls, token_mask)
            # where `nlls` is already (per_token_loss * attention_mask) so
            # summing across the batch gives the total NLL for that batch.
            losses = model._loss(x, mask)
            pass_nll += float(losses.nlls.sum().item())
            pass_tokens += int(losses.token_mask.sum().item())

            if mc == 0:
                n_sequences += int(x.shape[0])

            batches_seen += 1
            if progress:
                running_nll = pass_nll / max(pass_tokens, 1)
                sys.stdout.write(
                    f"\r[ppl] pass {mc + 1}/{n_samples}  "
                    f"batch {batches_seen}  running nll={running_nll:.3f}  "
                    f"ppl={math.exp(running_nll):.2f}"
                )
                sys.stdout.flush()

        if pass_tokens == 0:
            raise RuntimeError(
                "No tokens seen on validation pass; is the dataloader empty?"
            )

        per_pass_ppl.append(math.exp(pass_nll / pass_tokens))
        total_nll_sum += pass_nll
        total_token_sum += pass_tokens
        if tokens_per_pass == 0:
            tokens_per_pass = pass_tokens
        if progress:
            sys.stdout.write("\n")

    wall = time.time() - t0
    per_token_nll = total_nll_sum / total_token_sum
    ppl = math.exp(per_token_nll)

    dataset_name = "unknown"
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "data"):
        subset = getattr(cfg.data, "subset", None)
        dataset_name = f"{cfg.data.name}" + (f"/{subset}" if subset else "")

    return PPLResult(
        checkpoint="",  # caller fills in
        dataset=dataset_name,
        block_size=int(getattr(model, "block_size", -1)),
        n_samples=n_samples,
        n_sequences=n_sequences,
        tokens_per_pass=tokens_per_pass,
        per_token_nll=per_token_nll,
        ppl=ppl,
        per_pass_ppl=per_pass_ppl,
        wall_seconds=wall,
        device=device,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate importance-weighted MC-ELBO perplexity for an ABD3 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file.")
    ap.add_argument("--config-name", default="feasibility",
                    help="Hydra config under configs/ (e.g. feasibility, config).")
    ap.add_argument(
        "--overrides", nargs="*", default=[],
        help="Hydra-style overrides, e.g. data=ptb block_size=8.",
    )
    ap.add_argument("--n-samples", type=int, default=8,
                    help="Monte-Carlo ELBO draws (BD3-LMs default = 8).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=True,
                    help="Swap EMA shadow params in for eval (the train-time default).")
    ap.add_argument("--no-ema", dest="use_ema", action="store_false",
                    help="Evaluate the live (non-EMA) params instead.")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap val batches per MC pass (smoke runs).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None,
                    help="Optional path to dump the PPLResult as JSON.")
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    print(f"[ppl] loading checkpoint: {args.checkpoint}")
    model, cfg, tok = load_abd3_from_checkpoint(
        args.checkpoint,
        config_name=args.config_name,
        overrides=args.overrides,
        device=args.device,
        use_ema=args.use_ema,
    )
    # Keep a handle on the resolved config so compute_perplexity can read it.
    model.config = cfg

    subset = getattr(cfg.data, "subset", None)
    print(
        f"[ppl] dataset={cfg.data.name}"
        + (f"/{subset}" if subset else "")
        + f"  block_size={cfg.block_size}  n_samples={args.n_samples}"
        + f"  ema={'on' if args.use_ema else 'off'}  device={args.device}"
    )

    # Force num_workers=0 for eval — avoids the same DataLoader-worker crash
    # that bit us during ablation runs on the shared host.
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.loader.num_workers = 0

    _, val_loader = abd3_dataloader.get_dataloaders(cfg, tok)

    result = compute_perplexity(
        model, val_loader,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
        max_batches=args.max_batches,
        progress=args.progress,
    )
    result.checkpoint = args.checkpoint

    print()
    print("=" * 100)
    print(result.summary_line())
    print("=" * 100)

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.as_dict(), indent=2))
        print(f"[ppl] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
