"""Generation-perplexity evaluation under a frozen scorer LM.

What this measures
------------------
Given a corpus of **generated** texts (from :mod:`eval.generate`), re-score
each sample under a frozen causal LM (GPT-2 by default) and report
perplexity. This is the standard "does the sample look like natural text?"
signal in the diffusion-LM literature (Gong et al., 2023; Han et al.,
2023; BD3-LMs). Lower Gen-PPL = more fluent.

Note: Gen-PPL and model-PPL measure different things. Model-PPL (see
:mod:`eval.perplexity`) scores *held-out data* under *our* model — does
the model understand real text? Gen-PPL scores *our samples* under a
*reference* model — do our samples look like real text? A good diffusion
LM should win on both, but it's very possible (and embarrassing) to win
on the first and lose on the second via a low-diversity collapse, which
is why this file exists.

Record contract
---------------
Consumes JSONL with the ``text`` field populated (falls back to decoding
``token_ids`` with a caller-provided tokenizer if ``text`` is missing).
Emits a flat JSON object with per-sample PPL and corpus-level aggregates
so plots and paper tables can pick the columns they want.

CLI::

    python -m eval.gen_ppl \\
        --samples report/samples/feasibility_step4000.jsonl \\
        --scorer gpt2 --device cuda \\
        --json-out report/gen_ppl/feasibility_step4000.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import statistics
import sys
from collections.abc import Iterable, Sequence

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_texts_with_lm(
    texts: Sequence[str],
    *,
    scorer_model,
    scorer_tokenizer,
    device: str | torch.device = "cpu",
    max_length: int = 1024,
    batch_size: int = 4,
    min_tokens: int = 2,
    add_bos: bool = True,
) -> list[dict]:
    """Score each text under ``scorer_model`` and return per-sample NLL+PPL.

    The scorer is treated as a *pure function*: we put it in eval mode,
    disable grads, and always move it to ``device`` before the loop. We
    compute a shifted cross-entropy exactly the way HF's
    ``GPT2LMHeadModel`` does internally, but per-sample (instead of
    batch-averaged) so the output dict's ``ppl`` is a valid per-sample
    quantity.

    Parameters
    ----------
    add_bos
        Prepend the scorer's ``bos_token_id`` (or ``eos_token_id`` as a
        fallback for GPT-2, which ties them). Without this, the first
        token is unscorable and short texts lose a meaningful fraction
        of their signal.
    min_tokens
        Skip texts with fewer than ``min_tokens`` scorable positions
        (shift reduces the usable length by 1). Those samples are
        reported with ``ppl=None`` so downstream aggregates can exclude
        them cleanly.
    """
    scorer_model = scorer_model.to(device).eval()

    if scorer_tokenizer.pad_token_id is None:
        # GPT-2 has no pad token; reuse EOS as a pure padding sentinel. Combined
        # with the attention mask we build below this never leaks into the loss.
        # We set both the str handle and the id explicitly because some tokenizer
        # shims don't auto-sync the id via the property setter.
        scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
        scorer_tokenizer.pad_token_id = scorer_tokenizer.eos_token_id

    bos_id = scorer_tokenizer.bos_token_id
    if bos_id is None:
        bos_id = scorer_tokenizer.eos_token_id  # GPT-2 fallback.

    results: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        if add_bos:
            # GPT-2 tokenizer does not add BOS by default; we inject it as the
            # first position of the input_ids so position 0 is scorable.
            encoded = scorer_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length - 1,  # reserve slot for BOS.
            )
            input_ids = encoded["input_ids"]
            attn = encoded["attention_mask"]
            # prepend BOS
            bos_col = torch.full((input_ids.size(0), 1), bos_id, dtype=input_ids.dtype)
            ones_col = torch.ones_like(bos_col)
            input_ids = torch.cat([bos_col, input_ids], dim=1)
            attn = torch.cat([ones_col, attn], dim=1)
        else:
            encoded = scorer_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"]
            attn = encoded["attention_mask"]

        input_ids = input_ids.to(device)
        attn = attn.to(device)
        logits = scorer_model(input_ids=input_ids, attention_mask=attn).logits

        # Standard LM shift: logits at position i predict token at i+1.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().to(shift_logits.dtype)

        # Per-token NLL; zero out masked positions so they contribute nothing.
        nll_per_tok = (
            torch.nn.functional.cross_entropy(
                shift_logits.transpose(1, 2),  # [B, V, T]
                shift_labels,
                reduction="none",
            )
            * shift_mask
        )  # [B, T]

        # Per-sample sum and length.
        nll_sum = nll_per_tok.sum(dim=1)
        n_tokens = shift_mask.sum(dim=1)

        nll_sum_cpu = nll_sum.float().cpu().tolist()
        n_tokens_cpu = n_tokens.int().cpu().tolist()

        for text, nll, n_tok in zip(batch_texts, nll_sum_cpu, n_tokens_cpu, strict=True):
            if n_tok < min_tokens:
                results.append(
                    {
                        "text": text,
                        "n_tokens": int(n_tok),
                        "nll": None,
                        "ppl": None,
                        "skipped": True,
                    }
                )
                continue
            mean_nll = nll / n_tok
            results.append(
                {
                    "text": text,
                    "n_tokens": int(n_tok),
                    "nll": float(nll),
                    "mean_nll": float(mean_nll),
                    "ppl": float(math.exp(mean_nll)) if mean_nll < 50 else float("inf"),
                    "skipped": False,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Corpus aggregates
# ---------------------------------------------------------------------------


def _corpus_ppl(per_sample: Iterable[dict]) -> float:
    """Token-weighted corpus PPL = exp(total_nll / total_tokens).

    Equivalent to the perplexity you'd report if you treated all generated
    samples as a single long document. Robust to variable-length outputs
    unlike a naive mean of per-sample PPLs.
    """
    total_nll = 0.0
    total_tok = 0
    for r in per_sample:
        if r.get("skipped") or r.get("nll") is None:
            continue
        total_nll += float(r["nll"])
        total_tok += int(r["n_tokens"])
    if total_tok == 0:
        return float("nan")
    return math.exp(total_nll / total_tok)


def _mean_sample_ppl(per_sample: Iterable[dict]) -> float:
    """Unweighted mean of per-sample PPLs (ignoring skipped samples)."""
    vals = [
        r["ppl"]
        for r in per_sample
        if not r.get("skipped") and r.get("ppl") is not None and math.isfinite(r["ppl"])
    ]
    if not vals:
        return float("nan")
    return statistics.fmean(vals)


def _median_sample_ppl(per_sample: Iterable[dict]) -> float:
    vals = [
        r["ppl"]
        for r in per_sample
        if not r.get("skipped") and r.get("ppl") is not None and math.isfinite(r["ppl"])
    ]
    if not vals:
        return float("nan")
    return statistics.median(vals)


@dataclasses.dataclass
class GenPPLReport:
    """Top-level report, flat and JSON-serializable."""

    n_samples: int
    n_scored: int
    n_skipped: int
    corpus_ppl: float  # exp(total_nll / total_tokens) — preferred.
    mean_sample_ppl: float  # mean over samples — reported for parity with prior work.
    median_sample_ppl: float
    mean_tokens: float
    scorer: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        return (
            f"n={self.n_samples} (scored={self.n_scored}, skipped={self.n_skipped})  "
            f"corpus-PPL={self.corpus_ppl:.2f}  "
            f"mean-PPL={self.mean_sample_ppl:.2f}  "
            f"median-PPL={self.median_sample_ppl:.2f}  "
            f"mean-len={self.mean_tokens:.1f}  [scorer={self.scorer}]"
        )


def aggregate(per_sample: Sequence[dict], scorer: str) -> GenPPLReport:
    scored = [r for r in per_sample if not r.get("skipped")]
    lens = [r["n_tokens"] for r in scored]
    return GenPPLReport(
        n_samples=len(per_sample),
        n_scored=len(scored),
        n_skipped=len(per_sample) - len(scored),
        corpus_ppl=_corpus_ppl(per_sample),
        mean_sample_ppl=_mean_sample_ppl(per_sample),
        median_sample_ppl=_median_sample_ppl(per_sample),
        mean_tokens=(sum(lens) / len(lens)) if lens else 0.0,
        scorer=scorer,
    )


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def compute_gen_ppl(
    texts: Sequence[str],
    *,
    scorer_name: str = "gpt2",
    device: str = "cpu",
    batch_size: int = 4,
    max_length: int = 1024,
    min_tokens: int = 2,
) -> tuple[GenPPLReport, list[dict]]:
    """Load the scorer LM lazily, score every text, return report + per-sample rows.

    We import ``transformers`` inside the function so the module itself
    stays cheap to import (useful for unit tests and for :mod:`eval.quality`
    composition when a user only wants diversity metrics).
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(scorer_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(scorer_name)

    per_sample = score_texts_with_lm(
        texts,
        scorer_model=model,
        scorer_tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        min_tokens=min_tokens,
    )
    return aggregate(per_sample, scorer=scorer_name), per_sample


# ---------------------------------------------------------------------------
# Sample-file loader (re-export from diversity to keep one source of truth)
# ---------------------------------------------------------------------------


def load_samples(path: str | pathlib.Path) -> list[dict]:
    from eval.diversity import load_samples as _load

    return _load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generation-PPL under a frozen LM (default: GPT-2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--samples", required=True, help="Path to a SampleRecord JSONL.")
    ap.add_argument(
        "--scorer",
        default="gpt2",
        help="HF hub id of the scorer LM (gpt2 / gpt2-medium / gpt2-large).",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Per-sample truncation length; must be <= scorer context.",
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Skip samples with fewer scorable positions than this.",
    )
    ap.add_argument(
        "--json-out", default=None, help="Optional output JSON (includes per-sample rows)."
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    records = load_samples(args.samples)
    if not records:
        print(f"[gen_ppl] {args.samples}: empty; nothing to score", file=sys.stderr)
        return 1

    texts = [r.get("text", "") for r in records]
    # Drop fully-empty strings so GPT-2 doesn't see an empty batch (which triggers a
    # shape-zero attention mask and a noisy warning).
    nonempty = [t for t in texts if t.strip()]
    if not nonempty:
        print("[gen_ppl] all `text` fields are empty; nothing to score.", file=sys.stderr)
        return 1

    print(
        f"[gen_ppl] scoring {len(nonempty)}/{len(texts)} non-empty samples with "
        f"{args.scorer} on {args.device}"
    )

    report, per_sample = compute_gen_ppl(
        nonempty,
        scorer_name=args.scorer,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        min_tokens=args.min_tokens,
    )

    print("=" * 100)
    print(report.summary_line())
    print("=" * 100)

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Drop the `text` from per-sample rows to keep the output file compact —
        # the raw text already lives in the samples JSONL.
        rows = [{k: v for k, v in r.items() if k != "text"} for r in per_sample]
        payload = {
            "samples_path": str(args.samples),
            "scorer": args.scorer,
            "report": report.as_dict(),
            "per_sample": rows,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"[gen_ppl] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
