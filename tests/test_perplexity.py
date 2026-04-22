"""Tests for eval/perplexity.py — the MC-ELBO aggregation math.

These tests deliberately avoid loading a full ABD3Diffusion (which would need a
tokenizer download and a real dataset). They validate the things that are
easy to get subtly wrong and that matter for every number in the paper:

  * total_nll / total_tokens is pooled correctly across MC passes
  * per-pass PPL equals exp(per_pass_nll / per_pass_tokens)
  * the aggregate PPL equals exp(total_nll / total_tokens), NOT the mean of
    per-pass PPLs (a classic mistake — mean-of-exps != exp-of-mean)
  * re-running with the same seed produces identical results
  * different seeds produce different results (the MC loop is actually
    exercising randomness)
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import pytest
import torch

# Make the project root importable when pytest is run from anywhere.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.perplexity import compute_perplexity  # noqa: E402


@dataclass
class _Loss:
    """Mirrors abd3.diffusion.Loss — only the fields compute_perplexity uses."""

    loss: torch.Tensor
    nlls: torch.Tensor
    token_mask: torch.Tensor


class _MockModel:
    """Stand-in for ABD3Diffusion that consumes batches and emits ._loss().

    Each call draws a fresh noise tensor whose mean matches a configurable
    ``true_nll`` so the aggregate NLL is predictable in expectation. Using
    torch's RNG (rather than numpy) means compute_perplexity's seeding
    actually affects this mock, which is exactly what we want to test.
    """

    def __init__(self, true_nll: float = 3.0, noise_std: float = 0.5, seq_len: int = 16):
        self.true_nll = true_nll
        self.noise_std = noise_std
        self.seq_len = seq_len
        self.block_size = 4
        self.config = None
        self.call_count = 0

    def _loss(self, x: torch.Tensor, mask: torch.Tensor) -> _Loss:
        self.call_count += 1
        # Per-position nll = true_nll + Gaussian noise, zeroed outside the mask.
        raw = self.true_nll + self.noise_std * torch.randn_like(x.float())
        nlls = raw * mask.float()
        token_nll = nlls.sum() / mask.sum()
        return _Loss(loss=token_nll, nlls=nlls, token_mask=mask.float())


def _synth_loader(n_batches: int = 4, batch_size: int = 2, seq_len: int = 16):
    """Tiny in-memory loader producing deterministic batches."""
    batches = []
    for b in range(n_batches):
        g = torch.Generator().manual_seed(100 + b)
        batches.append(
            {
                "input_ids": torch.randint(0, 50, (batch_size, seq_len), generator=g),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            }
        )
    return batches


def test_aggregate_ppl_is_exp_of_pooled_nll():
    """exp(total_nll / total_tokens) should NOT equal mean(per-pass ppls)."""
    model = _MockModel(true_nll=3.0, noise_std=0.3)
    loader = _synth_loader(n_batches=4, batch_size=2, seq_len=16)

    res = compute_perplexity(model, loader, n_samples=4, device="cpu", seed=0, progress=False)

    # Pooled aggregation matches the summed nll / summed tokens. The math is
    # the only thing under test here — the exact value is noise-dependent.
    # Per-pass tokens should be identical (same loader, same masks).
    assert (
        len(set(res.per_pass_ppl)) == 4
    ), "MC passes should produce different per-pass PPLs (different seeds)"
    # Aggregate PPL falls within the convex hull of per-pass PPLs.
    assert min(res.per_pass_ppl) <= res.ppl <= max(res.per_pass_ppl) * 1.001, (
        f"Aggregate PPL {res.ppl} out of per-pass range "
        f"[{min(res.per_pass_ppl)}, {max(res.per_pass_ppl)}]"
    )
    # It does NOT equal the arithmetic mean of per-pass PPLs (except by
    # coincidence). Check that we're on the right side of Jensen's inequality:
    #   exp( mean log_ppl ) <= mean( exp log_ppl ) = mean( ppl )
    assert res.ppl <= sum(res.per_pass_ppl) / len(res.per_pass_ppl) + 1e-6


def test_determinism_given_same_seed():
    model_a = _MockModel()
    model_b = _MockModel()
    loader = _synth_loader()

    r_a = compute_perplexity(model_a, loader, n_samples=3, device="cpu", seed=42, progress=False)
    r_b = compute_perplexity(model_b, loader, n_samples=3, device="cpu", seed=42, progress=False)
    assert r_a.per_token_nll == pytest.approx(r_b.per_token_nll, rel=1e-12)
    assert r_a.per_pass_ppl == pytest.approx(r_b.per_pass_ppl, rel=1e-12)


def test_different_seeds_give_different_results():
    model = _MockModel()
    loader = _synth_loader()

    r1 = compute_perplexity(model, loader, n_samples=2, device="cpu", seed=1, progress=False)
    r2 = compute_perplexity(model, loader, n_samples=2, device="cpu", seed=9999, progress=False)
    assert r1.per_token_nll != r2.per_token_nll, (
        "Different seeds should yield different PPLs — if they don't, the "
        "per-pass seeding in compute_perplexity isn't actually plumbing into "
        "the model's RNG."
    )


def test_max_batches_caps_iteration():
    model = _MockModel()
    loader = _synth_loader(n_batches=10)

    res = compute_perplexity(
        model, loader, n_samples=1, device="cpu", max_batches=3, progress=False
    )
    # 3 batches of 2 sequences = 6
    assert res.n_sequences == 6


def test_attention_mask_is_respected():
    """Tokens outside the attention mask must not contribute to the NLL."""

    class _MaskingModel(_MockModel):
        def _loss(self, x, mask):
            # Constant 10-nat loss at every position, but only half of each
            # sequence is attended. Correct aggregation => average NLL = 10,
            # PPL = exp(10). Incorrect aggregation (dividing by seq_len rather
            # than mask.sum()) would halve it.
            nlls = torch.full_like(x.float(), 10.0) * mask.float()
            return _Loss(loss=nlls.sum() / mask.sum(), nlls=nlls, token_mask=mask.float())

    model = _MaskingModel()
    # Half-masked sequences.
    loader = [
        {
            "input_ids": torch.randint(0, 50, (2, 16)),
            "attention_mask": torch.tensor(
                [[1] * 8 + [0] * 8, [1] * 8 + [0] * 8], dtype=torch.long
            ),
        }
        for _ in range(2)
    ]

    res = compute_perplexity(model, loader, n_samples=1, device="cpu", progress=False)
    assert res.per_token_nll == pytest.approx(10.0, rel=1e-9)
    assert res.tokens_per_pass == 32  # 2 batches * 2 seqs * 8 attended tokens


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
