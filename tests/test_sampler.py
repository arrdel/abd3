"""Tests for the semi-AR sampler in ABD3Diffusion.

Exercises the sliding-window x_accum path end-to-end on a tiny real model so
we catch regressions the previous (broken) sampler didn't catch — shape
mismatches in the past-blocks concat, wrong block indexing, or NFE miscounts.
A dedicated Diffusion instance is constructed here rather than loading a
checkpoint so the tests stay fast and offline.
"""

from __future__ import annotations

import pathlib
import sys

import omegaconf
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abd3.diffusion import ABD3Diffusion  # noqa: E402


class _MockTokenizer:
    """Minimal PreTrainedTokenizer stand-in for ABD3Diffusion.__init__."""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size
        # No mask_token → ABD3Diffusion will allocate one at vocab_size and
        # bump vocab_size by 1.
        self.mask_token = None


def _tiny_cfg(seq_len: int = 8, block_size: int = 2) -> omegaconf.DictConfig:
    """Smallest config that stands up a real ABD3Diffusion end-to-end."""
    return omegaconf.OmegaConf.create({
        "model": {
            "length": seq_len,
            "hidden_size": 16,
            "n_heads": 2,
            "n_blocks": 1,
            "cond_dim": 16,
            "dropout": 0.0,
            "attn_backend": "sdpa",
            "tie_word_embeddings": False,
            "max_seqlen": seq_len,
        },
        "block_size": block_size,
        "algo": {
            "parameterization": "subs",
            "T": 5,
            "cross_attn": True,
            "self_conditioning": True,
            "self_cond_prob": 0.5,
            "time_conditioning": True,
            "mixed_block_sizes": False,
            "adaptive_stopping": False,  # default OFF in tests unless overridden
            "stop_entropy_threshold": 0.1,
            "stop_agreement_threshold": 2,
        },
        "noise": {"type": "loglinear"},
        "loader": {"eval_batch_size": 2},
        "training": {"ema": 0, "antithetic_sampling": False, "sampling_eps": 1e-3},
    })


def _build_model(seq_len: int = 8, block_size: int = 2,
                 adaptive_stopping: bool = False,
                 stop_agreement_threshold: int = 2):
    cfg = _tiny_cfg(seq_len=seq_len, block_size=block_size)
    cfg.algo.adaptive_stopping = adaptive_stopping
    cfg.algo.stop_agreement_threshold = stop_agreement_threshold
    tok = _MockTokenizer(vocab_size=32)
    model = ABD3Diffusion(cfg, tokenizer=tok)
    model.eval()
    return model


def test_sample_shape_and_total_nfe():
    """Basic sanity: output has shape [n, seq_len] and NFE is exactly num_blocks * num_steps."""
    model = _build_model(seq_len=8, block_size=2)
    torch.manual_seed(0)
    x, total_nfe, per_block_nfe = model.sample(
        n_samples=3, num_steps=4, track_nfe_per_block=True, progress=False)

    assert x.shape == (3, 8), f"expected (3, 8), got {tuple(x.shape)}"
    # seq_len=8, block_size=2 -> 4 blocks. num_steps=4. No adaptive stopping.
    assert total_nfe == 4 * 4, f"total_nfe should be 16, got {total_nfe}"
    assert per_block_nfe == [4, 4, 4, 4], f"got {per_block_nfe}"


def test_sample_seed_determinism():
    """Same seed => bit-identical samples. Guards against hidden RNG leaks."""
    model = _build_model(seq_len=8, block_size=2)
    torch.manual_seed(123)
    x_a, nfe_a = model.sample(n_samples=2, num_steps=3, progress=False)
    torch.manual_seed(123)
    x_b, nfe_b = model.sample(n_samples=2, num_steps=3, progress=False)
    assert torch.equal(x_a, x_b), "seed determinism broken"
    assert nfe_a == nfe_b


def test_adaptive_stopping_reduces_nfe_when_triggered():
    """Force agreement by freezing the model so argmax stabilises immediately.

    With all model parameters zeroed, the sampler's substitution-parameterised
    logits collapse to a degenerate distribution whose argmax is constant,
    which satisfies the adaptive-stopping agreement check after a few steps.
    """
    model = _build_model(
        seq_len=16, block_size=4,
        adaptive_stopping=True,
        stop_agreement_threshold=2,
    )
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.zero_()

    torch.manual_seed(7)
    # num_steps is large enough to give adaptive stopping room to fire
    # (the check only activates from step_idx > 2).
    _, total_nfe, per_block_nfe = model.sample(
        n_samples=2, num_steps=20, track_nfe_per_block=True, progress=False)

    num_blocks = 16 // 4
    # With a stable argmax, every block should early-exit well under 20 steps.
    assert total_nfe < num_blocks * 20, (
        f"adaptive stopping didn't fire: NFE={total_nfe} vs theoretical {num_blocks * 20}"
    )
    assert all(nfe < 20 for nfe in per_block_nfe), (
        f"no per-block NFE under the cap: {per_block_nfe}"
    )
    assert len(per_block_nfe) == num_blocks


def test_sample_rejects_indivisible_block_size():
    """seq_len must be divisible by block_size. Fail loud, not silent."""
    model = _build_model(seq_len=8, block_size=2)
    with pytest.raises(ValueError, match="divisible by"):
        model.sample(n_samples=1, num_steps=3, block_size=3, progress=False)


def test_sample_returns_tuple_lengths_match_flag():
    """Return arity should flip on ``track_nfe_per_block``."""
    model = _build_model(seq_len=8, block_size=2)
    torch.manual_seed(0)
    out_2 = model.sample(n_samples=1, num_steps=3, progress=False)
    assert len(out_2) == 2

    torch.manual_seed(0)
    out_3 = model.sample(n_samples=1, num_steps=3,
                         track_nfe_per_block=True, progress=False)
    assert len(out_3) == 3
    assert isinstance(out_3[2], list)
    assert len(out_3[2]) == 4  # num_blocks


def test_sampled_tokens_stay_in_vocab_range():
    """Sanity: decoded ids are all < model.vocab_size (incl the mask slot)."""
    model = _build_model(seq_len=8, block_size=2)
    torch.manual_seed(0)
    x, _ = model.sample(n_samples=2, num_steps=3, progress=False)
    assert x.min().item() >= 0
    assert x.max().item() < model.vocab_size
    # Substitution parameterization forbids mask_index from winning argmax,
    # so no sampled position should remain as mask_index.
    assert (x == model.mask_index).sum().item() == 0, (
        "mask_index leaked into final samples — substitution param broken?"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
