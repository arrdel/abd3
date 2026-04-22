"""Tests for EMA checkpoint round-trip (Phase 1.3).

Guards against a regression of the silent EMA reset bug that produced
near-uniform predictions at eval time: ``shadow_params`` is a plain Python
list, not a registered buffer, so unless ``on_save_checkpoint`` /
``on_load_checkpoint`` hand-serialise it, Lightning's state_dict will carry
*only* the live weights through a checkpoint round-trip.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abd3.models.ema import ExponentialMovingAverage  # noqa: E402


def _tiny_module(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def test_ema_state_dict_captures_shadows():
    model = _tiny_module(0)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    # Step the EMA forward a few times so shadows != live params.
    for _ in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model.parameters())

    sd = ema.state_dict()
    assert "decay" in sd and sd["decay"] == pytest.approx(0.9)
    assert "shadow_params" in sd
    assert len(sd["shadow_params"]) == len(list(model.parameters()))

    # Tensors must be detached CPU copies, not aliased to the live shadows.
    sd["shadow_params"][0].zero_()
    assert not torch.all(ema.shadow_params[0] == 0), (
        "state_dict returned live references; mutating the serialised tensor "
        "must NOT corrupt the in-memory EMA."
    )


def test_load_state_dict_restores_exact_values():
    # Build an EMA, capture shadows.
    src_model = _tiny_module(0)
    src_ema = ExponentialMovingAverage(src_model.parameters(), decay=0.99)
    for _ in range(3):
        with torch.no_grad():
            for p in src_model.parameters():
                p.add_(torch.randn_like(p) * 0.2)
        src_ema.update(src_model.parameters())
    captured = src_ema.state_dict()

    # Build a fresh EMA from a *different* model init — its shadows are wrong.
    tgt_model = _tiny_module(42)
    tgt_ema = ExponentialMovingAverage(tgt_model.parameters(), decay=0.5)
    # Sanity: the two EMAs start with different shadows.
    assert not torch.allclose(src_ema.shadow_params[0], tgt_ema.shadow_params[0])

    tgt_ema.load_state_dict(captured)
    # After loading, shadows should match bit-for-bit.
    for src_s, tgt_s in zip(src_ema.shadow_params, tgt_ema.shadow_params):
        assert torch.allclose(src_s, tgt_s, atol=0, rtol=0)
    # And decay should have been restored.
    assert tgt_ema.decay == pytest.approx(0.99)


def test_load_state_dict_rejects_shape_mismatch():
    model = _tiny_module(0)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)

    bad = {"decay": 0.9, "shadow_params": [torch.zeros(99, 99)]}  # wrong count
    with pytest.raises(ValueError, match="param count mismatch"):
        ema.load_state_dict(bad)

    # Right count, wrong shape.
    bad = {"decay": 0.9, "shadow_params": [torch.zeros_like(p).squeeze()
                                            if p.ndim >= 2 else torch.zeros_like(p)
                                            for p in ema.shadow_params]}
    if any(s.shape != t.shape for s, t in zip(bad["shadow_params"], ema.shadow_params)):
        with pytest.raises(ValueError, match="shape mismatch"):
            ema.load_state_dict(bad)


def test_end_to_end_roundtrip_through_torch_save(tmp_path):
    """The thing that actually matters: save → load → identical behaviour."""
    model = _tiny_module(0)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    # Drive the shadows far from live params.
    for _ in range(20):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        ema.update(model.parameters())

    # Capture a deterministic "signal" from the shadow params — using them to
    # transform a fixed input should give a reproducible output.
    probe = torch.randn(3, 8)
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    before = model(probe).detach().clone()
    ema.restore(model.parameters())

    # Save.
    ckpt_path = tmp_path / "ema.pt"
    torch.save({"ema": ema.state_dict(), "model": model.state_dict()}, ckpt_path)

    # Rebuild fresh model + EMA from scratch, then load.
    fresh_model = _tiny_module(999)  # deliberately different init
    fresh_ema = ExponentialMovingAverage(fresh_model.parameters(), decay=0.5)

    payload = torch.load(ckpt_path, weights_only=False)
    fresh_model.load_state_dict(payload["model"])
    fresh_ema.load_state_dict(payload["ema"])

    fresh_ema.store(fresh_model.parameters())
    fresh_ema.copy_to(fresh_model.parameters())
    after = fresh_model(probe).detach()
    fresh_ema.restore(fresh_model.parameters())

    assert torch.allclose(before, after, atol=1e-6), (
        "EMA roundtrip through torch.save failed: swap-in outputs differ by "
        f"{(before - after).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
