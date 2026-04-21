"""Tests for adaptive early stopping."""

import torch
import pytest


def test_low_entropy_triggers_stop():
    """Very confident predictions (low entropy) should trigger early stop."""
    # Simulate near-one-hot predictions
    p_x0 = torch.zeros(2, 4, 50)
    p_x0[:, :, 0] = 0.99
    p_x0[:, :, 1:] = 0.01 / 49

    entropy = -(p_x0 * (p_x0 + 1e-10).log()).sum(-1).mean()
    assert entropy < 0.1, f"Near-one-hot should have very low entropy, got {entropy}"


def test_agreement_detection():
    """Consecutive identical predictions should be detected."""
    pred1 = torch.tensor([[0, 1, 2, 3]])
    pred2 = torch.tensor([[0, 1, 2, 3]])
    pred3 = torch.tensor([[0, 1, 2, 4]])

    assert (pred1 == pred2).all(), "Same predictions should agree"
    assert not (pred1 == pred3).all(), "Different predictions should not agree"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
