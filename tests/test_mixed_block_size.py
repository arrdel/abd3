"""Tests for mixed block-size training."""

import torch
import pytest


def test_sample_block_size():
    """Block size sampling should return valid choices."""
    from abd3.models.attention import sample_block_size, BLOCK_SIZE_CHOICES
    for _ in range(100):
        bs = sample_block_size()
        assert bs in BLOCK_SIZE_CHOICES


def test_mask_generation_different_block_sizes():
    """Masks should differ for different block sizes."""
    from abd3.models.attention import generate_masks_for_block_size
    m4 = generate_masks_for_block_size(32, 4, backend='sdpa', device='cpu')
    m8 = generate_masks_for_block_size(32, 8, backend='sdpa', device='cpu')
    assert not torch.equal(m4['self_attn_mask'], m8['self_attn_mask']), \
        "Different block sizes should produce different masks"


def test_mask_cache():
    """MaskCache should return same masks for same block size."""
    from abd3.models.attention import MaskCache
    cache = MaskCache(32, backend='sdpa', device='cpu')
    m1 = cache.get_masks(4)
    m2 = cache.get_masks(4)
    assert m1 is m2, "Cache should return same object"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
