"""
ABD3 Attention Masks: Adaptive block-size masks for training and inference.

Key changes from BD3-LMs:
- Dynamic mask generation for mixed block sizes
- Two-stream architecture: separate x_t self-attention + cross-attention to x_0
- Support for variable block sizes within a single batch
"""

from functools import partial

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    create_block_mask = None  # type: ignore[assignment]


def block_self_attn_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """Block-diagonal self-attention mask for x_t tokens.

    Each x_t block attends only to itself (bidirectional within block).
    This is the M_BD component from BD3-LMs, but ONLY for x_t.
    In our two-stream design, x_0 is handled via cross-attention.
    """
    block_q = q_idx // block_size
    block_kv = kv_idx // block_size
    return block_q == block_kv


def block_cross_attn_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """Cross-attention mask: x_t attends to x_0 from previous blocks.

    x_t tokens in block b attend to x_0 tokens from blocks < b.
    This replaces the M_OBC component from BD3-LMs.
    """
    block_q = q_idx // block_size
    block_kv = kv_idx // block_size
    return block_q > block_kv


def generate_self_attn_mask(seq_len, block_size, backend="flex", device="cuda"):
    """Generate block-diagonal self-attention mask for x_t stream.

    Args:
        seq_len: Length of x_t sequence (N tokens)
        block_size: Current block size
        backend: 'flex' or 'sdpa'
        device: torch device

    Returns:
        Attention mask suitable for the chosen backend
    """
    if backend == "flex" and FLEX_ATTN_AVAILABLE:
        return create_block_mask(
            partial(block_self_attn_mask, block_size=block_size, n=seq_len),
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )
    else:
        q_idx = torch.arange(seq_len, device=device)[:, None]
        kv_idx = torch.arange(seq_len, device=device)[None, :]
        return block_self_attn_mask(None, None, q_idx, kv_idx, block_size=block_size, n=seq_len)


def generate_cross_attn_mask(seq_len, block_size, backend="flex", device="cuda"):
    """Generate cross-attention mask: x_t attends to past x_0 blocks.

    Args:
        seq_len: Length of sequence (N tokens)
        block_size: Current block size
        backend: 'flex' or 'sdpa'
        device: torch device

    Returns:
        Cross-attention mask [N_xt, N_x0]
    """
    if backend == "flex" and FLEX_ATTN_AVAILABLE:
        return create_block_mask(
            partial(block_cross_attn_mask, block_size=block_size, n=seq_len),
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )
    else:
        q_idx = torch.arange(seq_len, device=device)[:, None]
        kv_idx = torch.arange(seq_len, device=device)[None, :]
        return block_cross_attn_mask(None, None, q_idx, kv_idx, block_size=block_size, n=seq_len)


def generate_masks_for_block_size(seq_len, block_size, backend="flex", device="cuda"):
    """Generate both self-attention and cross-attention masks.

    This is the main entry point for mask generation in ABD3.
    Called once per batch when using mixed block-size training.

    Args:
        seq_len: Sequence length N
        block_size: Block size for this batch
        backend: Attention backend
        device: torch device

    Returns:
        dict with 'self_attn_mask' and 'cross_attn_mask'
    """
    return {
        "self_attn_mask": generate_self_attn_mask(seq_len, block_size, backend, device),
        "cross_attn_mask": generate_cross_attn_mask(seq_len, block_size, backend, device),
    }


# ============================================================================
# Mixed block-size utilities
# ============================================================================

BLOCK_SIZE_CHOICES = [1, 2, 4, 8, 16]


def sample_block_size(choices=None):
    """Sample a random block size for mixed block-size training.

    Args:
        choices: List of valid block sizes. Defaults to [1, 2, 4, 8, 16].

    Returns:
        int: sampled block size
    """
    if choices is None:
        choices = BLOCK_SIZE_CHOICES
    idx = torch.randint(0, len(choices), (1,)).item()
    return choices[idx]


class MaskCache:
    """Caches generated masks to avoid recomputation for repeated block sizes."""

    def __init__(self, seq_len, backend="flex", device=None):
        self.seq_len = seq_len
        self.backend = backend
        self.device = device  # None = lazy, set on first call
        self._cache = {}

    def get_masks(self, block_size, device=None):
        if device is not None and self.device is None:
            self.device = device
        dev = self.device or "cpu"

        cache_key = (block_size, str(dev))
        if cache_key not in self._cache:
            self._cache[cache_key] = generate_masks_for_block_size(
                self.seq_len, block_size, self.backend, dev
            )
        return self._cache[cache_key]

    def clear(self):
        self._cache.clear()
