"""
ABD3 Two-Stream DIT: Efficient Discrete Diffusion Transformer.

Key changes from BD3-LMs DIT:
1. Two-stream architecture: x_t self-attention + cross-attention to x_0 (not 2N concat)
2. Per-block time conditioning (adaLN receives per-block sigma, not zero)
3. Self-conditioning: optional input of previous x̂_0 prediction
4. Support for dynamic block sizes (mixed block-size training)
"""

import math
import typing
from functools import partial

import einops
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    import flash_attn.layers.rotary
except ImportError:
    flash_attn = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False

import huggingface_hub
import omegaconf

from .attention import generate_masks_for_block_size, MaskCache

# ============================================================================
# JIT fusion flags
# ============================================================================
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


# ============================================================================
# Utility functions (inherited from BD3-LMs)
# ============================================================================

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(x: torch.Tensor, bias: typing.Optional[torch.Tensor],
                                        scale: torch.Tensor, residual: typing.Optional[torch.Tensor],
                                        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(x: torch.Tensor, bias: typing.Optional[torch.Tensor],
                                            scale: torch.Tensor, residual: typing.Optional[torch.Tensor],
                                            prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


# ============================================================================
# Positional embeddings
# ============================================================================

class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


# ============================================================================
# Core layers
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        # Device-agnostic autocast (works on CPU, CUDA, MPS)
        device_type = x.device.type if x.device.type != 'mps' else 'cpu'
        with torch.amp.autocast(device_type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half).to(t.dtype).to(t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, adaLN):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        if c is not None:
            if c.shape[0] == x.shape[0]:
                shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            else:
                shift, scale = rearrange(
                    self.adaLN_modulation(c), '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


# ============================================================================
# Two-Stream Transformer Block (ABD3 core innovation)
# ============================================================================

class TwoStreamDiTBlock(nn.Module):
    """Transformer block with separate self-attention (x_t) and cross-attention (to x_0).

    Unlike BD3-LMs which concatenates [x_t; x_0] into a 2N sequence,
    we process x_t with block-diagonal self-attention (N tokens), and
    cross-attend to cached x_0 key-values from previous blocks.

    This halves the attention cost from O((2N)²) to O(N²) + O(N·N_past).
    """

    def __init__(self, n, dim, n_heads, cond_dim, adaLN=True,
                 mlp_ratio=4, dropout=0.1, block_size=1,
                 attn_backend='flash_attn', max_seqlen=1024):
        super().__init__()
        self.n = n
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.block_size = block_size
        self.adaLN = adaLN
        self.attn_backend = attn_backend
        self.max_seqlen = max_seqlen

        # Self-attention on x_t
        self.norm1 = LayerNorm(dim)
        self.self_attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_attn_out = nn.Linear(dim, dim, bias=False)

        # Cross-attention to x_0 (NEW in ABD3)
        self.norm_cross = LayerNorm(dim)
        self.cross_attn_q = nn.Linear(dim, dim, bias=False)
        self.cross_attn_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_attn_out = nn.Linear(dim, dim, bias=False)

        # FFN
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))

        self.dropout = dropout

        # adaLN modulation: 6 for self-attn + 3 for cross-attn gate = 9 total
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 9 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

        # KV cache for x_0 (populated during encoding, reused during x_t processing)
        self.x0_kv_cache = None
        self.kv_cache = None
        self.cache_idx = 0

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def _self_attention(self, x, rotary_cos_sin, mask=None):
        """Block-diagonal self-attention on x_t."""
        qkv = self.self_attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        with torch.amp.autocast(x.device.type if x.device.type != 'mps' else 'cpu', enabled=False):
            cos, sin = rotary_cos_sin
            if self.attn_backend == 'flash_attn' and flash_attn is not None:
                qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
            else:
                qkv = apply_rotary_pos_emb_torchscript(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        # SDPA attention with block-diagonal mask
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = [rearrange(t, 'b s h d -> b h s d') for t in [q, k, v]]

        if mask is not None and not isinstance(mask, bool):
            mask = mask.bool() if hasattr(mask, 'bool') else mask

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        return self.self_attn_out(attn_out)

    def _cross_attention(self, x, x0_kv, cross_mask=None):
        """Cross-attention: x_t queries attend to x_0 key-values from past blocks."""
        if x0_kv is None:
            return torch.zeros_like(x)

        q = self.cross_attn_q(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)

        kv = self.cross_attn_kv(x0_kv)
        k, v = rearrange(kv, 'b s (two h d) -> two b h s d', two=2, h=self.n_heads)

        if cross_mask is not None and not isinstance(cross_mask, bool):
            cross_mask = cross_mask.bool() if hasattr(cross_mask, 'bool') else cross_mask

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=cross_mask, dropout_p=self.dropout if self.training else 0.0)
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        return self.cross_attn_out(attn_out)

    def forward(self, x, rotary_cos_sin, c=None, x0_embed=None,
                self_attn_mask=None, cross_attn_mask=None, **kwargs):
        """
        Args:
            x: [B, N, D] - x_t embeddings (N tokens, NOT 2N)
            rotary_cos_sin: rotary positional embeddings
            c: [B, D] or [B*num_blocks, D] - per-block time conditioning
            x0_embed: [B, N, D] - x_0 embeddings for cross-attention (detached)
            self_attn_mask: block-diagonal mask for x_t self-attention
            cross_attn_mask: mask for x_t → x_0 cross-attention
        """
        batch_size = x.shape[0]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Parse adaLN modulation
        shift_msa, scale_msa, gate_msa = None, None, None
        gate_cross = None
        shift_mlp, scale_mlp, gate_mlp = None, None, None

        if c is not None and self.adaLN:
            if c.shape[0] == batch_size:
                chunks = self.adaLN_modulation(c)[:, None].chunk(9, dim=2)
            else:
                chunks = rearrange(
                    self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
                ).chunk(9, dim=-1)
            (shift_msa, scale_msa, gate_msa,
             gate_cross,
             shift_mlp, scale_mlp, gate_mlp,
             # 2 extra for future use
             _, _) = chunks

        # 1. Self-attention on x_t (block-diagonal)
        x_skip = x
        if c is not None:
            x_norm = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        else:
            x_norm = self.norm1(x)

        self_attn_out = self._self_attention(x_norm, rotary_cos_sin, mask=self_attn_mask)

        if c is not None:
            x = bias_dropout_scale_fn(self_attn_out, None, gate_msa, x_skip, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(self_attn_out, None, scale, x_skip, self.dropout)

        # 2. Cross-attention to x_0 (NEW in ABD3)
        if x0_embed is not None:
            x_skip2 = x
            x_cross_norm = self.norm_cross(x)
            cross_out = self._cross_attention(x_cross_norm, x0_embed, cross_mask=cross_attn_mask)

            if gate_cross is not None:
                x = bias_dropout_scale_fn(cross_out, None, gate_cross, x_skip2, self.dropout)
            else:
                scale = torch.ones(1, device=x.device, dtype=x.dtype)
                x = bias_dropout_scale_fn(cross_out, None, scale, x_skip2, self.dropout)

        # 3. FFN
        if c is not None:
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout)

        return x


# ============================================================================
# Main ABD3 DIT Model
# ============================================================================

class ABD3DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """ABD3 Discrete Diffusion Transformer with two-stream architecture.
    
    Innovations:
    - Two-stream: x_t self-attention + cross-attention to x_0 (halves memory)
    - Per-block time conditioning via adaLN
    - Self-conditioning support (previous x̂_0 prediction as auxiliary input)
    - Dynamic block size support for mixed block-size training
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.n = config.model.length
        self.vocab_size = vocab_size
        self.block_size = config.block_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.n_heads = config.model.n_heads
        self.adaLN = True
        self.attn_backend = getattr(config.model, 'attn_backend', 'sdpa')

        # Embeddings
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)

        # Self-conditioning projection (NEW in ABD3)
        # Projects previous x̂_0 prediction into the hidden space
        self.self_cond_proj = nn.Linear(dim, dim, bias=False)
        self.self_cond_proj.weight.data.zero_()  # Initialize to zero so it's a no-op initially

        # x_0 encoder: lightweight projection for cross-attention values
        # Processes x_0 WITHOUT gradient (detached) to save memory
        self.x0_proj = nn.Linear(dim, dim, bias=False)

        # Transformer blocks (two-stream)
        self.blocks = nn.ModuleList([
            TwoStreamDiTBlock(
                n=config.model.length,
                dim=dim,
                n_heads=config.model.n_heads,
                cond_dim=cond_dim,
                adaLN=True,
                dropout=config.model.dropout,
                block_size=self.block_size,
                attn_backend=self.attn_backend,
                max_seqlen=getattr(config.model, 'max_seqlen', 1024))
            for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=True)

        # Mask cache for efficient mixed block-size training
        self.mask_cache = MaskCache(self.n, self.attn_backend, device=None)

    def _encode_x0(self, x0_tokens):
        """Encode x_0 tokens into key-value representations.
        
        This is done WITHOUT gradient (detached) to save memory.
        The x_0 embeddings serve as context for cross-attention.
        
        Args:
            x0_tokens: [B, N] int tensor of clean token indices
        
        Returns:
            x0_embed: [B, N, D] detached x_0 embeddings
        """
        with torch.no_grad():
            x0_embed = self.vocab_embed(x0_tokens)
            x0_embed = self.x0_proj(x0_embed)
        return x0_embed.detach()

    def gen_mask(self, seq_len, block_size, attn_backend=None, device=None):
        """Generate attention masks for a given block size."""
        if attn_backend is None:
            attn_backend = self.attn_backend
        return self.mask_cache.get_masks(block_size, device=device)

    def reset_kv_cache(self, eval_batch_size=None):
        """Reset KV caches for all blocks."""
        for block in self.blocks:
            block.kv_cache = None
            block.x0_kv_cache = None
            block.cache_idx = 0

    def forward(self, x_t, sigma, x0=None, prev_x0_hat=None,
                block_size=None, sample_mode=False, store_kv=False):
        """
        Forward pass for ABD3.
        
        Args:
            x_t: [B, N] int tensor - noised tokens (N tokens, NOT 2N)
            sigma: [B] float tensor - per-sample noise level (or per-block)
            x0: [B, N] int tensor - clean tokens for cross-attention (training only)
            prev_x0_hat: [B, N] int tensor - previous x̂_0 prediction (self-conditioning)
            block_size: int - override block size (for mixed block-size training)
            sample_mode: bool - sampling mode (use cached KV)
            store_kv: bool - whether to store KV cache
        
        Returns:
            logits: [B, N, V] - predicted token logits
        """
        if block_size is None:
            block_size = self.block_size

        # Embed x_t
        x = self.vocab_embed(x_t)

        # Self-conditioning: add projected previous prediction (NEW in ABD3)
        if prev_x0_hat is not None:
            with torch.no_grad():
                prev_embed = self.vocab_embed(prev_x0_hat)
            x = x + self.self_cond_proj(prev_embed)

        # Encode x_0 for cross-attention (detached, no gradient)
        x0_embed = None
        if x0 is not None:
            x0_embed = self._encode_x0(x0)

        # Time conditioning
        if sigma is not None:
            t_cond = F.silu(self.sigma_map(sigma))
        else:
            t_cond = None

        # Positional embeddings (only N tokens, not 2N!)
        rotary_cos_sin = self.rotary_emb(x)

        # Get masks for current block size
        masks = self.mask_cache.get_masks(block_size, device=x.device)
        self_attn_mask = masks['self_attn_mask'] if not sample_mode else None
        cross_attn_mask = masks['cross_attn_mask'] if not sample_mode else None

        # Transformer forward (N tokens through self-attn + cross-attn)
        for block in self.blocks:
            x = block(
                x, rotary_cos_sin, c=t_cond,
                x0_embed=x0_embed,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask)
        x = self.output_layer(x, t_cond)

        return x  # [B, N, V] - only N tokens, not 2N!
