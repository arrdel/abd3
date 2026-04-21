"""Tests for two-stream architecture."""

import torch
import pytest


def test_two_stream_output_shape():
    """Output should be [B, N, V], NOT [B, 2N, V]."""
    from abd3.models.dit import ABD3DIT
    import omegaconf

    config = omegaconf.OmegaConf.create({
        'model': {'length': 32, 'hidden_size': 64, 'n_heads': 4,
                  'n_blocks': 1, 'cond_dim': 32, 'dropout': 0.0,
                  'attn_backend': 'sdpa', 'tie_word_embeddings': False,
                  'max_seqlen': 32},
        'block_size': 4,
        'algo': {'cross_attn': True, 'parameterization': 'subs'},
        'loader': {'eval_batch_size': 2}
    })
    model = ABD3DIT(config, vocab_size=50)
    x_t = torch.randint(0, 50, (2, 32))
    sigma = torch.tensor([0.5, 0.5])
    x0 = torch.randint(0, 50, (2, 32))

    out = model(x_t, sigma, x0=x0)
    assert out.shape == (2, 32, 50), \
        f"Expected (2, 32, 50) but got {out.shape}. Two-stream should output N, not 2N!"


def test_x0_encoding_is_detached():
    """x_0 encoding should not receive gradients."""
    from abd3.models.dit import ABD3DIT
    import omegaconf

    config = omegaconf.OmegaConf.create({
        'model': {'length': 16, 'hidden_size': 32, 'n_heads': 2,
                  'n_blocks': 1, 'cond_dim': 16, 'dropout': 0.0,
                  'attn_backend': 'sdpa', 'tie_word_embeddings': False,
                  'max_seqlen': 16},
        'block_size': 4,
        'algo': {'cross_attn': True, 'parameterization': 'subs'},
        'loader': {'eval_batch_size': 2}
    })
    model = ABD3DIT(config, vocab_size=30)
    x0 = torch.randint(0, 30, (2, 16))
    embed = model._encode_x0(x0)
    assert not embed.requires_grad, "x_0 embeddings should be detached (no gradient)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
