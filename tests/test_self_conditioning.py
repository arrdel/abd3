"""Tests for self-conditioning mechanism."""

import torch
import pytest


def test_self_cond_zero_init():
    """Self-conditioning projection should be zero-initialized (no-op initially)."""
    from abd3.models.dit import ABD3DIT
    import omegaconf

    config = omegaconf.OmegaConf.create({
        'model': {'length': 64, 'hidden_size': 128, 'n_heads': 4,
                  'n_blocks': 2, 'cond_dim': 64, 'dropout': 0.0,
                  'attn_backend': 'sdpa', 'tie_word_embeddings': False,
                  'max_seqlen': 64},
        'block_size': 4,
        'algo': {'cross_attn': True, 'parameterization': 'subs'},
        'loader': {'eval_batch_size': 2}
    })
    model = ABD3DIT(config, vocab_size=100)
    assert (model.self_cond_proj.weight == 0).all(), \
        "Self-cond projection should be zero-initialized"


def test_self_cond_changes_output():
    """With non-zero self-cond weights, prev_x0_hat should affect output."""
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
    torch.manual_seed(0)
    model = ABD3DIT(config, vocab_size=50)
    # Make self-cond non-trivial. Use a random (non-degenerate) weight so the
    # projection is not rank-1 (a constant-per-token vector would be erased by
    # the downstream LayerNorm). Also break adaLN-Zero init on the final output
    # layer so the self-cond signal can actually propagate to logits.
    torch.nn.init.normal_(model.self_cond_proj.weight, std=0.1)
    torch.nn.init.normal_(model.output_layer.linear.weight, std=0.02)

    x_t = torch.randint(0, 50, (2, 32))
    sigma = torch.tensor([0.5, 0.5])
    x0 = torch.randint(0, 50, (2, 32))

    out_no_sc = model(x_t, sigma, x0=x0, prev_x0_hat=None)
    out_with_sc = model(x_t, sigma, x0=x0, prev_x0_hat=x0)

    assert not torch.allclose(out_no_sc, out_with_sc, atol=1e-3), \
        "Self-conditioning should change model output"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
