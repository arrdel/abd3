"""Tests for eval/generate.py record-construction logic.

We stub out ``ABD3Diffusion.sample`` so the tests don't need a full checkpoint
or tokenizer download. The things under test are the parts of the pipeline
that are easy to get wrong and that downstream quality eval consumes:

  * NFE bookkeeping: total_nfe == sum(per_block_nfe); theoretical_nfe and
    nfe_savings are computed from the model's num_blocks * num_steps.
  * Per-sample records carry the correct sample_id, shared NFE trace, and
    decoded text.
  * Sampler name reflects the model's self-conditioning + adaptive-stopping
    flags — so ablation tables can't silently mis-label their rows.
"""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.generate import SampleRecord, generate  # noqa: E402


class _FakeTokenizer:
    def batch_decode(self, rows, skip_special_tokens=False):
        return [f"decoded<{len(r)}tok>" for r in rows]


def _make_model(
    *,
    seq_len: int = 32,
    block_size: int = 4,
    T: int = 10,
    self_cond: bool = True,
    adaptive: bool = True,
    per_block_nfe: list[int] | None = None,
) -> MagicMock:
    m = MagicMock()
    m.num_tokens = seq_len
    m.block_size = block_size
    m.T = T
    m.self_conditioning = self_cond
    m.adaptive_stopping = adaptive

    num_blocks = seq_len // block_size
    if per_block_nfe is None:
        per_block_nfe = [T] * num_blocks

    def _sample(n_samples, num_steps, block_size, track_nfe_per_block, progress):
        assert track_nfe_per_block, "generate() must request NFE tracking"
        # Fake output ids — each row is a count-up sequence so we can spot
        # wrong slicing in the decoded text.
        x = torch.arange(seq_len).unsqueeze(0).repeat(n_samples, 1)
        total = sum(per_block_nfe)
        return x, total, list(per_block_nfe)

    m.sample.side_effect = _sample
    return m


def test_record_structure_and_nfe_accounting():
    model = _make_model(per_block_nfe=[10, 10, 7, 3, 3, 3, 3, 3])  # 8 blocks
    tok = _FakeTokenizer()
    records = generate(
        model,
        tok,
        n_samples=3,
        num_steps=10,
        block_size=4,
        seed=7,
        algo_name="abd3",
        checkpoint_path="/tmp/ckpt",
    )

    assert len(records) == 3
    for i, r in enumerate(records):
        assert isinstance(r, SampleRecord)
        assert r.sample_id == i
        assert r.text == "decoded<32tok>"
        assert r.block_size == 4
        assert r.num_steps == 10
        # Total NFE equals the sum of per-block NFEs emitted by the sampler.
        assert r.total_nfe == sum([10, 10, 7, 3, 3, 3, 3, 3])
        assert r.per_block_nfe == [10, 10, 7, 3, 3, 3, 3, 3]
        # Theoretical = num_blocks * num_steps = 8 * 10.
        assert r.theoretical_nfe == 80
        assert r.nfe_savings == pytest.approx(1 - 42 / 80)
        assert r.seed == 7
        assert r.algo == "abd3"
        assert r.checkpoint == "/tmp/ckpt"


def test_sampler_name_reflects_innovations():
    tok = _FakeTokenizer()

    m_full = _make_model(self_cond=True, adaptive=True)
    r_full = generate(m_full, tok, n_samples=1, num_steps=10, block_size=4)
    assert r_full[0].sampler == "ddpm_self_cond+adaptive_stop"

    m_base = _make_model(self_cond=False, adaptive=False)
    r_base = generate(m_base, tok, n_samples=1, num_steps=10, block_size=4)
    assert r_base[0].sampler == "ddpm"

    m_sc = _make_model(self_cond=True, adaptive=False)
    r_sc = generate(m_sc, tok, n_samples=1, num_steps=10, block_size=4)
    assert r_sc[0].sampler == "ddpm_self_cond"


def test_defaults_come_from_model_when_unset():
    model = _make_model(seq_len=64, block_size=8, T=20)
    tok = _FakeTokenizer()
    records = generate(model, tok, n_samples=1)  # no num_steps / block_size
    # num_blocks = 64 // 8 = 8; theoretical = 8 * 20 = 160.
    assert records[0].num_steps == 20
    assert records[0].block_size == 8
    assert records[0].theoretical_nfe == 160


def test_nfe_savings_zero_when_no_early_stop():
    # Every block uses all T steps => no savings.
    model = _make_model(per_block_nfe=[10] * 8)
    tok = _FakeTokenizer()
    records = generate(model, tok, n_samples=2, num_steps=10, block_size=4)
    for r in records:
        assert r.nfe_savings == pytest.approx(0.0)
        assert r.total_nfe == r.theoretical_nfe


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
