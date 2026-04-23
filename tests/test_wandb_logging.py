"""Tests for the WandB integration scaffolding in abd3/main.py and
the per-block-size logging bookkeeping in abd3/diffusion.py.

These tests stay offline: we never instantiate a real WandbLogger — we
only check the helpers that pick a run name, derive tags, and choose
between CSVLogger/WandbLogger/no-op fallback. The diffusion-side
checks use a tiny real ABD3Diffusion (same pattern as test_sampler)
so they catch regressions in the mixed-block-size bookkeeping that
``training_step`` relies on.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys

import lightning as L
import omegaconf
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abd3 import main as abd3_main  # noqa: E402
from abd3.diffusion import ABD3Diffusion  # noqa: E402

# ---------------------------------------------------------------------------
# Run-name / tag helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> omegaconf.DictConfig:
    base = {
        "wandb": {"enabled": False, "project": "abd3-test", "name": "abd3-run"},
        "algo": {
            "self_conditioning": False,
            "mixed_block_sizes": False,
            "adaptive_stopping": False,
        },
        "data": {"name": "wikitext"},
        "block_size": 4,
        "training": {"max_steps": 500},
    }
    cfg = omegaconf.OmegaConf.create(base)
    for path, value in overrides.items():
        omegaconf.OmegaConf.update(cfg, path, value, force_add=True)
    return cfg


def test_run_name_uses_custom_name_when_meaningful():
    cfg = _cfg(**{"wandb.name": "my-cool-run"})
    assert abd3_main._resolve_run_name(cfg) == "my-cool-run"


def test_run_name_synthesizes_from_flags():
    cfg = _cfg(
        **{
            "algo.self_conditioning": True,
            "algo.mixed_block_sizes": True,
            "algo.adaptive_stopping": False,
            "block_size": 8,
            "training.max_steps": 5000,
        }
    )
    name = abd3_main._resolve_run_name(cfg)
    assert name.startswith("abd3-wikitext-bs8-")
    assert "rdr" in name and "mix" in name
    assert "astop" not in name  # disabled flag shouldn't show up
    assert "-s5000-" in name


def test_run_name_defaults_tag_is_base_when_no_flags():
    cfg = _cfg()
    name = abd3_main._resolve_run_name(cfg)
    assert "-base-" in name


def test_resolve_run_tags_baseline_vs_rdr():
    base_tags = abd3_main._resolve_run_tags(_cfg())
    rdr_tags = abd3_main._resolve_run_tags(_cfg(**{"algo.self_conditioning": True}))
    assert "baseline" in base_tags
    assert "rdr" in rdr_tags
    assert "baseline" not in rdr_tags


def test_resolve_run_tags_includes_block_size_and_data():
    tags = abd3_main._resolve_run_tags(_cfg(**{"block_size": 16, "data.name": "openwebtext"}))
    assert any(t == "bs=16" for t in tags)
    assert any(t.startswith("data=openwebtext") for t in tags)


# ---------------------------------------------------------------------------
# Logger selection
# ---------------------------------------------------------------------------


def test_build_logger_defaults_to_csv_when_wandb_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = abd3_main._build_logger(_cfg())
    assert isinstance(logger, L.pytorch.loggers.CSVLogger)


def test_build_logger_respects_env_override(tmp_path, monkeypatch):
    """ABD3_DISABLE_WANDB=1 must downgrade to CSVLogger even if the yaml
    enables wandb. Used in CI / dry-runs where we don't want a wandb session."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ABD3_DISABLE_WANDB", "1")
    logger = abd3_main._build_logger(_cfg(**{"wandb.enabled": True}))
    assert isinstance(logger, L.pytorch.loggers.CSVLogger)


def test_build_logger_falls_back_when_wandb_missing(tmp_path, monkeypatch):
    """Pretending wandb isn't importable must trigger the CSVLogger fallback
    with a printed warning, not a hard crash."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ABD3_DISABLE_WANDB", raising=False)

    # Block the ``import wandb`` inside _build_logger without touching
    # the real module (if installed). Done via sys.modules since _build_logger
    # uses a plain `import wandb` rather than importlib.
    real_wandb = sys.modules.pop("wandb", None)
    sys.modules["wandb"] = None  # type: ignore[assignment]
    try:
        logger = abd3_main._build_logger(_cfg(**{"wandb.enabled": True}))
    finally:
        if real_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = real_wandb

    assert isinstance(logger, L.pytorch.loggers.CSVLogger)


# ---------------------------------------------------------------------------
# Per-block-size bookkeeping inside _loss
# ---------------------------------------------------------------------------


class _MockTokenizer:
    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size
        self.mask_token = None


def _tiny_diffusion(mixed: bool):
    cfg = omegaconf.OmegaConf.create(
        {
            "model": {
                "length": 8,
                "hidden_size": 16,
                "n_heads": 2,
                "n_blocks": 1,
                "cond_dim": 16,
                "dropout": 0.0,
                "attn_backend": "sdpa",
                "tie_word_embeddings": False,
                "max_seqlen": 8,
            },
            "block_size": 2,
            "algo": {
                "parameterization": "subs",
                "T": 5,
                "cross_attn": True,
                "self_conditioning": False,
                "self_cond_prob": 0.5,
                "time_conditioning": True,
                "mixed_block_sizes": mixed,
                "block_size_choices": [2, 4],
                "adaptive_stopping": False,
                "stop_entropy_threshold": 0.1,
                "stop_agreement_threshold": 2,
            },
            "noise": {"type": "loglinear"},
            "loader": {"eval_batch_size": 2},
            "training": {"ema": 0, "antithetic_sampling": False, "sampling_eps": 1e-3},
        }
    )
    return ABD3Diffusion(cfg, tokenizer=_MockTokenizer())


def test_last_loss_block_size_matches_fixed_config_when_not_mixed():
    model = _tiny_diffusion(mixed=False)
    model.train()
    x = torch.randint(0, model.vocab_size - 1, (2, 8))
    mask = torch.ones_like(x)
    model._loss(x, mask)
    assert model._last_loss_block_size == 2  # config block_size


def test_last_loss_block_size_is_one_of_choices_when_mixed():
    """In mixed-block training the stashed value must be one of the
    configured choices so per-bs logging keys don't explode."""
    model = _tiny_diffusion(mixed=True)
    model.train()
    seen: set[int] = set()
    # Run a few times to collect at least two different block sizes with
    # the given choices [2, 4]. Flaky-safe cap at 50 trials.
    for _ in range(50):
        x = torch.randint(0, model.vocab_size - 1, (2, 8))
        mask = torch.ones_like(x)
        model._loss(x, mask)
        seen.add(model._last_loss_block_size)
        if len(seen) >= 2:
            break
    assert seen.issubset({2, 4})
    assert len(seen) >= 1


# ---------------------------------------------------------------------------
# training_step / on_before_optimizer_step logging emission
# ---------------------------------------------------------------------------


class _RecordingLogger:
    """Bare minimum stand-in for pl.Logger that captures ``self.log`` calls."""

    def __init__(self):
        self.calls: list[tuple[str, float]] = []

    def log_metrics(self, metrics, step=None):  # pragma: no cover
        for k, v in metrics.items():
            self.calls.append((k, float(v)))


def test_training_step_emits_expected_keys():
    model = _tiny_diffusion(mixed=True)
    model.train()
    captured: list[tuple[str, float]] = []

    def fake_log(name, value, *args, **kwargs):
        # Skip non-numeric / nested objects just to be safe.
        with contextlib.suppress(TypeError, ValueError):
            captured.append((name, float(value)))

    model.log = fake_log  # type: ignore[assignment]

    batch = {
        "input_ids": torch.randint(0, model.vocab_size - 1, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model.training_step(batch, 0)

    keys = [k for k, _ in captured]
    assert "train/loss" in keys
    assert "train/ppl" in keys
    assert "train/block_size" in keys
    # At least one per-block-size key emitted during mixed training.
    assert any(k.startswith("train/loss_bs") for k in keys)


def test_on_before_optimizer_step_emits_grad_norm():
    model = _tiny_diffusion(mixed=False)
    model.train()
    captured: list[tuple[str, float]] = []

    def fake_log(name, value, *args, **kwargs):
        captured.append((name, float(value)))

    model.log = fake_log  # type: ignore[assignment]

    x = torch.randint(0, model.vocab_size - 1, (2, 8))
    mask = torch.ones_like(x)
    losses = model._loss(x, mask)
    losses.loss.backward()

    model.on_before_optimizer_step(optimizer=None)
    keys = [k for k, _ in captured]
    assert "train/grad_norm" in keys
    gn = next(v for k, v in captured if k == "train/grad_norm")
    assert gn > 0.0


def test_bounded_ppl_caps_large_losses():
    """bounded_ppl must not emit inf even when the loss is enormous
    (e.g. during first optimisation steps). inf pollutes WandB auto-scaling."""
    huge = ABD3Diffusion._bounded_ppl(1e9)
    assert huge != float("inf")
    assert huge > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
