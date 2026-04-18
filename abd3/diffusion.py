"""
ABD3 Diffusion: Core diffusion logic with Recurrent Draft Refinement (RDR).

RDR combines self-conditioning + adaptive stopping into a unified technique:
  - Inner loop: T draft-refinement steps within each block (discrete argmax feedback)
  - Outer loop: autoregressive block generation with cross-attention context
  - Draft convergence triggers adaptive early stopping

Key changes from BD3-LMs:
1. RDR: Recurrent Draft Refinement (self-conditioning + adaptive stopping)
2. Two-stream: x_t (N tokens) + cross-attention to x_0 (no 2N concat)
3. Mixed block-size training: Random block_size ∈ {1,2,4,8,16} per batch
4. Per-block time conditioning: adaLN receives actual sigma, not zero
"""

import itertools
import math
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from . import noise_schedule, utils
from .models import dit as models_dit
from .models import ema as models_ema
from .models.attention import sample_block_size, BLOCK_SIZE_CHOICES


def _sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


class ABD3Diffusion(L.LightningModule):
    """ABD3: Self-Conditioned Block Diffusion with Adaptive Block Sizes.
    
    Innovations over BD3-LMs:
    1. Self-conditioning within blocks
    2. Efficient two-stream architecture (N tokens, not 2N)
    3. Mixed block-size training
    4. Adaptive early stopping during inference
    5. Per-block time conditioning (sigma ≠ 0)
    """

    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # Mask token setup
        if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id

        self.parameterization = config.algo.parameterization
        self.num_tokens = config.model.length
        self.T = config.algo.T
        self.neg_infinity = -1e9

        # Block size configuration
        self.block_size = getattr(config, 'block_size', config.model.length)
        self.mixed_block_sizes = getattr(config.algo, 'mixed_block_sizes', False)
        self.block_size_choices = getattr(config.algo, 'block_size_choices', BLOCK_SIZE_CHOICES)

        # Self-conditioning (NEW in ABD3)
        self.self_conditioning = getattr(config.algo, 'self_conditioning', True)
        self.self_cond_prob = getattr(config.algo, 'self_cond_prob', 0.5)

        # Per-block time conditioning (NEW in ABD3 - enabled by default)
        self.time_conditioning = getattr(config.algo, 'time_conditioning', True)

        # Adaptive early stopping (NEW in ABD3)
        self.adaptive_stopping = getattr(config.algo, 'adaptive_stopping', True)
        self.stop_entropy_threshold = getattr(config.algo, 'stop_entropy_threshold', 0.1)
        self.stop_agreement_threshold = getattr(config.algo, 'stop_agreement_threshold', 2)

        # Antithetic sampling
        self.antithetic_sampling = config.training.antithetic_sampling

        # Build backbone (two-stream DIT)
        self.backbone = models_dit.ABD3DIT(config, vocab_size=self.vocab_size)

        # Noise schedule
        self.noise = noise_schedule.get_noise(config)

        # EMA
        if config.training.ema > 0:
            self.ema = models_ema.ExponentialMovingAverage(
                self._get_parameters(), decay=config.training.ema)
        else:
            self.ema = None

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(), self.noise.parameters())

    # ========================================================================
    # Parameterization
    # ========================================================================

    def _subs_parameterization(self, logits, xt):
        """Substitution parameterization: lock unmasked tokens."""
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = (xt != self.mask_index)
        logits[unmasked] = self.neg_infinity
        logits[unmasked, xt[unmasked]] = 0
        return logits

    def _process_sigma(self, sigma):
        """Process sigma for time conditioning.
        
        ABD3 change: We keep actual sigma values (per-block conditioning)
        instead of zeroing them out like BD3-LMs.
        """
        if self.parameterization == 'ar':
            return None
        if sigma.ndim == 2:
            sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        # ABD3: Keep actual sigma for time conditioning (BD3-LMs zeros this out!)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def _sigma_from_p(self, p):
        return -torch.log1p(-p)

    # ========================================================================
    # Forward pass
    # ========================================================================

    def forward(self, x_t, sigma, x0=None, prev_x0_hat=None,
                block_size=None, sample_mode=False, store_kv=False):
        """Forward pass through the two-stream model.
        
        Args:
            x_t: [B, N] noised tokens (N tokens, NOT 2N!)
            sigma: noise level
            x0: [B, N] clean tokens for cross-attention
            prev_x0_hat: [B, N] previous prediction for RDR (draft refinement)
            block_size: override block size
            sample_mode: inference mode
            store_kv: cache KV
        
        Returns:
            log probabilities [B, N, V]
        """
        sigma = self._process_sigma(sigma)

        logits = self.backbone(
            x_t, sigma, x0=x0, prev_x0_hat=prev_x0_hat,
            block_size=block_size, sample_mode=sample_mode,
            store_kv=store_kv)

        if self.parameterization == 'subs':
            return self._subs_parameterization(logits=logits, xt=x_t)
        return logits

    # ========================================================================
    # Noise sampling (with mixed block sizes)
    # ========================================================================

    def q_xt(self, x0, move_chance):
        """Sample noised x_t from clean x_0."""
        move_indices = torch.rand(*x0.shape, device=x0.device) < move_chance
        x_t = torch.where(move_indices, self.mask_index, x0)
        return x_t

    def _sample_t(self, shape, device, block_size=None,
                  sampling_eps_min=1e-3, sampling_eps_max=1.0):
        """Sample timesteps, one per block."""
        if block_size is None:
            block_size = self.block_size
        batch_size = shape[0]
        seq_len = shape[1]
        num_blocks = seq_len // block_size

        _eps_b = torch.rand(batch_size, num_blocks, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device) / (batch_size * num_blocks)
            offset = offset.view(batch_size, num_blocks)
            _eps_b = (_eps_b / (batch_size * num_blocks) + offset) % 1

        t = _eps_b
        if block_size != seq_len:
            t = t.repeat_interleave(block_size, dim=-1)

        t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
        return t

    # ========================================================================
    # Training loss (with self-conditioning + mixed block sizes)
    # ========================================================================

    def _forward_pass_diffusion(self, x0, block_size=None,
                                 sampling_eps_min=1e-3, sampling_eps_max=1.0):
        """Compute diffusion training loss with RDR innovations."""
        if block_size is None:
            block_size = self.block_size

        # Sample timesteps (one per block, repeated across block tokens)
        t = self._sample_t(x0.shape, x0.device, block_size=block_size,
                           sampling_eps_min=sampling_eps_min,
                           sampling_eps_max=sampling_eps_max)

        loss_scale, p = self.noise(t)

        # Compute per-block sigma for time conditioning
        # Take one value per block (all tokens in a block share the same t)
        num_blocks = x0.shape[1] // block_size
        if num_blocks > 1 and block_size < x0.shape[1]:
            # p has shape [B, N], take first token of each block
            p_per_block = p[:, ::block_size]  # [B, num_blocks]
            sigma = self._sigma_from_p(p_per_block.mean(dim=-1))  # [B]
        else:
            sigma = self._sigma_from_p(p.mean(dim=-1))  # [B]

        # Sample noised x_t
        x_t = self.q_xt(x0, p)

        # ---- RDR: Recurrent Draft Refinement (Innovation #1) ----
        # With probability self_cond_prob, generate a draft and feed it back
        prev_x0_hat = None
        if self.self_conditioning and self.training:
            if torch.rand(1).item() < self.self_cond_prob:
                with torch.no_grad():
                    logits_init = self.forward(
                        x_t, sigma, x0=x0, prev_x0_hat=None,
                        block_size=block_size)
                    prev_x0_hat = logits_init.argmax(dim=-1).detach()

        # ---- Two-stream forward (Innovation #2) ----
        model_output = self.forward(
            x_t, sigma, x0=x0, prev_x0_hat=prev_x0_hat,
            block_size=block_size)

        utils.print_nans(model_output, 'model_output')

        # Compute loss
        log_p_theta = torch.gather(
            input=model_output, dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        loss = loss_scale * log_p_theta
        return loss

    def _loss(self, x0, attention_mask, sampling_eps_min=None, sampling_eps_max=None):
        if sampling_eps_min is None:
            sampling_eps_min = getattr(self.config.training, 'sampling_eps', 1e-3)
        if sampling_eps_max is None:
            sampling_eps_max = 1.0

        # ---- Mixed block-size training (Innovation #3) ----
        if self.mixed_block_sizes and self.training:
            block_size = sample_block_size(self.block_size_choices)
            # Ensure seq_len is divisible by block_size
            seq_len = x0.shape[1]
            if seq_len % block_size != 0:
                # Truncate to nearest multiple
                new_len = (seq_len // block_size) * block_size
                x0 = x0[:, :new_len]
                attention_mask = attention_mask[:, :new_len]
        else:
            block_size = self.block_size

        loss = self._forward_pass_diffusion(
            x0, block_size=block_size,
            sampling_eps_min=sampling_eps_min,
            sampling_eps_max=sampling_eps_max)

        nlls = loss * attention_mask
        token_nll = nlls.sum() / attention_mask.sum()
        return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)

    # ========================================================================
    # Training hooks
    # ========================================================================

    def training_step(self, batch, batch_idx):
        losses = self._loss(batch['input_ids'], batch['attention_mask'])
        self.log('train/loss', losses.loss.item(), on_step=True, on_epoch=False,
                 sync_dist=True, prog_bar=True)
        return losses.loss

    def validation_step(self, batch, batch_idx):
        losses = self._loss(batch['input_ids'], batch['attention_mask'])
        self.log('val/loss', losses.loss.item(), on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        return losses.loss

    def on_validation_epoch_end(self):
        pass  # Metrics logged automatically via self.log in validation_step

    def configure_optimizers(self):
        # Separate weight decay for non-bias/norm params
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.config.optim.lr,
           betas=(self.config.optim.beta1, self.config.optim.beta2))

        # Linear warmup + cosine decay
        warmup_steps = getattr(self.config.optim, 'warmup_steps', 1000)
        max_steps = self.config.training.max_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    # ========================================================================
    # Sampling with self-conditioning + adaptive stopping
    # ========================================================================

    def _sample_prior(self, n_samples, seq_len):
        return torch.full((n_samples, seq_len), self.mask_index,
                          dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _ddpm_update_with_self_cond(self, x, t, dt, block_size,
                                     prev_x0_hat=None, past_x0=None):
        """Single DDPM denoising step with self-conditioning.
        
        Args:
            x: [B, N] current noised sequence
            t: [B, 1] current timestep
            dt: timestep delta
            block_size: current block size
            prev_x0_hat: [B, block_size] previous x̂_0 prediction (self-conditioning)
            past_x0: [B, N-block_size] decoded past blocks for cross-attention
        
        Returns:
            (new_x0_hat, x_next): updated prediction and next state
        """
        _, move_chance_t = self.noise(t)
        _, move_chance_s = self.noise(t - dt)
        sigma_t = self._sigma_from_p(move_chance_t)
        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]
        mask_prob = move_chance_s / move_chance_t

        # Build full x0 context for cross-attention
        x0_context = None
        if past_x0 is not None:
            # Pad with masks for current block
            x0_context = torch.cat([
                past_x0,
                torch.full((x.shape[0], block_size), self.mask_index,
                           device=x.device, dtype=torch.long)
            ], dim=1)

        # Forward pass with self-conditioning (Innovation #1 at inference)
        x_block = x[:, -block_size:]

        # Build prev_x0_hat for full sequence if needed
        full_prev = None
        if prev_x0_hat is not None:
            if past_x0 is not None:
                full_prev = torch.cat([past_x0, prev_x0_hat], dim=1)
            else:
                full_prev = prev_x0_hat

        p_x0 = self.forward(
            x, sigma_t, x0=x0_context, prev_x0_hat=full_prev,
            block_size=block_size, sample_mode=True)
        p_x0 = p_x0[:, -block_size:].to(torch.float64).exp()

        # Get new x0_hat prediction for next step's self-conditioning
        new_x0_hat = p_x0.argmax(dim=-1)

        # DDPM update
        q_xs = p_x0 * (1 - mask_prob)
        q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
        x_block_new = _sample_categorical(q_xs)

        # Keep unmasked tokens
        copy_flag = (x[:, -block_size:] != self.mask_index).to(x.dtype)
        x_block_new = copy_flag * x[:, -block_size:] + (1 - copy_flag) * x_block_new
        x_new = torch.cat([x[:, :-block_size], x_block_new], dim=-1)

        return new_x0_hat, x_new

    @torch.no_grad()
    def _check_early_stop(self, p_x0, prev_prediction, step):
        """Check if denoising should stop early (Innovation #4).
        
        Criteria:
        1. Entropy of predictions is below threshold
        2. Consecutive predictions agree for N steps
        
        Args:
            p_x0: [B, block_size, V] current prediction probabilities
            prev_prediction: [B, block_size] previous argmax prediction
            step: current step number
        
        Returns:
            bool: whether to stop early
        """
        if not self.adaptive_stopping or step < 3:
            return False

        # Criterion 1: Low entropy
        entropy = -(p_x0 * (p_x0 + 1e-10).log()).sum(-1).mean()
        if entropy < self.stop_entropy_threshold:
            return True

        # Criterion 2: Agreement with previous prediction
        if prev_prediction is not None:
            current = p_x0.argmax(dim=-1)
            agreement = (current == prev_prediction).all()
            if agreement:
                return True

        return False

    @torch.no_grad()
    def sample(self, n_samples, num_steps=None, block_size=None):
        """Generate samples with all ABD3 innovations.
        
        Args:
            n_samples: number of sequences to generate
            num_steps: denoising steps per block (can be reduced by adaptive stopping)
            block_size: block size for generation (can differ from training)
        """
        if num_steps is None:
            num_steps = self.T
        if block_size is None:
            block_size = self.block_size

        seq_len = self.num_tokens
        num_blocks = seq_len // block_size

        # Start fully masked
        x = self._sample_prior(n_samples, seq_len)
        past_x0 = None
        total_nfes = 0

        ones = torch.ones((n_samples, 1), device=self.device)

        for block_idx in tqdm(range(num_blocks), desc='blocks'):
            # Initialize block as masked
            eps = 1e-5
            timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
            dt = (1 - eps) / num_steps

            prev_x0_hat = None  # For self-conditioning
            prev_prediction = None  # For adaptive stopping
            consecutive_agreements = 0

            for step_idx in range(num_steps):
                t = timesteps[step_idx] * ones

                prev_x0_hat, x = self._ddpm_update_with_self_cond(
                    x, t, dt, block_size,
                    prev_x0_hat=prev_x0_hat,
                    past_x0=past_x0)
                total_nfes += 1

                # ---- Adaptive early stopping (Innovation #4) ----
                if self.adaptive_stopping and step_idx > 2:
                    if prev_prediction is not None and (prev_x0_hat == prev_prediction).all():
                        consecutive_agreements += 1
                    else:
                        consecutive_agreements = 0

                    if consecutive_agreements >= self.stop_agreement_threshold:
                        break

                prev_prediction = prev_x0_hat.clone()

            # Block is done — update past context
            block_start = block_idx * block_size
            block_end = (block_idx + 1) * block_size
            if past_x0 is None:
                past_x0 = x[:, block_start:block_end].clone()
            else:
                past_x0 = torch.cat([past_x0, x[:, block_start:block_end]], dim=1)

        return x, total_nfes

    @torch.no_grad()
    def restore_model_and_sample(self, num_steps=None):
        """Sample with EMA model."""
        if self.ema:
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())

        samples, nfes = self.sample(
            n_samples=self.config.loader.eval_batch_size,
            num_steps=num_steps)

        text_samples = self.tokenizer.batch_decode(samples)

        if self.ema:
            self.ema.restore(self._get_parameters())

        return text_samples
