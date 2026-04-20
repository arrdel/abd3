"""
ABD3: Block Diffusion with Recurrent Draft Refinement.

Hydra entry point for training, evaluation, and sampling.
"""

import os
import sys

import hydra
import lightning as L
import omegaconf
import torch
import transformers

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abd3 import diffusion, dataloader

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(config: omegaconf.DictConfig):
    """Main training/evaluation entry point."""
    L.seed_everything(config.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.mode == 'train':
        train(config, tokenizer)
    elif config.mode == 'eval':
        evaluate(config, tokenizer)
    elif config.mode == 'sample':
        sample(config, tokenizer)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def train(config, tokenizer):
    """Training loop."""
    model = diffusion.ABD3Diffusion(config, tokenizer=tokenizer)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"ABD3 Model Summary")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Block size:       {config.block_size}")
    print(f"  Seq length:       {config.model.length}")
    print(f"  RDR (self-cond):  {config.algo.self_conditioning}")
    print(f"  Mixed blocks:     {config.algo.mixed_block_sizes}")
    print(f"  Adaptive stop:    {config.algo.adaptive_stopping}")
    print(f"  Time conditioning:{config.algo.time_conditioning}")
    print(f"{'='*60}\n")

    train_dl, val_dl = dataloader.get_dataloaders(config, tokenizer)

    callbacks = []
    if config.checkpointing.save_dir:
        callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
            dirpath=config.checkpointing.save_dir,
            every_n_train_steps=config.checkpointing.save_every,
            save_top_k=3,
            monitor='val/loss',
            mode='min'))

    # Learning rate monitor
    callbacks.append(L.pytorch.callbacks.LearningRateMonitor(logging_interval='step'))

    logger = None
    if hasattr(config, 'wandb') and config.wandb.enabled:
        logger = L.pytorch.loggers.WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name)
    else:
        logger = L.pytorch.loggers.CSVLogger(
            save_dir=os.getcwd(),
            name='logs')

    # When training on >1 GPU, DDP needs `find_unused_parameters=True` because
    # `self_cond_proj` doesn't receive gradients on steps where the model is
    # invoked without `prev_x0_hat` (self-conditioning dropout).
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator='auto',
        devices='auto',
        strategy=strategy,
        precision=config.training.precision if hasattr(config.training, 'precision') else 'bf16-mixed',
        gradient_clip_val=config.optim.grad_clip if hasattr(config.optim, 'grad_clip') else 1.0,
        accumulate_grad_batches=config.training.accum if hasattr(config.training, 'accum') else 1,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=config.training.val_check_interval if hasattr(config.training, 'val_check_interval') else 1.0,
        log_every_n_steps=50,
        enable_progress_bar=True)

    trainer.fit(model, train_dl, val_dl)


def evaluate(config, tokenizer):
    """Evaluation."""
    model = diffusion.ABD3Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config, tokenizer=tokenizer,
        strict=False, weights_only=False).to('cuda')
    model.eval()

    _, val_dl = dataloader.get_dataloaders(config, tokenizer)

    trainer = L.Trainer(accelerator='auto', devices=1)
    trainer.validate(model, val_dl)


def sample(config, tokenizer):
    """Generate text samples."""
    model = diffusion.ABD3Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config, tokenizer=tokenizer,
        strict=False, weights_only=False).to('cuda')
    model.eval()

    texts = model.restore_model_and_sample()
    print("\n=== Generated Samples ===")
    for i, text in enumerate(texts):
        print(f"\n--- Sample {i+1} ---")
        print(text[:500])


if __name__ == '__main__':
    main()
