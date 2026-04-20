"""
Quick smoke tester for the feasibility configuration. Runs a single training step
on CPU to validate wiring (model forward + loss computation).

Usage:
    python tools/smoke_run.py

This is for local validation during development; it does not require GPUs.
"""
import torch
from hydra import compose, initialize_config_dir
import os
from abd3 import diffusion, dataloader
import transformers


def main():
    config_dir = os.path.abspath('configs')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='feasibility')
    cfg.loader.batch_size = 2
    cfg.loader.eval_batch_size = 2
    cfg.training.max_steps = 1

    tok = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_dl, val_dl = dataloader.get_dataloaders(cfg, tok)
    model = diffusion.ABD3Diffusion(cfg, tokenizer=tok)
    model.train()

    batch = next(iter(train_dl))
    # move to cpu
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to('cpu')

    loss = model.training_step(batch, 0)
    print('Smoke step loss:', loss)


if __name__ == '__main__':
    main()
