# ABD3: Self-Conditioned Block Diffusion with Adaptive Block Sizes for Language Modeling

> **Paper Title**: *ABD3: Self-Conditioned Block Diffusion with Adaptive Block Sizes for Language Modeling*

## Overview

ABD3 extends Block Discrete Denoising Diffusion (BD3-LMs) with four key innovations:

1. **Self-Conditioning within Blocks**: Feed back the model's previous x̂₀ prediction during denoising (50% probability during training, always during sampling), yielding 1–2 PPL improvement.

2. **Efficient Two-Stream Training**: Replace the 2N concatenation with cross-attention. Encode x₀ blocks once (no gradient), then the main model (N tokens) cross-attends to cached x₀ KV. Halves memory → 2× longer sequences or 2× larger batches.

3. **Mixed Block-Size Training**: Sample block_size ∈ {1, 2, 4, 8, 16} randomly each batch. Enables adaptive block sizing at inference and graceful interpolation from pure AR (L'=1) to MDLM (L'=N).

4. **Adaptive Early Stopping**: During inference, stop denoising a block when predictions stabilize (low entropy or consecutive agreement). Saves 30–60% NFEs on easy blocks.

## Project Structure

```
abd3/
├── README.md
├── intro.md                    # Technical analysis & gap identification
├── baselines/
│   └── bd3lms/                 # Original BD3-LMs codebase (kuleshov-group/bd3lms)
├── abd3/                       # Our method
│   ├── __init__.py
│   ├── main.py                 # Hydra entry point
│   ├── diffusion.py            # Core diffusion: self-conditioning, adaptive stopping
│   ├── noise_schedule.py       # Noise schedules (inherited + extensions)
│   ├── dataloader.py           # Data loading with mixed block-size support
│   ├── metrics.py              # Evaluation metrics
│   ├── utils.py                # Utilities
│   └── models/
│       ├── __init__.py
│       ├── dit.py              # Two-stream DIT with cross-attention
│       ├── ema.py              # EMA
│       └── attention.py        # Adaptive block masks, overlapping blocks
├── configs/
│   ├── config.yaml             # Main config
│   ├── algo/
│   │   └── abd3.yaml           # Algorithm config
│   ├── model/
│   │   └── dit_small.yaml      # Model architecture config
│   ├── data/
│   │   └── openwebtext.yaml    # Dataset config
│   └── noise/
│       └── loglinear.yaml      # Noise schedule config
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── tests/
│   ├── test_self_conditioning.py
│   ├── test_two_stream.py
│   ├── test_mixed_block_size.py
│   └── test_adaptive_stopping.py
└── requirements.txt
```

## Key Differences from BD3-LMs

| Feature | BD3-LMs | ABD3 (Ours) |
|---------|---------|-------------|
| Self-conditioning | ✗ | ✓ (50% training, always inference) |
| Architecture | 2N concat (x_t; x_0) | N tokens + cross-attn to cached x₀ KV |
| Block size | Fixed at training | Mixed {1,2,4,8,16}, adaptive at inference |
| Denoising steps | Fixed T per block | Adaptive early stopping |
| Time conditioning | Disabled (σ=0) | Per-block σ via adaLN |
| Memory (training) | O((2N)²) attention | O(N²) + O(N·N) cross-attn |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train
bash scripts/train.sh

# Evaluate
bash scripts/eval.sh
```

## Expected Results

- **PPL**: 20.73 → ~18–19 (closing gap to AR baseline at 17.54)
- **Memory**: 2× reduction during training
- **Inference**: 30–60% fewer NFEs with adaptive stopping
- **Flexibility**: Single model across block sizes 1–16

## Citation

```bibtex
@article{abd3_2026,
  title={ABD3: Self-Conditioned Block Diffusion with Adaptive Block Sizes for Language Modeling},
  author={},
  year={2026}
}
```
