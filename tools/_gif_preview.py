"""Throwaway helper: render a few sample frames from the block diffusion GIF
to static PNGs so we can inspect the composition."""

import pathlib
import matplotlib.pyplot as plt

from tools.make_block_diffusion_gif import FIG_H, FIG_W, build_frames, draw_frame

out_dir = pathlib.Path("figures/_preview")
out_dir.mkdir(parents=True, exist_ok=True)

frames = build_frames(seed=7)
# Sample a few interesting frame indices.
picks = [0, 3, 6, 9, 12, 16, 20, 23, len(frames) - 1]

for idx in picks:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_frame(ax, frames[idx])
    path = out_dir / f"frame_{idx:02d}.png"
    fig.savefig(path, dpi=130, facecolor="#ffffff")
    plt.close(fig)
    print(f"wrote {path}")
