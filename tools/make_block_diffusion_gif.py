"""
Render a GIF of ABD3 block diffusion with blocks arranged in a 3×3 grid.
No overlaps, modern fonts, and the full sentence stays visible.
"""

from __future__ import annotations

import argparse
import pathlib
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Content: 32 tokens → 8 blocks of size 4. Arrange in a grid (3 cols, 3 rows).
# ---------------------------------------------------------------------------
FINAL_TOKENS: list[str] = [
    " He", " and", " the", " others", " stayed", " with", " the", " injured",
    " all", " night", ",", " after", " the", " medical", " team", " had",
    " left", " and", " after", " the", " generators", " gave", " out", " and",
    " the", " tents", " turned", " pitch", " black", ".", "<|eos|>",
]

BLOCK_SIZE = 4
NUM_BLOCKS = len(FINAL_TOKENS) // BLOCK_SIZE   # = 8
GRID_COLS = 3
GRID_ROWS = (NUM_BLOCKS + GRID_COLS - 1) // GRID_COLS   # = 3

INNER_STEPS = 6
PAUSE_AFTER_BLOCK = 2

NOISE_VOCAB: list[str] = [
    " small", " big", " red", " blue", " old", " slow", " happy", " green",
    " ran", " sat", " slept", " barked", " chased", " ate", " saw", " heard",
    " dog", " cat", " bird", " mouse", " man", " boy", " girl", " cow",
    " park", " road", " yard", " house", " tree", " grass", " stone", " river",
    " very", " quickly", " quietly", " today", " then", " often", " never", " soon",
]

MASK_TOKEN = "[MASK]"

# ---------------------------------------------------------------------------
# Modern, clean colour palette
# ---------------------------------------------------------------------------
COLOR_BG            = "#fefcf5"
COLOR_DONE_FILL     = "#2b9348"
COLOR_DONE_EDGE     = "#1f6e3a"
COLOR_DONE_TEXT     = "#ffffff"
COLOR_ACTIVE_BG     = "#fff8e7"
COLOR_ACTIVE_RING   = "#ff9f1c"
COLOR_NOISE_FILL    = "#ffe8c7"
COLOR_NOISE_EDGE    = "#e88d1d"
COLOR_NOISE_TEXT    = "#8b5a2b"
COLOR_MID_FILL      = "#b7efc5"
COLOR_MID_EDGE      = "#228848"
COLOR_MID_TEXT      = "#1a4422"
COLOR_FUTURE_FILL   = "#edf2f7"
COLOR_FUTURE_EDGE   = "#cbd5e0"
COLOR_FUTURE_TEXT   = "#a0aec0"
COLOR_CONF_BAR      = "#2b9348"
COLOR_CONF_TRACK    = "#e2e8f0"
COLOR_TITLE         = "#1e293b"
COLOR_CAPTION       = "#475569"

# ---------------------------------------------------------------------------
# Layout – dimensions in inches
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 14.0, 9.5        # taller to accommodate grid

# Token square size and gaps (within a block)
SQ_W, SQ_H = 0.70, 0.70
SQ_GAP = 0.08

# Spacing between blocks in the grid
BLOCK_H_GAP = 0.45              # horizontal gap between blocks (columns)
BLOCK_V_GAP = 0.75              # vertical gap between rows of blocks

# Starting position for the first block (top‑left corner of its token row area)
START_X = 1.2
START_Y = 5.8                   # y‑coordinate of the first block's token row centre

# Confidence bar (relative to each block's token row)
CONF_H = 0.12
CONF_Y_OFFSET = -0.65           # below token row centre

# Block label (above token row)
LABEL_Y_OFFSET = +0.85

# Adaptive text offset (above label)
ADAPT_OFFSET = +0.40

# ---------------------------------------------------------------------------
# Helper functions for grid coordinates
# ---------------------------------------------------------------------------
def _block_grid_pos(blk: int) -> tuple[int, int]:
    """Return (row, col) for the given block index."""
    return (blk // GRID_COLS, blk % GRID_COLS)

def _block_origin(blk: int) -> tuple[float, float]:
    """Return (x0, y_centre) where the block's token row starts (left edge)."""
    row, col = _block_grid_pos(blk)
    x0 = START_X + col * (BLOCK_SIZE * (SQ_W + SQ_GAP) + BLOCK_H_GAP)
    y_centre = START_Y - row * (SQ_H + BLOCK_V_GAP + CONF_H + 0.2)
    return x0, y_centre

def _token_x(blk: int, within: int) -> float:
    """X coordinate of the LEFT edge of a token inside its block."""
    x0, _ = _block_origin(blk)
    return x0 + within * (SQ_W + SQ_GAP)

def _block_bounds(blk: int) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) of the whole block area (for background)."""
    x0, yc = _block_origin(blk)
    left = x0 - 0.15
    right = x0 + BLOCK_SIZE * (SQ_W + SQ_GAP) - SQ_GAP + 0.15
    bottom = yc - SQ_H/2 - 0.4
    top = yc + SQ_H/2 + 1.0
    return left, right, bottom, top

def _pretty_token(tok: str) -> str:
    if tok == "<|eos|>":
        return "⟨eos⟩"
    if tok == MASK_TOKEN:
        return "▂▂▂▂"
    return tok.replace(" ", "·", 1).lstrip("·") or tok

def _running_text(tokens: list[str], active_block: int, step: int) -> str:
    """Running sentence: only blocks that are fully decoded."""
    shown = []
    for blk in range(NUM_BLOCKS):
        start = blk * BLOCK_SIZE
        end = start + BLOCK_SIZE
        if blk < active_block or (blk == active_block and step >= INNER_STEPS):
            shown.append("".join(tokens[start:end]).replace("<|eos|>", ""))
        elif blk == active_block:
            shown.append(" …")
    joined = "".join(shown).strip()
    return joined or "…"

# ---------------------------------------------------------------------------
# Frame simulation (unchanged logic, works with any layout)
# ---------------------------------------------------------------------------
def build_frames(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    frames = []
    nfe = 0
    saved_nfe = 0
    final_state = [MASK_TOKEN] * len(FINAL_TOKENS)
    final_conf = [0.0] * len(FINAL_TOKENS)

    def token_at_step(target: str, step: int, total: int) -> tuple[str, float]:
        if step == 0:
            return MASK_TOKEN, 0.0
        if step >= total:
            return target, 1.0
        reveal_p = (step / total) ** 0.8
        if rng.random() < reveal_p:
            return target, min(1.0, 0.5 + step / total * 0.5)
        return rng.choice(NOISE_VOCAB), step / total * 0.5

    for blk in range(NUM_BLOCKS):
        blk_start = blk * BLOCK_SIZE
        blk_end = blk_start + BLOCK_SIZE
        lock_step = [rng.randint(max(1, INNER_STEPS - 4), INNER_STEPS - 1) for _ in range(BLOCK_SIZE)]
        adaptive_stop_step = max(lock_step)

        for step in range(adaptive_stop_step + 1):
            nfe_this_step = BLOCK_SIZE if step > 0 else 0
            nfe += nfe_this_step
            for i, pos in enumerate(range(blk_start, blk_end)):
                if step == 0:
                    final_state[pos] = MASK_TOKEN
                    final_conf[pos] = 0.0
                elif step >= lock_step[i]:
                    final_state[pos] = FINAL_TOKENS[pos]
                    final_conf[pos] = 1.0
                else:
                    tok, conf = token_at_step(FINAL_TOKENS[pos], step, lock_step[i])
                    final_state[pos] = tok
                    final_conf[pos] = conf
            block_saved = (INNER_STEPS - adaptive_stop_step) * BLOCK_SIZE if step == adaptive_stop_step else 0
            frames.append({
                "block": blk,
                "step": step,
                "nfe": nfe,
                "saved_nfe": saved_nfe + block_saved,
                "tokens": list(final_state),
                "confs": list(final_conf),
                "adaptive_stop_step": adaptive_stop_step,
                "adaptive_fired": step == adaptive_stop_step and adaptive_stop_step < INNER_STEPS,
            })
        saved_nfe += (INNER_STEPS - adaptive_stop_step) * BLOCK_SIZE
        for _ in range(PAUSE_AFTER_BLOCK):
            frames.append({
                "block": blk,
                "step": adaptive_stop_step,
                "nfe": nfe,
                "saved_nfe": saved_nfe,
                "tokens": list(final_state),
                "confs": list(final_conf),
                "adaptive_stop_step": adaptive_stop_step,
                "adaptive_fired": adaptive_stop_step < INNER_STEPS,
                "pause_after_block": True,
            })
    for _ in range(6):
        frames.append({
            "block": NUM_BLOCKS - 1,
            "step": INNER_STEPS,
            "nfe": nfe,
            "saved_nfe": saved_nfe,
            "tokens": list(final_state),
            "confs": list(final_conf),
            "finished": True,
            "adaptive_fired": True,
        })
    return frames

# ---------------------------------------------------------------------------
# Drawing routine – now uses grid layout
# ---------------------------------------------------------------------------
def draw_frame(ax, frame: dict) -> None:
    ax.clear()
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLOR_BG)

    active_block = frame["block"]
    step = frame["step"]
    nfe = frame["nfe"]
    tokens = frame["tokens"]
    confs = frame["confs"]
    finished = frame.get("finished", False)

    # ----- Title and subtitle (top) -----
    ax.text(FIG_W / 2, FIG_H - 0.45, "ABD3 — Semi-Autoregressive Block Diffusion",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=COLOR_TITLE, family="sans-serif")
    ax.text(FIG_W / 2, FIG_H - 0.95,
            "Blocks run left→right, top→bottom · tokens within a block refine in parallel",
            ha="center", va="center", fontsize=11, style="italic", color=COLOR_CAPTION)

    # ----- Running sentence panel (still at top, now narrower to avoid overflow) -----
    panel_y = FIG_H - 1.85
    panel_h = 0.75
    panel = patches.FancyBboxPatch((0.6, panel_y), FIG_W - 1.2, panel_h,
                                   boxstyle="round,pad=0.02,rounding_size=0.12",
                                   linewidth=1.0, edgecolor="#cbd5e0", facecolor="#ffffff")
    ax.add_patch(panel)
    ax.text(0.85, panel_y + panel_h / 2 + 0.05, "Decoded so far:",
            ha="left", va="center", fontsize=10, fontweight="bold", color=COLOR_CAPTION)
    running = _running_text(tokens, active_block, step)
    # Dynamic font size for long sentences
    fs = 15 if len(running) < 60 else (13 if len(running) < 100 else 11)
    ax.text(2.5, panel_y + panel_h / 2, running,
            ha="left", va="center", fontsize=fs, fontweight="semibold",
            color=COLOR_DONE_EDGE if finished else COLOR_TITLE, family="serif",
            wrap=True)

    # ----- Draw each block -----
    for blk in range(NUM_BLOCKS):
        x0, yc = _block_origin(blk)
        left, right, bottom, top = _block_bounds(blk)
        blk_start = blk * BLOCK_SIZE

        # Active block background
        if blk == active_block and not finished:
            band = patches.FancyBboxPatch((left, bottom), right - left, top - bottom,
                                          boxstyle="round,pad=0.0,rounding_size=0.15",
                                          linewidth=2.2, edgecolor=COLOR_ACTIVE_RING,
                                          facecolor=COLOR_ACTIVE_BG, linestyle="--",
                                          alpha=0.9)
            ax.add_patch(band)

        # Block label
        label = f"Block {blk}"
        if blk < active_block or finished:
            label += " ✓"
        elif blk == active_block:
            label += f"  ·  step {step}/{INNER_STEPS}"
        ax.text((left + right) / 2, yc + LABEL_Y_OFFSET, label,
                ha="center", va="center", fontsize=11,
                fontweight="bold",
                color=(COLOR_DONE_EDGE if blk < active_block or finished
                       else COLOR_ACTIVE_RING if blk == active_block
                       else COLOR_FUTURE_EDGE))

        # Adaptive stop indicator
        if blk == active_block and frame.get("adaptive_fired") and not finished:
            saved_steps = INNER_STEPS - step
            if saved_steps > 0:
                ax.text((left + right) / 2, yc + LABEL_Y_OFFSET + ADAPT_OFFSET,
                        f"⚡ adaptive stop @ {step}  ·  saved {saved_steps * BLOCK_SIZE} NFE",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color=COLOR_DONE_EDGE, family="monospace")

        # Parallel refine hint (only while refining)
        if blk == active_block and not finished and 0 < step < INNER_STEPS:
            ax.text((left + right) / 2, yc - SQ_H/2 - 0.55, "↺ parallel refine ↺",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=COLOR_ACTIVE_RING, alpha=0.8)

        # Draw tokens and confidence bars for this block
        for i in range(BLOCK_SIZE):
            pos = blk_start + i
            tok = tokens[pos]
            conf = confs[pos]
            x = _token_x(blk, i)

            # Determine style
            if blk < active_block or finished:
                fill, edge, txt_color = COLOR_DONE_FILL, COLOR_DONE_EDGE, COLOR_DONE_TEXT
                display = _pretty_token(FINAL_TOKENS[pos])
            elif blk > active_block:
                fill, edge, txt_color = COLOR_FUTURE_FILL, COLOR_FUTURE_EDGE, COLOR_FUTURE_TEXT
                display = "▂▂▂▂"
            else:  # active block
                if tok == MASK_TOKEN:
                    fill, edge, txt_color = COLOR_FUTURE_FILL, COLOR_FUTURE_EDGE, COLOR_FUTURE_TEXT
                    display = "▂▂▂▂"
                elif conf >= 1.0:
                    fill, edge, txt_color = COLOR_DONE_FILL, COLOR_DONE_EDGE, COLOR_DONE_TEXT
                    display = _pretty_token(tok)
                elif conf >= 0.6:
                    fill, edge, txt_color = COLOR_MID_FILL, COLOR_MID_EDGE, COLOR_MID_TEXT
                    display = _pretty_token(tok)
                else:
                    fill, edge, txt_color = COLOR_NOISE_FILL, COLOR_NOISE_EDGE, COLOR_NOISE_TEXT
                    display = _pretty_token(tok)

            rect = patches.FancyBboxPatch((x, yc - SQ_H/2), SQ_W, SQ_H,
                                          boxstyle="round,pad=0.0,rounding_size=0.10",
                                          linewidth=1.5, edgecolor=edge, facecolor=fill)
            ax.add_patch(rect)

            fontsize = 10 if len(display) <= 6 else (9 if len(display) <= 9 else 7)
            ax.text(x + SQ_W/2, yc, display,
                    ha="center", va="center", fontsize=fontsize,
                    color=txt_color, fontweight="bold" if conf >= 1.0 else "semibold",
                    family="monospace")

            # Confidence bar
            bar_y = yc + CONF_Y_OFFSET
            bar_width = SQ_W - 0.08
            track = patches.Rectangle((x + 0.04, bar_y), bar_width, CONF_H,
                                      linewidth=0, facecolor=COLOR_CONF_TRACK)
            ax.add_patch(track)
            conf_display = 1.0 if (blk < active_block or finished) else conf
            if conf_display > 0:
                bar = patches.Rectangle((x + 0.04, bar_y), bar_width * conf_display, CONF_H,
                                        linewidth=0, facecolor=COLOR_CONF_BAR)
                ax.add_patch(bar)

    # ----- Status bar (bottom) -----
    sb = patches.FancyBboxPatch((0.6, 0.45), FIG_W - 1.2, 0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.0, edgecolor="#cbd5e0", facecolor="#ffffff")
    ax.add_patch(sb)

    theoretical_nfe = NUM_BLOCKS * INNER_STEPS * BLOCK_SIZE
    saved_nfe = frame.get("saved_nfe", 0)
    saved_pct = 100.0 * saved_nfe / max(theoretical_nfe, 1)
    if finished:
        status_text = f"✓ Complete  |  NFE {nfe} / fixed‑T {theoretical_nfe}   •   ⚡ saved {saved_nfe} ({saved_pct:.1f}%)"
    else:
        status_text = (f"Block {min(active_block, NUM_BLOCKS-1)+1}/{NUM_BLOCKS}   •   "
                       f"step {step}/{INNER_STEPS}   •   NFE {nfe}   •   "
                       f"saved {saved_nfe} ({saved_pct:.1f}% vs fixed‑T)")
    ax.text(FIG_W / 2, 0.68, status_text,
            ha="center", va="center", fontsize=11,
            color=COLOR_TITLE, family="monospace")

    # ----- Legend (bottom left) -----
    legend_items = [
        (COLOR_DONE_FILL, "decoded"),
        (COLOR_MID_FILL, "refining"),
        (COLOR_NOISE_FILL, "noisy draft"),
        (COLOR_FUTURE_FILL, "masked / future"),
    ]
    lx = 0.65
    leg_y = 1.25
    for fill, label in legend_items:
        chip = patches.FancyBboxPatch((lx, leg_y), 0.24, 0.24,
                                      boxstyle="round,pad=0.0,rounding_size=0.06",
                                      linewidth=1.0, edgecolor="#a0aec0", facecolor=fill)
        ax.add_patch(chip)
        ax.text(lx + 0.30, leg_y + 0.12, label,
                ha="left", va="center", fontsize=9, color=COLOR_CAPTION)
        lx += 1.45


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------
def render_gif(out_path: pathlib.Path, fps: int, seed: int) -> None:
    frames = build_frames(seed=seed)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=130)
    fig.patch.set_facecolor(COLOR_BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def _update(i):
        draw_frame(ax, frames[i])
        return []

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000 // fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[gif] saved {out_path}  ({len(frames)} frames @ {fps} fps)")


def main() -> int:
    p = argparse.ArgumentParser(description="ABD3 block diffusion GIF – grid layout, modern fonts")
    p.add_argument("--out", default="figures/block_diffusion.gif", type=pathlib.Path)
    p.add_argument("--fps", type=int, default=1, help="Frames per second (default 5)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    args = p.parse_args()
    render_gif(args.out, args.fps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())