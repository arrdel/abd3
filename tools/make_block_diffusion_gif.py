"""Render a GIF that visualizes ABD3 semi-autoregressive block diffusion.

The animation uses a concrete sentence and shows two things at once:

  1. Across blocks — the model walks left → right one block at a time. Past
     blocks are frozen (green), the active block is refining (orange → green),
     future blocks are still masked (gray).
  2. Within a block — all B tokens refine **in parallel** across T inner
     denoising steps. Early steps show noisy guesses; later steps converge on
     the final tokens. A confidence bar beneath each square grows with step.

Output: ``figures/block_diffusion.gif`` (Pillow-encoded, no ffmpeg needed).

Usage:
    python -m tools.make_block_diffusion_gif
    python -m tools.make_block_diffusion_gif --out figures/demo.gif --fps 4
"""

from __future__ import annotations

import argparse
import pathlib
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Content: a real sentence, BPE-style tokens (leading space = GPT-2 convention).
# 12 tokens, 3 blocks of B=4. Feel free to swap.
# ---------------------------------------------------------------------------

FINAL_TOKENS: list[str] = [
    "The",   " quick", " brown",  " fox",      # block 0
    " jumped", " over",  " the",   " lazy",     # block 1
    " dog",   " today", ".",       "<|eos|>",   # block 2
]

BLOCK_SIZE = 4
NUM_BLOCKS = len(FINAL_TOKENS) // BLOCK_SIZE
INNER_STEPS = 6            # T_max refinement steps per block
PAUSE_AFTER_BLOCK = 2      # extra frames to linger after a block finalizes

# Plausible noise vocab for intermediate guesses — single BPE-ish tokens.
NOISE_VOCAB: list[str] = [
    " small", " big", " red", " blue", " old", " slow", " happy", " green",
    " ran", " sat", " slept", " barked", " chased", " ate", " saw", " heard",
    " dog", " cat", " bird", " mouse", " man", " boy", " girl", " cow",
    " park", " road", " yard", " house", " tree", " grass", " stone", " river",
    " very", " quickly", " quietly", " today", " then", " often", " never", " soon",
]

MASK_TOKEN = "[MASK]"


# ---------------------------------------------------------------------------
# Palette — modern, readable, colorblind-friendlyish.
# ---------------------------------------------------------------------------

COLOR_DONE_FILL   = "#10b981"   # emerald-500 — frozen / decoded
COLOR_DONE_EDGE   = "#047857"   # emerald-700
COLOR_DONE_TEXT   = "#ffffff"
COLOR_ACTIVE_BG   = "#fff7ed"   # amber-50 backdrop under active block
COLOR_ACTIVE_RING = "#fb923c"   # amber-400 block outline
COLOR_NOISE_FILL  = "#fed7aa"   # amber-200
COLOR_NOISE_EDGE  = "#f97316"   # amber-500
COLOR_NOISE_TEXT  = "#9a3412"   # amber-900
COLOR_MID_FILL    = "#bbf7d0"   # green-200 (partial confidence, mid-refinement)
COLOR_MID_EDGE    = "#16a34a"   # green-600
COLOR_MID_TEXT    = "#064e3b"
COLOR_FUTURE_FILL = "#f3f4f6"   # gray-100
COLOR_FUTURE_EDGE = "#9ca3af"   # gray-400
COLOR_FUTURE_TEXT = "#6b7280"   # gray-500
COLOR_CONF_BAR    = "#059669"   # emerald-600
COLOR_CONF_TRACK  = "#e5e7eb"
COLOR_BG          = "#ffffff"
COLOR_TITLE       = "#111827"
COLOR_CAPTION     = "#4b5563"

# Layout constants — inches and data coords.
FIG_W, FIG_H = 14.0, 6.6
SQ_W, SQ_H = 0.95, 0.95
SQ_GAP = 0.10
BLOCK_GAP = 0.55
ROW_Y = 2.8           # vertical center of the token row
CONF_Y = ROW_Y - 0.70 # confidence bar baseline
CONF_H = 0.14
BLOCK_LABEL_Y = ROW_Y + 0.95


# ---------------------------------------------------------------------------
# Animation plan
# ---------------------------------------------------------------------------


def build_frames(seed: int = 7) -> list[dict]:
    """Return a list of frame dicts. Each frame describes the full state at
    that moment: which block is active, what tokens every position currently
    shows, per-token confidence, and the running NFE counter."""

    rng = random.Random(seed)
    frames: list[dict] = []

    nfe = 0
    saved_nfe = 0                # cumulative NFE saved by adaptive stopping
    final_state: list[str] = [MASK_TOKEN] * len(FINAL_TOKENS)
    final_conf: list[float] = [0.0] * len(FINAL_TOKENS)

    def token_at_step(target: str, step: int, total: int) -> tuple[str, float]:
        """Simulate refinement: step 0 → MASK, step `total` → target, in
        between pick either the true token (once "locked") or a noise draw.

        Confidence grows linearly with step; once locked to the target, we
        pin confidence to 1.0 so the visual doesn't flicker.
        """
        if step == 0:
            return MASK_TOKEN, 0.0
        if step >= total:
            return target, 1.0
        # Probability of revealing increases roughly quadratically with step.
        reveal_p = (step / total) ** 0.8
        if rng.random() < reveal_p:
            return target, min(1.0, 0.5 + step / total * 0.5)
        return rng.choice(NOISE_VOCAB), step / total * 0.5

    for blk in range(NUM_BLOCKS):
        blk_start = blk * BLOCK_SIZE
        blk_end = blk_start + BLOCK_SIZE

        # Each position in the block "locks" at some inner step within
        # [INNER_STEPS - 3, INNER_STEPS]. Adaptive stopping fires once *all*
        # positions have locked, so the actual number of model calls for this
        # block is `max(lock_step)`, not the full INNER_STEPS.
        # Adaptive stopping will fire *before* INNER_STEPS on every block, so
        # lock steps live in [INNER_STEPS - 4, INNER_STEPS - 1]. That mirrors
        # what we measured in the feasibility efficiency benchmark (4× NFE
        # reduction vs. fixed-T sampling).
        lock_step = [rng.randint(max(1, INNER_STEPS - 4), INNER_STEPS - 1) for _ in range(BLOCK_SIZE)]
        adaptive_stop_step = max(lock_step)

        for step in range(adaptive_stop_step + 1):   # 0 .. adaptive_stop_step inclusive
            nfe_this_step = BLOCK_SIZE if step > 0 else 0
            nfe += nfe_this_step

            # Update positions inside this block.
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

            # Tentative savings-so-far: any block that finishes early adds
            # (INNER_STEPS - adaptive_stop_step) * BLOCK_SIZE to saved_nfe.
            block_saved = (INNER_STEPS - adaptive_stop_step) * BLOCK_SIZE if step == adaptive_stop_step else 0
            frames.append(
                dict(
                    block=blk,
                    step=step,
                    nfe=nfe,
                    saved_nfe=saved_nfe + block_saved,
                    tokens=list(final_state),
                    confs=list(final_conf),
                    adaptive_stop_step=adaptive_stop_step,
                    adaptive_fired=(step == adaptive_stop_step and adaptive_stop_step < INNER_STEPS),
                )
            )

        # Commit the block's savings once it closes out, then add pause frames.
        saved_nfe += (INNER_STEPS - adaptive_stop_step) * BLOCK_SIZE
        for _ in range(PAUSE_AFTER_BLOCK):
            frames.append(
                dict(
                    block=blk,
                    step=adaptive_stop_step,
                    nfe=nfe,
                    saved_nfe=saved_nfe,
                    tokens=list(final_state),
                    confs=list(final_conf),
                    adaptive_stop_step=adaptive_stop_step,
                    adaptive_fired=(adaptive_stop_step < INNER_STEPS),
                    pause_after_block=True,
                )
            )

    # Extra lingering frames on the completed sentence.
    for _ in range(6):
        frames.append(
            dict(
                block=NUM_BLOCKS - 1,
                step=INNER_STEPS,
                nfe=nfe,
                saved_nfe=saved_nfe,
                tokens=list(final_state),
                confs=list(final_conf),
                finished=True,
                adaptive_fired=True,
            )
        )

    return frames


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _token_x(pos: int) -> float:
    """X-coordinate (in inches-ish data units) of the LEFT edge of token `pos`."""
    blk = pos // BLOCK_SIZE
    within = pos % BLOCK_SIZE
    x = 0.6 + blk * (BLOCK_SIZE * (SQ_W + SQ_GAP) + BLOCK_GAP) + within * (SQ_W + SQ_GAP)
    return x


def _block_bounds(blk: int) -> tuple[float, float]:
    left = _token_x(blk * BLOCK_SIZE) - 0.18
    right = _token_x(blk * BLOCK_SIZE + BLOCK_SIZE - 1) + SQ_W + 0.18
    return left, right


def _pretty_token(tok: str) -> str:
    """Strip leading space for cleaner display; show <eos> compactly."""
    if tok == "<|eos|>":
        return "⟨eos⟩"
    if tok == MASK_TOKEN:
        return "▂▂▂▂"
    return tok.replace(" ", "·", 1).lstrip("·") or tok


def _running_text(tokens: list[str], active_block: int, step: int, inner_steps: int) -> str:
    """Join fully-decoded tokens into a running sentence at the top of the frame.

    Only completed blocks contribute; partially-decoded current block shows
    as a grey placeholder ellipsis so the reader sees "where we are" without
    flickering noise words in the headline.
    """
    shown = []
    for blk in range(NUM_BLOCKS):
        start = blk * BLOCK_SIZE
        end = start + BLOCK_SIZE
        if blk < active_block or (blk == active_block and step >= inner_steps):
            shown.append("".join(tokens[start:end]).replace("<|eos|>", ""))
        elif blk == active_block:
            shown.append(" …")
        # Future blocks contribute nothing to the running sentence.
    joined = "".join(shown).strip()
    return joined or "…"


def draw_frame(ax, frame: dict):
    """Paint one frame onto the axis (clears axis first)."""
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

    # ----- Title -----
    ax.text(
        FIG_W / 2,
        FIG_H - 0.35,
        "ABD3 — Semi-Autoregressive Block Diffusion",
        ha="center", va="center",
        fontsize=19, fontweight="bold", color=COLOR_TITLE,
    )
    subtitle = ("Blocks run left → right; tokens within a block refine in parallel across T inner steps.")
    ax.text(
        FIG_W / 2,
        FIG_H - 0.80,
        subtitle,
        ha="center", va="center",
        fontsize=11, style="italic", color=COLOR_CAPTION,
    )

    # ----- Running sentence panel -----
    panel_y = FIG_H - 1.85
    panel_h = 0.75
    panel = patches.FancyBboxPatch(
        (0.6, panel_y),
        FIG_W - 1.2,
        panel_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.1,
        edgecolor="#d1d5db",
        facecolor="#f9fafb",
    )
    ax.add_patch(panel)
    ax.text(
        0.85, panel_y + panel_h / 2 + 0.05,
        "Decoded so far:",
        ha="left", va="center",
        fontsize=10, fontweight="bold", color=COLOR_CAPTION,
    )
    running = _running_text(tokens, active_block, step, INNER_STEPS)
    ax.text(
        2.55, panel_y + panel_h / 2,
        running,
        ha="left", va="center",
        fontsize=15, fontweight="semibold",
        color=COLOR_DONE_EDGE if finished else COLOR_TITLE,
        family="serif",
    )

    # ----- Block markers (background bands for active block) -----
    for blk in range(NUM_BLOCKS):
        left, right = _block_bounds(blk)
        if blk == active_block and not finished:
            band = patches.FancyBboxPatch(
                (left, ROW_Y - 0.95),
                right - left, 2.10,
                boxstyle="round,pad=0.0,rounding_size=0.15",
                linewidth=2.2,
                edgecolor=COLOR_ACTIVE_RING,
                facecolor=COLOR_ACTIVE_BG,
                linestyle="--",
                alpha=0.95,
            )
            ax.add_patch(band)

        label = f"Block {blk}"
        if blk < active_block or finished:
            label += " ✓"
        elif blk == active_block:
            label += f"  · step {step}/{INNER_STEPS}"
        ax.text(
            (left + right) / 2,
            BLOCK_LABEL_Y,
            label,
            ha="center", va="center",
            fontsize=11,
            fontweight="bold",
            color=(
                COLOR_DONE_EDGE if blk < active_block or finished
                else COLOR_ACTIVE_RING if blk == active_block
                else COLOR_FUTURE_EDGE
            ),
        )

        # Adaptive-stop firing caption — sits just above the block label.
        if blk == active_block and frame.get("adaptive_fired"):
            saved_steps = INNER_STEPS - step
            if saved_steps > 0:
                ax.text(
                    (left + right) / 2,
                    BLOCK_LABEL_Y + 0.40,
                    f"⚡ adaptive stop @ step {step}  ·  saved {saved_steps * BLOCK_SIZE} NFE",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=COLOR_DONE_EDGE,
                )

    # ----- Token squares + confidence bars -----
    for pos, tok in enumerate(tokens):
        x = _token_x(pos)
        blk = pos // BLOCK_SIZE
        conf = confs[pos]

        # Decide the colour scheme for this square.
        if blk < active_block or finished:
            fill, edge, txt_color = COLOR_DONE_FILL, COLOR_DONE_EDGE, COLOR_DONE_TEXT
            display = _pretty_token(FINAL_TOKENS[pos])
        elif blk > active_block:
            fill, edge, txt_color = COLOR_FUTURE_FILL, COLOR_FUTURE_EDGE, COLOR_FUTURE_TEXT
            display = "▂▂▂▂"
        else:
            # Active block
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

        rect = patches.FancyBboxPatch(
            (x, ROW_Y - SQ_H / 2),
            SQ_W, SQ_H,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(rect)

        # Token text. Shrink for long noise words.
        fontsize = 11 if len(display) <= 6 else (10 if len(display) <= 9 else 8)
        ax.text(
            x + SQ_W / 2,
            ROW_Y,
            display,
            ha="center", va="center",
            fontsize=fontsize,
            color=txt_color,
            fontweight="bold" if conf >= 1.0 else "semibold",
            family="monospace",
        )

        # Confidence track (drawn for every in-progress and completed token).
        track = patches.Rectangle(
            (x + 0.05, CONF_Y),
            SQ_W - 0.10, CONF_H,
            linewidth=0.5, edgecolor="none",
            facecolor=COLOR_CONF_TRACK,
        )
        ax.add_patch(track)
        conf_display = 1.0 if (blk < active_block or finished) else conf
        if conf_display > 0:
            bar = patches.Rectangle(
                (x + 0.05, CONF_Y),
                (SQ_W - 0.10) * conf_display, CONF_H,
                linewidth=0.0,
                facecolor=COLOR_CONF_BAR if conf_display >= 1.0 else COLOR_MID_EDGE,
            )
            ax.add_patch(bar)

    # ----- Status bar at bottom -----
    status_y = 0.55
    status_box = patches.FancyBboxPatch(
        (0.6, status_y - 0.25),
        FIG_W - 1.2, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0,
        edgecolor="#d1d5db",
        facecolor="#f9fafb",
    )
    ax.add_patch(status_box)

    theoretical_nfe = NUM_BLOCKS * INNER_STEPS * BLOCK_SIZE   # fixed-T reference (no adaptive stop)
    saved_nfe = frame.get("saved_nfe", 0)
    saved_pct = 100.0 * saved_nfe / max(theoretical_nfe, 1)
    parts = [
        f"Block {min(active_block, NUM_BLOCKS-1)+1}/{NUM_BLOCKS}",
        f"step {step}/{INNER_STEPS}",
        f"NFE used {nfe}",
        f"saved {saved_nfe} ({saved_pct:4.1f}% vs fixed-T)",
    ]
    if finished:
        parts = [
            f"Blocks {NUM_BLOCKS}/{NUM_BLOCKS} ✓",
            f"NFE {nfe} / fixed-T {theoretical_nfe}",
            f"saved {saved_nfe} ({saved_pct:4.1f}%)",
            "Done",
        ]
    status_text = "   •   ".join(parts)
    ax.text(
        FIG_W / 2,
        status_y,
        status_text,
        ha="center", va="center",
        fontsize=11.5,
        color=COLOR_TITLE,
        family="monospace",
    )

    # ----- Legend chips bottom-left -----
    legend_items = [
        (COLOR_DONE_FILL, COLOR_DONE_EDGE, "decoded"),
        (COLOR_MID_FILL,  COLOR_MID_EDGE,  "refining"),
        (COLOR_NOISE_FILL, COLOR_NOISE_EDGE, "noisy draft"),
        (COLOR_FUTURE_FILL, COLOR_FUTURE_EDGE, "masked / future"),
    ]
    lx = 0.60
    ly = 1.40
    for fill, edge, label in legend_items:
        chip = patches.FancyBboxPatch(
            (lx, ly), 0.26, 0.26,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            linewidth=1.0, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(chip)
        ax.text(lx + 0.34, ly + 0.13, label, ha="left", va="center",
                fontsize=9, color=COLOR_CAPTION)
        lx += 1.55


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
    print(f"[gif] wrote {out_path}  ({len(frames)} frames @ {fps} fps)")


def main() -> int:
    p = argparse.ArgumentParser(description="Render the ABD3 block-diffusion GIF.")
    p.add_argument("--out", default="figures/block_diffusion.gif", type=pathlib.Path)
    p.add_argument("--fps", type=int, default=3, help="Frames per second.")
    p.add_argument("--seed", type=int, default=7, help="RNG seed for noise draws.")
    args = p.parse_args()
    render_gif(args.out, args.fps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
