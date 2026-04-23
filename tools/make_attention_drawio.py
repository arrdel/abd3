"""Generate a multi-page draw.io file illustrating ABD3 attention.

Produces ``figures/abd3_attention.drawio`` with three pages:

  1. Baseline (BD3-LMs) self-attention on a concatenated 2N sequence —
     every noisy token re-attends over all 2N positions at every step.

  2. ABD3 two-stream attention — a dedicated clean context stream (K, V only)
     plus a noisy generation stream that self-attends within the current block
     and cross-attends into the clean stream.

  3. KV caching in ABD3 — completed blocks contribute K,V to a growing cache
     exactly once; future blocks cross-attend into the cache at near-zero cost.

Words are drawn as coloured squares. Q, K, V extraction and attention flow are
drawn with labelled arrows. Import the file directly at https://app.diagrams.net.

Usage:
    python -m tools.make_attention_drawio
    python -m tools.make_attention_drawio --out figures/abd3_attention.drawio
"""

from __future__ import annotations

import argparse
import pathlib
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Palette + shared style presets
# ---------------------------------------------------------------------------

# Tokens / words
CLEAN_FILL  = "#dbeafe"   # blue-100 — clean past / context token
CLEAN_STROKE = "#2563eb"  # blue-600
CLEAN_TEXT  = "#1e3a8a"   # blue-900

NOISY_FILL  = "#fed7aa"   # amber-200 — noisy / current-block token
NOISY_STROKE = "#ea580c"  # orange-600
NOISY_TEXT  = "#7c2d12"   # orange-900

DONE_FILL   = "#bbf7d0"   # green-200 — already-decoded block
DONE_STROKE = "#15803d"   # green-700
DONE_TEXT   = "#14532d"

FUTURE_FILL = "#f3f4f6"   # gray-100 — future block (not yet started)
FUTURE_STROKE = "#9ca3af"
FUTURE_TEXT = "#4b5563"

CACHE_FILL  = "#ede9fe"   # violet-100 — KV cache
CACHE_STROKE = "#7c3aed"  # violet-600
CACHE_TEXT  = "#4c1d95"

Q_COLOR = "#dc2626"   # red-600
K_COLOR = "#0891b2"   # cyan-600
V_COLOR = "#7c3aed"   # violet-600

# Arrow styles (strings concatenated into draw.io cell style).
ARROW_Q  = f"endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;strokeColor={Q_COLOR};strokeWidth=2;fontColor={Q_COLOR};fontSize=11;fontStyle=1;"
ARROW_K  = f"endArrow=classic;html=1;rounded=0;strokeColor={K_COLOR};strokeWidth=2;fontColor={K_COLOR};fontSize=11;fontStyle=1;"
ARROW_V  = f"endArrow=classic;html=1;rounded=0;strokeColor={V_COLOR};strokeWidth=2;fontColor={V_COLOR};fontSize=11;fontStyle=1;"
ARROW_ATTN = f"endArrow=classic;html=1;rounded=0;strokeColor={Q_COLOR};strokeWidth=1.6;dashed=1;fontColor={Q_COLOR};fontSize=10;"
ARROW_CROSS = f"endArrow=classic;html=1;rounded=0;strokeColor=#0e7490;strokeWidth=2.2;fontColor=#0e7490;fontSize=11;fontStyle=1;"


# ---------------------------------------------------------------------------
# XML builder helpers
# ---------------------------------------------------------------------------


class Diagram:
    """Builds one `<diagram>` page of a draw.io file."""

    def __init__(self, name: str, page_w: int = 1400, page_h: int = 900) -> None:
        self.name = name
        self.page_w = page_w
        self.page_h = page_h
        self._next_id = 2  # 0 and 1 are reserved for root + parent.
        self.cells: list[ET.Element] = []

    def _new_id(self) -> str:
        i = self._next_id
        self._next_id += 1
        return f"n{i}"

    def add_node(self, x: float, y: float, w: float, h: float,
                 value: str = "", style: str = "rounded=1;whiteSpace=wrap;html=1;",
                 node_id: str | None = None) -> str:
        nid = node_id or self._new_id()
        cell = ET.Element("mxCell", {
            "id": nid,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": "1",
        })
        geom = ET.SubElement(cell, "mxGeometry", {
            "x": f"{x}",
            "y": f"{y}",
            "width": f"{w}",
            "height": f"{h}",
            "as": "geometry",
        })
        self.cells.append(cell)
        return nid

    def add_text(self, x: float, y: float, w: float, h: float, value: str,
                 style: str = "text;html=1;align=center;verticalAlign=middle;fontSize=12;") -> str:
        return self.add_node(x, y, w, h, value, style=style)

    def add_edge(self, source: str, target: str, style: str = ARROW_Q,
                 label: str = "", waypoints: list[tuple[float, float]] | None = None) -> str:
        nid = self._new_id()
        cell = ET.Element("mxCell", {
            "id": nid,
            "value": label,
            "style": style,
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        })
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if waypoints:
            array = ET.SubElement(geom, "Array", {"as": "points"})
            for wx, wy in waypoints:
                ET.SubElement(array, "mxPoint", {"x": f"{wx}", "y": f"{wy}"})
        self.cells.append(cell)
        return nid

    def to_xml(self) -> ET.Element:
        diagram = ET.Element("diagram", {"name": self.name, "id": self.name.replace(" ", "_")})
        model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1422", "dy": "757",
            "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1",
            "arrows": "1", "fold": "1",
            "page": "1", "pageScale": "1",
            "pageWidth": f"{self.page_w}",
            "pageHeight": f"{self.page_h}",
            "math": "0", "shadow": "0",
        })
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        for cell in self.cells:
            root.append(cell)
        return diagram


# ---------------------------------------------------------------------------
# Reusable shape builders
# ---------------------------------------------------------------------------


def token_style(fill: str, stroke: str, text: str) -> str:
    return (
        "rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fontColor={text};"
        "fontSize=15;fontStyle=1;strokeWidth=2;shadow=1;"
    )


def small_box_style(fill: str, stroke: str, text: str = "#ffffff") -> str:
    return (
        "rounded=1;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fontColor={text};"
        "fontSize=11;fontStyle=1;strokeWidth=1.4;"
    )


def title(d: Diagram, text: str, sub: str | None = None, y: int = 20) -> None:
    d.add_text(40, y, d.page_w - 80, 40, text,
               style="text;html=1;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#111827;")
    if sub:
        d.add_text(40, y + 40, d.page_w - 80, 30, sub,
                   style="text;html=1;align=center;verticalAlign=middle;fontSize=13;fontStyle=2;fontColor=#4b5563;")


def section_label(d: Diagram, x: float, y: float, w: float, text: str, color: str = "#374151") -> None:
    d.add_text(x, y, w, 24, text,
               style=f"text;html=1;align=center;verticalAlign=middle;fontSize=14;fontStyle=1;fontColor={color};")


def draw_qkv_triplet(d: Diagram, cx: float, top_y: float, colors=(Q_COLOR, K_COLOR, V_COLOR)) -> tuple[str, str, str]:
    """Draw three tiny pills labelled Q K V stacked under a token. Return their ids."""
    labels = ("Q", "K", "V")
    ids = []
    box_w, box_h, gap = 30, 24, 6
    total_w = 3 * box_w + 2 * gap
    x0 = cx - total_w / 2
    for i, (label, col) in enumerate(zip(labels, colors)):
        x = x0 + i * (box_w + gap)
        style = small_box_style(fill=col, stroke=col)
        ids.append(d.add_node(x, top_y, box_w, box_h, label, style=style))
    return tuple(ids)  # type: ignore[return-value]


def draw_kv_pair(d: Diagram, cx: float, top_y: float) -> tuple[str, str]:
    """Draw only K and V pills under a clean-stream token (no Q). Return ids."""
    labels = ("K", "V")
    colors = (K_COLOR, V_COLOR)
    ids = []
    box_w, box_h, gap = 30, 24, 6
    total_w = 2 * box_w + gap
    x0 = cx - total_w / 2
    for i, (label, col) in enumerate(zip(labels, colors)):
        x = x0 + i * (box_w + gap)
        style = small_box_style(fill=col, stroke=col)
        ids.append(d.add_node(x, top_y, box_w, box_h, label, style=style))
    return tuple(ids)  # type: ignore[return-value]


def draw_attention_matrix(
    d: Diagram,
    x: float,
    y: float,
    pattern: list[list[bool]],
    cell: int = 26,
    labels_rows: list[str] | None = None,
    labels_cols: list[str] | None = None,
    row_is_clean: list[bool] | None = None,
    col_is_clean: list[bool] | None = None,
    fill_on: str = "#fca5a5",
    stroke_on: str = "#b91c1c",
    fill_off: str = "#f3f4f6",
    stroke_off: str = "#d1d5db",
    title: str = "",
    legend_line: str = "",
) -> tuple[float, float]:
    """Render a Q×K attention matrix as a grid of small rounded squares. A cell
    is 'on' (filled red) if attention is actually computed there, 'off'
    (gray) otherwise. Returns the (width, height) of the drawn matrix."""
    n_rows = len(pattern)
    n_cols = len(pattern[0]) if n_rows else 0

    grid_w = n_cols * cell
    grid_h = n_rows * cell
    pad = 28

    # Title + legend above the grid.
    if title:
        d.add_text(x, y - 50, grid_w + 80, 24, title,
                   style="text;html=1;align=center;verticalAlign=middle;fontSize=13;fontStyle=1;fontColor=#111827;")
    if legend_line:
        d.add_text(x, y - 28, grid_w + 80, 20, legend_line,
                   style="text;html=1;align=center;verticalAlign=middle;fontSize=10;fontStyle=2;fontColor=#6b7280;")

    # Column labels (token indices or names) above the grid.
    if labels_cols:
        for j, lab in enumerate(labels_cols):
            col = col_is_clean[j] if col_is_clean is not None else True
            fc = CLEAN_STROKE if col else NOISY_STROKE
            d.add_text(x + 32 + j * cell, y - 4, cell, 22, lab,
                       style=f"text;html=1;align=center;verticalAlign=middle;fontSize=9;fontColor={fc};")
    # Row labels to the left.
    if labels_rows:
        for i, lab in enumerate(labels_rows):
            row = row_is_clean[i] if row_is_clean is not None else True
            fc = CLEAN_STROKE if row else NOISY_STROKE
            d.add_text(x, y + 20 + i * cell, 30, cell - 4, lab,
                       style=f"text;html=1;align=right;verticalAlign=middle;fontSize=9;fontColor={fc};")

    # The grid itself (shifted right by 32 px to leave room for row labels).
    grid_x = x + 32
    grid_y = y + 20
    for i in range(n_rows):
        for j in range(n_cols):
            on = pattern[i][j]
            fc, sc = (fill_on, stroke_on) if on else (fill_off, stroke_off)
            d.add_node(
                grid_x + j * cell, grid_y + i * cell, cell - 2, cell - 2,
                "",
                style=(
                    "rounded=1;whiteSpace=wrap;html=1;"
                    f"fillColor={fc};strokeColor={sc};"
                    "strokeWidth=0.8;"
                ),
            )

    # Axis captions.
    d.add_text(grid_x - 30, grid_y + grid_h + 4, grid_w + 60, 20,
               "Key index →",
               style="text;html=1;align=center;verticalAlign=middle;fontSize=10;fontStyle=2;fontColor=#6b7280;")
    d.add_text(grid_x - 70, grid_y + grid_h / 2 - 50, 20, 100,
               "Query",
               style="text;html=1;align=center;verticalAlign=middle;fontSize=10;fontStyle=2;fontColor=#6b7280;rotation=-90;")

    return grid_w + 64, grid_h + 80


def legend_chip(d: Diagram, x: float, y: float, color_fill: str, color_stroke: str,
                label: str, label_width: float = 150) -> float:
    """Draw a legend chip + label pair. Returns the right edge in pixels so the
    caller can chain the next chip without manual spacing math."""
    chip_w = 22
    d.add_node(x, y, chip_w, 22, "",
               style=f"rounded=1;whiteSpace=wrap;html=1;fillColor={color_fill};strokeColor={color_stroke};strokeWidth=1.4;")
    d.add_text(x + chip_w + 4, y - 2, label_width, 26, label,
               style="text;html=1;align=left;verticalAlign=middle;fontSize=11;fontColor=#374151;")
    return x + chip_w + 4 + label_width + 16   # 16 px gap to next chip


def build_legend_row(d: Diagram, y: float, items: list[tuple[str, str, str, float]],
                     left_pad: float = 60) -> None:
    """Lay out a row of legend chips from left to right with no overlap."""
    x = left_pad
    for fill, stroke, label, lw in items:
        x = legend_chip(d, x, y, fill, stroke, label, label_width=lw)


# ---------------------------------------------------------------------------
# Page 1 — Baseline 2N self-attention
# ---------------------------------------------------------------------------


BASELINE_CLEAN_WORDS = ["The", "cat", "sat", "on"]
BASELINE_NOISY_WORDS = ["the", "mat", "[MASK]", "[MASK]"]


def build_page_baseline() -> Diagram:
    d = Diagram("1 — Baseline self-attention (2N concat)", page_w=1400, page_h=900)

    title(
        d,
        "Baseline (BD3-LMs): 2N self-attention on concatenated clean + noisy",
        "Every noisy token re-computes Q, K, V and attends over ALL 2N tokens — each denoising step.",
    )

    # Row 1: 4 clean past tokens + 4 noisy current tokens, all in a single 2N row.
    tok_w, tok_h, tok_gap = 110, 70, 16
    row_y = 200
    n_clean = len(BASELINE_CLEAN_WORDS)
    n_noisy = len(BASELINE_NOISY_WORDS)
    total = n_clean + n_noisy
    row_w = total * tok_w + (total - 1) * tok_gap
    start_x = (d.page_w - row_w) / 2

    # Section labels above the row.
    clean_w = n_clean * tok_w + (n_clean - 1) * tok_gap
    noisy_w = n_noisy * tok_w + (n_noisy - 1) * tok_gap
    section_label(d, start_x, row_y - 60, clean_w,
                  "Clean past  (N tokens)", color=CLEAN_STROKE)
    section_label(d, start_x + clean_w + tok_gap, row_y - 60, noisy_w,
                  "Noisy current block  (N tokens)", color=NOISY_STROKE)

    tokens: list[str] = []
    qkv: list[tuple[str, str, str]] = []
    words = BASELINE_CLEAN_WORDS + BASELINE_NOISY_WORDS

    for i, w in enumerate(words):
        is_clean = i < n_clean
        x = start_x + i * (tok_w + tok_gap)
        if is_clean:
            style = token_style(CLEAN_FILL, CLEAN_STROKE, CLEAN_TEXT)
        else:
            style = token_style(NOISY_FILL, NOISY_STROKE, NOISY_TEXT)
        tid = d.add_node(x, row_y, tok_w, tok_h, w, style=style)
        tokens.append(tid)
        # Q/K/V triplet under every token.
        qkv.append(draw_qkv_triplet(d, cx=x + tok_w / 2, top_y=row_y + tok_h + 18))

    # Attention matrix: 2N × 2N with every cell "on" — that's the whole baseline story.
    matrix_x = 120
    matrix_y = row_y + tok_h + 150
    row_labels = [f"t{i+1}" for i in range(total)]
    col_labels = [f"t{j+1}" for j in range(total)]
    row_is_clean = [i < n_clean for i in range(total)]
    pattern = [[True] * total for _ in range(total)]
    draw_attention_matrix(
        d, matrix_x, matrix_y, pattern,
        cell=28,
        labels_rows=row_labels, labels_cols=col_labels,
        row_is_clean=row_is_clean, col_is_clean=row_is_clean,
        title="Attention pattern  (filled = attention is computed)",
        legend_line="rows = queries   ·   cols = keys   ·   2N × 2N = 64 cells — every single one computed",
    )

    # To the right of the matrix: a callout panel summarising the cost.
    panel_x = matrix_x + total * 28 + 120
    panel_y = matrix_y - 10
    d.add_node(
        panel_x, panel_y, d.page_w - panel_x - 60, 330,
        "",
        style=("rounded=1;whiteSpace=wrap;html=1;"
               "fillColor=#fff1f2;strokeColor=#b91c1c;strokeWidth=1.8;"),
    )
    d.add_text(
        panel_x + 20, panel_y + 20, d.page_w - panel_x - 100, 34,
        "Baseline — full 2N self-attention",
        style="text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#7f1d1d;",
    )
    d.add_text(
        panel_x + 20, panel_y + 60, d.page_w - panel_x - 100, 260,
        ("• Noisy + clean tokens concatenated into a single 2N-length sequence.\n\n"
         "• Q, K, V recomputed for ALL 2N positions at EVERY inner step.\n\n"
         "• Attention matrix is 2N × 2N — every cell filled.\n\n"
         "• Compute:  O((2N)² · d) per step\n"
         "• Memory:   O((2N)²)\n\n"
         "• Clean past is re-encoded every step even though it is frozen."),
        style=("text;html=1;align=left;verticalAlign=top;fontSize=13;fontStyle=0;"
               "fontColor=#7f1d1d;spacingLeft=4;spacingRight=4;"),
    )

    # Bottom caption.
    d.add_text(40, d.page_h - 90, d.page_w - 80, 60,
               ("Every diffusion step, the clean-past tokens are re-encoded too — even though they are frozen. "
                "The dense 2N × 2N pattern above is the cost ABD3 removes."),
               style="text;html=1;align=center;verticalAlign=middle;fontSize=13;fontStyle=0;fontColor=#374151;spacingLeft=20;spacingRight=20;")

    # Legend
    build_legend_row(d, y=110, items=[
        (CLEAN_FILL, CLEAN_STROKE, "Clean past token", 160),
        (NOISY_FILL, NOISY_STROKE, "Noisy current-block token", 200),
        (Q_COLOR, Q_COLOR, "Q (query)", 90),
        (K_COLOR, K_COLOR, "K (key)", 80),
        (V_COLOR, V_COLOR, "V (value)", 90),
    ])

    return d


# ---------------------------------------------------------------------------
# Page 2 — ABD3 two-stream
# ---------------------------------------------------------------------------


TWOSTREAM_CLEAN = ["The", "cat", "sat", "on"]
TWOSTREAM_NOISY = ["the", "mat", "[MASK]", "[MASK]"]


def build_page_twostream() -> Diagram:
    d = Diagram("2 — ABD3 two-stream (self + cross)", page_w=1400, page_h=1050)

    title(
        d,
        "ABD3 two-stream attention: clean stream (K, V) + noisy stream (Q, K, V)",
        "Generation-stream queries self-attend within the block and cross-attend to the frozen clean stream.",
    )

    tok_w, tok_h, gap = 110, 70, 18
    n = len(TWOSTREAM_CLEAN)
    row_w = n * tok_w + (n - 1) * gap
    start_x = (d.page_w - row_w) / 2

    # --- Clean stream (top) ---
    clean_y = 210
    section_label(d, start_x - 20, clean_y - 42, row_w + 40,
                  "Context stream — frozen clean past (K, V only, computed ONCE)",
                  color=CLEAN_STROKE)

    clean_tokens: list[str] = []
    clean_kv: list[tuple[str, str]] = []
    for i, w in enumerate(TWOSTREAM_CLEAN):
        x = start_x + i * (tok_w + gap)
        style = token_style(CLEAN_FILL, CLEAN_STROKE, CLEAN_TEXT)
        tid = d.add_node(x, clean_y, tok_w, tok_h, w, style=style)
        clean_tokens.append(tid)
        clean_kv.append(draw_kv_pair(d, cx=x + tok_w / 2, top_y=clean_y + tok_h + 16))

    # --- Generation stream (bottom) ---
    noisy_y = 540
    section_label(d, start_x - 20, noisy_y - 42, row_w + 40,
                  "Generation stream — noisy current block (Q, K, V re-computed every inner step)",
                  color=NOISY_STROKE)

    noisy_tokens: list[str] = []
    noisy_qkv: list[tuple[str, str, str]] = []
    for i, w in enumerate(TWOSTREAM_NOISY):
        x = start_x + i * (tok_w + gap)
        style = token_style(NOISY_FILL, NOISY_STROKE, NOISY_TEXT)
        tid = d.add_node(x, noisy_y, tok_w, tok_h, w, style=style)
        noisy_tokens.append(tid)
        noisy_qkv.append(draw_qkv_triplet(d, cx=x + tok_w / 2, top_y=noisy_y + tok_h + 16))

    # --- Cross-attention: each noisy Q ↑ to every clean K.
    # Anchor on the Q pill (ids stored in noisy_qkv[i][0]) and each clean K pill (clean_kv[j][0]).
    for (qi, _, _) in noisy_qkv:
        for (kj, _) in clean_kv:
            style = (
                "endArrow=classic;html=1;rounded=0;"
                "strokeColor=#0e7490;strokeWidth=1.1;dashed=1;"
                "exitX=0.5;exitY=0;entryX=0.5;entryY=1;"
            )
            d.add_edge(qi, kj, style=style)

    # --- Self-attention inside the generation stream: noisy Q → noisy K (all-to-all within block).
    for (qi, _, _) in noisy_qkv:
        for (_, kj, _) in noisy_qkv:
            if qi == kj:
                continue
            style = (
                "endArrow=classic;html=1;rounded=0;"
                f"strokeColor={Q_COLOR};strokeWidth=1.0;dashed=1;"
                "exitX=0;exitY=0.5;entryX=1;entryY=0.5;"
            )
            d.add_edge(qi, kj, style=style)

    # --- Side panels annotating the two attention types.
    # Cross-attention panel (left).
    d.add_node(40, 330, 360, 130,
               "Cross-attention\n"
               "Q  (noisy)  ⟶  K, V  (clean)\n"
               "• reads frozen past once per inner step\n"
               "• contributes O(B · N · d) per step",
               style=("rounded=1;whiteSpace=wrap;html=1;"
                      "fillColor=#ecfeff;strokeColor=#0e7490;fontColor=#155e75;"
                      "fontSize=12;fontStyle=0;strokeWidth=1.8;align=left;verticalAlign=middle;spacingLeft=12;"))

    # Self-attention panel (right).
    d.add_node(d.page_w - 400, 330, 360, 130,
               "Self-attention\n"
               "Q  (noisy)  ⟶  K, V  (noisy, same block)\n"
               "• B × B within the current block\n"
               "• contributes O(B²·d) per step — tiny",
               style=("rounded=1;whiteSpace=wrap;html=1;"
                      f"fillColor=#fff7ed;strokeColor={NOISY_STROKE};fontColor={NOISY_TEXT};"
                      "fontSize=12;fontStyle=0;strokeWidth=1.8;align=left;verticalAlign=middle;spacingLeft=12;"))

    # Attention matrix — ABD3 pattern. Only noisy rows have attention; those
    # rows fill across all 2N columns (N clean via cross-attn + N noisy via self-attn).
    # Clean rows are completely empty (no Q, no attention scores for clean past).
    matrix_size = 2 * n
    mpattern = [[False] * matrix_size for _ in range(matrix_size)]
    row_is_clean_m = [i < n for i in range(matrix_size)]
    for i in range(matrix_size):
        for j in range(matrix_size):
            # Only noisy-stream queries (rows i >= n) compute attention,
            # and they attend to all 2N keys.
            if i >= n:
                mpattern[i][j] = True
    draw_attention_matrix(
        d, 120, 760, mpattern,
        cell=22,
        labels_rows=[f"t{i+1}" for i in range(matrix_size)],
        labels_cols=[f"t{j+1}" for j in range(matrix_size)],
        row_is_clean=row_is_clean_m, col_is_clean=row_is_clean_m,
        fill_on="#86efac", stroke_on="#15803d",
        title="Attention pattern — only noisy-query rows filled (clean rows skipped)",
        legend_line="half the matrix is free — 50% fewer attention scores than the baseline",
    )

    # Cost footer — shrink to single line so the matrix has room above.
    d.add_text(40, d.page_h - 56, d.page_w - 80, 36,
               ("Compute per inner step: O(B · N · d) cross + O(B² · d) self  ≪  O((2N)² · d) baseline. "
                "Clean-stream K,V computed ONCE per block (not every inner step)."),
               style="text;html=1;align=center;verticalAlign=middle;fontSize=13;fontStyle=0;fontColor=#065f46;spacingLeft=20;spacingRight=20;")

    # Legend (two rows to avoid cramming).
    build_legend_row(d, y=106, items=[
        (CLEAN_FILL, CLEAN_STROKE, "Clean past (frozen)", 180),
        (NOISY_FILL, NOISY_STROKE, "Noisy current block", 170),
        (Q_COLOR, Q_COLOR, "Q (query)", 90),
        (K_COLOR, K_COLOR, "K (key)", 80),
        (V_COLOR, V_COLOR, "V (value)", 90),
    ])
    build_legend_row(d, y=132, items=[
        ("#ecfeff", "#0e7490", "cross-attn (Q → clean K,V)", 230),
        ("#fff7ed", NOISY_STROKE, "self-attn (Q → noisy K,V)", 230),
    ])

    return d


# ---------------------------------------------------------------------------
# Page 3 — KV caching over blocks
# ---------------------------------------------------------------------------


KV_BLOCKS = [
    ("Block 0", ["The", "cat", "sat", "on"], "done"),
    ("Block 1", ["the", "mat", "and", "purred"], "active"),
    ("Block 2", ["very", "content", "-ed", "-ly."], "future"),
]


def build_page_kvcache() -> Diagram:
    d = Diagram("3 — KV caching across blocks", page_w=1500, page_h=900)

    title(
        d,
        "KV cache in ABD3: each block writes its K, V exactly once — future blocks read it for free",
        "Time flows left → right. Decoded blocks stream K, V into the cache; the active block reads the cache via cross-attention.",
    )

    # 3-column timeline layout.
    col_w = 440
    col_gap = 30
    n_cols = len(KV_BLOCKS)
    total_w = n_cols * col_w + (n_cols - 1) * col_gap
    start_x = (d.page_w - total_w) / 2

    tok_w, tok_h, tok_gap = 82, 60, 10
    block_inner_w = 4 * tok_w + 3 * tok_gap

    # Cache panel runs across the bottom.
    cache_x = start_x
    cache_y = 650
    cache_w = total_w
    cache_h = 140
    cache_id = d.add_node(
        cache_x, cache_y, cache_w, cache_h,
        "KV cache (grows as blocks complete)",
        style=("rounded=1;whiteSpace=wrap;html=1;"
               f"fillColor={CACHE_FILL};strokeColor={CACHE_STROKE};fontColor={CACHE_TEXT};"
               "fontSize=14;fontStyle=1;strokeWidth=2;verticalAlign=top;spacingTop=10;"),
    )

    # Sub-slots inside the cache — one per past block.
    slot_w = (cache_w - 40) / n_cols
    for i, (name, _, _) in enumerate(KV_BLOCKS):
        sx = cache_x + 20 + i * slot_w
        # Slot starts hollow; filled once its block completes.
        if i == 0:
            fill = "#ddd6fe"
            stroke = CACHE_STROKE
            text = f"{name}  K,V  ✓ cached"
        elif i == 1:
            fill = "#f3e8ff"
            stroke = "#a78bfa"
            text = f"{name}  K,V  (pending — block decoding)"
        else:
            fill = "#faf5ff"
            stroke = "#c4b5fd"
            text = f"{name}  K,V  — not yet"
        d.add_node(sx, cache_y + 46, slot_w - 10, cache_h - 66, text,
                   style=("rounded=1;whiteSpace=wrap;html=1;"
                          f"fillColor={fill};strokeColor={stroke};fontColor={CACHE_TEXT};"
                          "fontSize=11;fontStyle=1;strokeWidth=1.4;align=center;verticalAlign=middle;dashed=%s;" %
                          ("1" if i == 2 else "0")))

    # Each column: a per-block panel.
    col_headers: list[str] = []
    active_token_qids: list[str] = []  # Q pills of the active block (for arrows to cache)

    # Layout inside each column (top → bottom):
    #   header  ·  annotation text  ·  token row  ·  Q/K/V pills  ·  → cache
    # Putting annotations ABOVE the tokens keeps the pills-to-cache arrow path clear.
    for i, (name, words, state) in enumerate(KV_BLOCKS):
        cx = start_x + i * (col_w + col_gap)
        state_fill, state_stroke = {
            "done":   (DONE_FILL, DONE_STROKE),
            "active": ("#fff7ed", NOISY_STROKE),
            "future": (FUTURE_FILL, FUTURE_STROKE),
        }[state]
        col_id = d.add_node(cx, 160, col_w, 460,
                            "", style=("rounded=1;whiteSpace=wrap;html=1;"
                                        f"fillColor={state_fill};strokeColor={state_stroke};"
                                        "strokeWidth=1.6;dashed=%s;opacity=80;" % ("1" if state != "active" else "0")))
        # Column title.
        title_text = {
            "done":   f"{name} — decoded ✓",
            "active": f"{name} — actively refining",
            "future": f"{name} — future (masked)",
        }[state]
        title_color = {
            "done": DONE_STROKE,
            "active": NOISY_STROKE,
            "future": FUTURE_STROKE,
        }[state]
        d.add_text(cx + 10, 170, col_w - 20, 28, title_text,
                   style=(f"text;html=1;align=center;verticalAlign=middle;"
                          f"fontSize=14;fontStyle=1;fontColor={title_color};"))
        col_headers.append(title_text)

        # Annotation text BELOW the header but ABOVE the tokens.
        annot = {
            "done":   "K, V computed ONCE → streamed to cache. Never recomputed.",
            "active": "Q, K, V recomputed every inner step.\nQ cross-attends into cache.",
            "future": "Not started. Will read cache of Block 0 + Block 1 later.",
        }[state]
        d.add_text(cx + 14, 205, col_w - 28, 70, annot,
                   style=(f"text;html=1;align=center;verticalAlign=middle;"
                          f"fontSize=11;fontStyle=0;fontColor={title_color};"
                          "spacingLeft=6;spacingRight=6;"))

        # Token row — positioned in the lower half of the column so the Q/K/V
        # pills are close to the cache panel at the bottom.
        trow_x = cx + (col_w - block_inner_w) / 2
        trow_y = 320
        for j, w in enumerate(words):
            x = trow_x + j * (tok_w + tok_gap)
            if state == "done":
                style = token_style(DONE_FILL, DONE_STROKE, DONE_TEXT)
                text = w
            elif state == "active":
                style = token_style(NOISY_FILL, NOISY_STROKE, NOISY_TEXT)
                text = w if j < 2 else "[MASK]"
            else:
                style = token_style(FUTURE_FILL, FUTURE_STROKE, FUTURE_TEXT)
                text = "[MASK]"
            tid = d.add_node(x, trow_y, tok_w, tok_h, text, style=style)

            if state == "done":
                kvids = draw_kv_pair(d, cx=x + tok_w / 2, top_y=trow_y + tok_h + 16)
                for kid in kvids:
                    style_write = (
                        "endArrow=classic;html=1;rounded=0;"
                        f"strokeColor={CACHE_STROKE};strokeWidth=1.6;"
                        "exitX=0.5;exitY=1;entryX=0.5;entryY=0;"
                    )
                    d.add_edge(kid, cache_id, style=style_write, label="")
            elif state == "active":
                qid, kid, vid = draw_qkv_triplet(d, cx=x + tok_w / 2, top_y=trow_y + tok_h + 16)
                active_token_qids.append(qid)

        # Small "reads cache" hint under the active block, pointing toward
        # the cache sub-slots on the left.
        if state == "active":
            d.add_text(cx + 10, trow_y + tok_h + 60, col_w - 20, 22,
                       "↓ cross-attention reads ↓",
                       style=(f"text;html=1;align=center;verticalAlign=middle;"
                              f"fontSize=11;fontStyle=2;fontColor={Q_COLOR};"))
        elif state == "done":
            d.add_text(cx + 10, trow_y + tok_h + 60, col_w - 20, 22,
                       "↓ K,V written to cache ↓",
                       style=(f"text;html=1;align=center;verticalAlign=middle;"
                              f"fontSize=11;fontStyle=2;fontColor={CACHE_STROKE};"))

    # Read arrows: from each active-block Q up/down to the cache (cross-attention reads).
    for qid in active_token_qids:
        style_read = (
            "endArrow=classic;html=1;rounded=0;"
            f"strokeColor={Q_COLOR};strokeWidth=1.3;dashed=1;"
            "exitX=0.5;exitY=1;entryX=0.5;entryY=0;"
        )
        # Edge points toward the cache's first (completed) slot.
        d.add_edge(qid, cache_id, style=style_read, label="")

    # Legend (single row, spaced).
    build_legend_row(d, y=108, items=[
        (DONE_FILL, DONE_STROKE, "decoded block", 130),
        (NOISY_FILL, NOISY_STROKE, "active block", 120),
        (FUTURE_FILL, FUTURE_STROKE, "future block", 120),
        (CACHE_FILL, CACHE_STROKE, "KV cache", 90),
        (CACHE_STROKE, CACHE_STROKE, "K,V write (done → cache)", 210),
        (Q_COLOR, Q_COLOR, "cross-attn read (active → cache)", 260),
    ])

    # Bottom takeaway.
    d.add_text(40, d.page_h - 70, d.page_w - 80, 50,
               ("Amortised cost of attending over the entire past: O(B · (block-index · N) · d) per inner step, "
                "vs. O((block-index · N + B)² · d) for a naive 2N re-concatenation. "
                "For a long sequence generated block-by-block, the KV cache is what makes ABD3 fast."),
               style="text;html=1;align=center;verticalAlign=middle;fontSize=13;fontStyle=0;fontColor=#374151;spacingLeft=20;spacingRight=20;")

    return d


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def build_file(out_path: pathlib.Path) -> None:
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2026-04-23T00:00:00.000Z",
        "agent": "abd3/visualizer",
        "etag": "abd3attn",
        "version": "24.0.0",
        "type": "device",
    })

    for page in (build_page_baseline(), build_page_twostream(), build_page_kvcache()):
        mxfile.append(page.to_xml())

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[drawio] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate ABD3 attention drawio.")
    p.add_argument("--out", default="figures/abd3_attention.drawio", type=pathlib.Path)
    args = p.parse_args()
    build_file(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
