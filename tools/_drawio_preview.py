"""Render a low-fidelity preview of every page of a drawio file using
matplotlib. Meant only for iterating on layout locally.

Emits one PNG per diagram under figures/_preview/.
"""

import pathlib
import xml.etree.ElementTree as ET
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_style(s: str) -> dict:
    d = {}
    for part in (s or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d


def render(drawio_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    tree = ET.parse(drawio_path)
    root = tree.getroot()
    out_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, diagram in enumerate(root.findall("diagram")):
        name = diagram.get("name", f"page{page_idx}")
        model = diagram.find("mxGraphModel")
        page_w = float(model.get("pageWidth", 1400))
        page_h = float(model.get("pageHeight", 900))

        id_to_geom: dict[str, tuple[float, float, float, float]] = {}
        cells = diagram.findall(".//mxCell")

        fig, ax = plt.subplots(figsize=(page_w / 120, page_h / 120), dpi=120)
        ax.set_xlim(0, page_w)
        ax.set_ylim(page_h, 0)   # flip Y to match drawio
        ax.set_aspect("equal")
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        ax.set_title(name, fontsize=14)

        # Draw vertices first.
        for cell in cells:
            if cell.get("vertex") != "1":
                continue
            geom = cell.find("mxGeometry")
            if geom is None:
                continue
            x = float(geom.get("x", 0))
            y = float(geom.get("y", 0))
            w = float(geom.get("width", 0))
            h = float(geom.get("height", 0))
            id_to_geom[cell.get("id")] = (x, y, w, h)
            sty = parse_style(cell.get("style", ""))
            fill = sty.get("fillColor", "#ffffff")
            stroke = sty.get("strokeColor", "#cccccc")
            try:
                rect = patches.FancyBboxPatch(
                    (x, y), w, h,
                    boxstyle="round,pad=0.0,rounding_size=3",
                    linewidth=1.0,
                    facecolor=fill,
                    edgecolor=stroke,
                )
                ax.add_patch(rect)
            except Exception:
                rect = patches.Rectangle((x, y), w, h, linewidth=1.0,
                                         facecolor=fill, edgecolor=stroke)
                ax.add_patch(rect)
            val = cell.get("value", "")
            if val:
                text = re.sub(r"<[^>]+>", "", val)  # strip HTML
                fontsize_raw = sty.get("fontSize", "10")
                try:
                    fontsize = float(fontsize_raw) * 0.7
                except ValueError:
                    fontsize = 8
                ax.text(x + w / 2, y + h / 2, text,
                        ha="center", va="center", fontsize=min(fontsize, 12),
                        color=sty.get("fontColor", "#111827"),
                        wrap=True)

        # Then edges.
        for cell in cells:
            if cell.get("edge") != "1":
                continue
            src = cell.get("source")
            tgt = cell.get("target")
            if not (src in id_to_geom and tgt in id_to_geom):
                continue
            sg = id_to_geom[src]
            tg = id_to_geom[tgt]
            sx, sy = sg[0] + sg[2] / 2, sg[1] + sg[3] / 2
            tx, ty = tg[0] + tg[2] / 2, tg[1] + tg[3] / 2
            sty = parse_style(cell.get("style", ""))
            stroke = sty.get("strokeColor", "#666666")
            dashed = sty.get("dashed") == "1"
            lw = float(sty.get("strokeWidth", 1.2))
            ax.annotate(
                "", xy=(tx, ty), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->",
                    color=stroke,
                    linestyle="--" if dashed else "-",
                    linewidth=lw * 0.7,
                    alpha=0.75,
                ),
            )

        out = out_dir / f"{drawio_path.stem}_page{page_idx+1}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    import sys
    drawio = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "figures/abd3_attention.drawio")
    out_dir = pathlib.Path("figures/_preview")
    render(drawio, out_dir)
