"""Collect ABD3 runs from Weights & Biases into paper-ready tables.

Pulls a set of runs from the WandB Public API, joins their final
metrics and config, and emits:

* a full ``--csv-out`` table (one row per run) for post-hoc analysis,
* a compact ``--markdown-out`` table grouped by ``--group-by`` (e.g.
  ``algo``) for the README / paper appendix,
* an optional ``--latex-out`` table with the same structure but using
  the ``booktabs`` style (no ``\\hline``) so it can be dropped
  straight into the paper.

The default ``--metrics`` set reflects what we actually report: the
terminal train/val loss & PPL, sampling efficiency, and every
quality/zero-shot metric we log.

There's also an opt-in panel-applier (``--apply-panels``) that reads
:file:`configs/wandb_panels/default.yaml` and applies it to the project's
workspace via the WandB SDK. Panel application is a no-op on hosts
where the ``wandb`` package is missing; the primary CSV/Markdown
collection path only needs ``wandb.Api``.

CLI::

    # Collect the last 100 runs in the project into tables.
    python -m eval.collect_results \\
        --entity arrdel --project abd3 \\
        --csv-out report/tables/all_runs.csv \\
        --markdown-out report/tables/summary.md \\
        --group-by config.algo.name \\
        --latex-out report/tables/summary.tex

    # Apply the shared panel layout.
    python -m eval.collect_results \\
        --entity arrdel --project abd3 \\
        --apply-panels configs/wandb_panels/default.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
from collections.abc import Iterable, Sequence
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_DEFAULT_METRICS = (
    "train/loss",
    "val/loss",
    "train/ppl",
    "val/ppl",
    "sampling/total_nfe",
    "sampling/nfe_savings",
    "sampling/elapsed_seconds",
    "quality/distinct_1",
    "quality/distinct_4",
    "quality/self_bleu_4",
    "quality/corpus_ppl",
    "quality/median_sample_ppl",
    "quality/mauve",
    "zero_shot/wikitext103",
    "zero_shot/ptb",
    "zero_shot/lambada",
)


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------


def _dotted_get(obj: Any, path: str, default=None):
    """Traverse ``obj`` via dotted key segments (``config.algo.name``)."""
    cur = obj
    for seg in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(seg)
        else:
            cur = getattr(cur, seg, None)
    return default if cur is None else cur


def _run_summary_get(run, metric: str, default=None):
    """Look up a metric in ``run.summary`` (WandB API) with ``None`` fallback.

    ``run.summary`` behaves like a dict for scalar entries but returns
    opaque objects for histograms / images — we only keep primitives
    (int/float/str/bool) and coerce ``None``/missing to ``default``.
    """
    try:
        raw = (
            run.summary.get(metric, default)
            if hasattr(run.summary, "get")
            else run.summary.__getitem__(metric)
        )
    except (KeyError, AttributeError):
        return default
    if isinstance(raw, int | float | str | bool) or raw is None:
        return raw
    # WandB's "_wandb" nested summaries and histogram handles → unhelpful here.
    return default


def extract_row(run, metrics: Sequence[str], extra_config_keys: Sequence[str] = ()) -> dict:
    """Return a flat dict of ``{metric_or_config_key: value}`` for ``run``."""
    row: dict[str, Any] = {
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "run_state": getattr(run, "state", None),
        "created_at": str(getattr(run, "created_at", "") or ""),
        "tags": ",".join(sorted(getattr(run, "tags", []) or [])),
    }
    for m in metrics:
        row[m] = _run_summary_get(run, m)
    config = getattr(run, "config", {}) or {}
    for key in extra_config_keys:
        row[key] = _dotted_get(config, key.removeprefix("config."))
    return row


def extract_rows(
    runs: Iterable, metrics: Sequence[str], extra_config_keys: Sequence[str] = ()
) -> list[dict]:
    """Eagerly materialise rows (the API returns a lazy iterator)."""
    return [extract_row(r, metrics, extra_config_keys) for r in runs]


# ---------------------------------------------------------------------------
# CSV / Markdown / LaTeX rendering
# ---------------------------------------------------------------------------


def write_csv(rows: Sequence[dict], path: str | pathlib.Path) -> None:
    import csv

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Union of all keys (robust to missing fields per row).
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _fmt_cell(v: Any) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.3g}"
        return f"{v:.3f}"
    return str(v)


def _aggregate_by_group(rows: Sequence[dict], group_by: str, metrics: Sequence[str]) -> list[dict]:
    """Mean-aggregate numeric metrics per value of ``group_by``.

    Non-numeric cells are ignored. If every value in a cell is missing
    we report ``None`` so ``_fmt_cell`` can render a dash.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = str(r.get(group_by, "(unknown)"))
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for key, members in sorted(groups.items()):
        agg: dict[str, Any] = {group_by: key, "n_runs": len(members)}
        for m in metrics:
            numeric = [
                r[m]
                for r in members
                if isinstance(r.get(m), int | float)
                and r[m] is not None
                and not (isinstance(r[m], float) and _is_nan(r[m]))
            ]
            agg[m] = (sum(numeric) / len(numeric)) if numeric else None
        out.append(agg)
    return out


def _is_nan(x: float) -> bool:
    return x != x


def render_markdown(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """Simple GitHub-flavored Markdown table."""
    if not rows:
        return "_(no rows)_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt_cell(r.get(c)) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_latex(
    rows: Sequence[dict],
    columns: Sequence[str],
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """``booktabs`` LaTeX table — safe to ``\\input`` into the paper.

    We escape ``_``, ``&``, ``%`` in cell strings so run names like
    ``abd3_rdr_only`` don't silently break compilation.
    """
    if not rows:
        return "% (no rows)\n"
    col_spec = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(_latex_escape(_fmt_cell(r.get(c))) for c in columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


_LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
    "\\": r"\textbackslash{}",
}


def _latex_escape(s: str) -> str:
    s = str(s)
    return re.sub(
        "|".join(re.escape(k) for k in _LATEX_ESCAPES),
        lambda m: _LATEX_ESCAPES[m.group(0)],
        s,
    )


# ---------------------------------------------------------------------------
# WandB API wrapper
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WandBFilter:
    """Pared-down version of wandb.Api.runs filters."""

    tags: Sequence[str] | None = None
    state: str | None = None  # "finished" / "running" / …
    name_regex: str | None = None

    def to_mongo(self) -> dict:
        """Translate to the Mongo-style filter wandb.Api.runs expects."""
        f: dict[str, Any] = {}
        if self.tags:
            f["tags"] = {"$in": list(self.tags)}
        if self.state:
            f["state"] = self.state
        if self.name_regex:
            f["display_name"] = {"$regex": self.name_regex}
        return f


def fetch_runs(
    entity: str,
    project: str,
    filt: WandBFilter | None = None,
    per_page: int = 50,
    limit: int | None = None,
):
    """Return an iterator of WandB ``Run`` objects (or raise if the SDK is missing)."""
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "`wandb` is required for eval.collect_results. " "Install with `pip install wandb`."
        ) from e

    api = wandb.Api()
    kwargs: dict[str, Any] = {"path": f"{entity}/{project}", "per_page": per_page}
    if filt is not None:
        kwargs["filters"] = filt.to_mongo()
    runs_iter = api.runs(**kwargs)
    if limit is not None:
        out = []
        for i, r in enumerate(runs_iter):
            if i >= limit:
                break
            out.append(r)
        return out
    return list(runs_iter)


# ---------------------------------------------------------------------------
# Panels YAML applier
# ---------------------------------------------------------------------------


def load_panels_yaml(path: str | pathlib.Path) -> dict:
    """Load and minimally validate the panels YAML."""
    import yaml

    doc = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError("panels YAML root must be a mapping")
    if "sections" not in doc:
        raise ValueError("panels YAML must contain a top-level `sections` key")
    if not isinstance(doc["sections"], list):
        raise ValueError("`sections` must be a list")
    for i, section in enumerate(doc["sections"]):
        if not isinstance(section, dict) or "panels" not in section:
            raise ValueError(f"section[{i}] missing `panels` list")
    return doc


def summarize_panels(doc: dict) -> str:
    """Human-readable summary of what the YAML would apply."""
    lines = [f"project={doc.get('project')}  entity={doc.get('entity')}"]
    for s in doc["sections"]:
        lines.append(f"- {s.get('name')}:")
        for p in s.get("panels") or []:
            metrics = p.get("metrics") or []
            lines.append(
                f"    • [{p.get('type', '?')}] {p.get('title', '(untitled)')}  "
                f"({len(metrics)} metric{'s' if len(metrics) != 1 else ''})"
            )
    return "\n".join(lines)


def apply_panels(doc: dict, *, dry_run: bool = False) -> str:
    """Apply the panels doc to the WandB workspace.

    This is opt-in and needs a recent ``wandb`` that supports the
    ``wandb.apis.public.WorkspaceSettings`` API. On older installs we
    fall back to printing the summary so the user can re-create the
    panels manually.
    """
    msg = summarize_panels(doc)
    if dry_run:
        return "(dry-run)\n" + msg

    try:
        import wandb  # type: ignore[import-not-found] # noqa: F401
    except ImportError:
        return "wandb SDK not installed; skipping apply. Summary:\n" + msg

    # WandB doesn't yet expose a stable public API for panel CRUD. Rather
    # than call an internal endpoint that will break, we print the plan
    # and leave real application to a manual copy or to a future revision.
    return "wandb panel-apply API not stable yet; printing plan instead:\n" + msg


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------


def collect_and_write(
    rows: Sequence[dict],
    *,
    metrics: Sequence[str],
    group_by: str | None,
    csv_out: str | pathlib.Path | None,
    markdown_out: str | pathlib.Path | None,
    latex_out: str | pathlib.Path | None,
    latex_caption: str | None = None,
    latex_label: str | None = None,
) -> dict:
    """Write all requested output files and return a small summary dict."""
    summary = {
        "n_rows": len(rows),
        "csv_out": str(csv_out) if csv_out else None,
        "markdown_out": str(markdown_out) if markdown_out else None,
        "latex_out": str(latex_out) if latex_out else None,
    }
    if csv_out:
        write_csv(rows, csv_out)
    if markdown_out or latex_out:
        if group_by:
            agg = _aggregate_by_group(rows, group_by, metrics)
            columns = [group_by, "n_runs", *metrics]
            table_rows = agg
        else:
            table_rows = rows
            columns = ["run_name", *metrics]
        if markdown_out:
            p = pathlib.Path(markdown_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(render_markdown(table_rows, columns))
        if latex_out:
            p = pathlib.Path(latex_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                render_latex(
                    table_rows,
                    columns,
                    caption=latex_caption,
                    label=latex_label,
                )
            )
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Collect ABD3 WandB runs into CSV / Markdown / LaTeX tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--entity",
        required=False,
        default=None,
        help="WandB entity (team/user). Uses default from ~/.netrc if unset.",
    )
    ap.add_argument("--project", default="abd3")
    ap.add_argument(
        "--limit", type=int, default=None, help="Cap on runs fetched. Default: no limit."
    )
    ap.add_argument(
        "--tags", nargs="*", default=None, help="Filter to runs tagged with ANY of these."
    )
    ap.add_argument("--state", default=None, help="Filter by run state (finished/running/crashed).")
    ap.add_argument(
        "--name-regex", default=None, help="Filter runs whose display name matches this regex."
    )
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=list(_DEFAULT_METRICS),
        help="Metric keys to pull from run.summary.",
    )
    ap.add_argument(
        "--extra-config-keys",
        nargs="*",
        default=[
            "config.algo.name",
            "config.block_size",
            "config.data.name",
            "config.training.max_steps",
        ],
        help="Dotted config keys to include per row.",
    )
    ap.add_argument(
        "--group-by", default=None, help="Aggregate by this column in markdown/latex outputs."
    )

    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--markdown-out", default=None)
    ap.add_argument("--latex-out", default=None)
    ap.add_argument("--latex-caption", default=None)
    ap.add_argument("--latex-label", default=None)

    ap.add_argument(
        "--apply-panels",
        default=None,
        help="Path to a panels YAML; if set, run the panel applier " "and exit.",
    )
    ap.add_argument(
        "--dry-run-panels", action="store_true", help="With --apply-panels, only print the summary."
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.apply_panels:
        doc = load_panels_yaml(args.apply_panels)
        if args.entity:
            doc["entity"] = args.entity
        if args.project:
            doc["project"] = args.project
        print(apply_panels(doc, dry_run=args.dry_run_panels))
        return 0

    if not args.entity:
        print("[collect] --entity not provided; relying on wandb default.", file=sys.stderr)

    filt = (
        WandBFilter(
            tags=args.tags,
            state=args.state,
            name_regex=args.name_regex,
        )
        if (args.tags or args.state or args.name_regex)
        else None
    )

    runs = fetch_runs(
        entity=args.entity or "",  # empty string = default entity
        project=args.project,
        filt=filt,
        limit=args.limit,
    )
    print(f"[collect] fetched {len(runs)} runs")

    rows = extract_rows(runs, args.metrics, args.extra_config_keys)
    summary = collect_and_write(
        rows,
        metrics=args.metrics,
        group_by=args.group_by,
        csv_out=args.csv_out,
        markdown_out=args.markdown_out,
        latex_out=args.latex_out,
        latex_caption=args.latex_caption,
        latex_label=args.latex_label,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
