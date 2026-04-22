"""Tests for eval/collect_results.py.

WandB's Public API is mocked end-to-end with plain Python objects. We
assert on:

* row extraction (dotted config lookup, non-primitive stripping)
* CSV / Markdown / LaTeX rendering
* per-group aggregation
* LaTeX escaping of underscore-rich run names
* panels YAML loader error paths
"""

from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import collect_results as cr  # noqa: E402

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSummary(dict):
    """Behaves like wandb's Summary — dict-like + ``.get`` + arbitrary objects."""

    pass


def _fake_run(**kw):
    return SimpleNamespace(
        id=kw.get("id", "abc123"),
        name=kw.get("name", "run"),
        state=kw.get("state", "finished"),
        created_at=kw.get("created_at", "2026-04-22T00:00:00"),
        tags=kw.get("tags", []),
        summary=_FakeSummary(kw.get("summary", {})),
        config=kw.get("config", {}),
    )


# ---------------------------------------------------------------------------
# Dotted lookup + row extraction
# ---------------------------------------------------------------------------


def test_dotted_get_traverses_nested_dict():
    assert cr._dotted_get({"a": {"b": {"c": 7}}}, "a.b.c") == 7


def test_dotted_get_returns_default_on_missing():
    assert cr._dotted_get({"a": {}}, "a.b.c", default="oops") == "oops"


def test_run_summary_get_strips_nonprimitive_values():
    run = _fake_run(summary={"good": 1.5, "bad": object()})
    assert cr._run_summary_get(run, "good") == 1.5
    assert cr._run_summary_get(run, "bad") is None
    assert cr._run_summary_get(run, "missing", default=42) == 42


def test_extract_row_pulls_config_and_metrics():
    run = _fake_run(
        id="r0",
        name="abd3-rdr",
        state="finished",
        tags=["rdr", "bs=4"],
        summary={"val/loss": 1.23, "val/ppl": 3.45},
        config={"algo": {"name": "abd3"}, "block_size": 4},
    )
    row = cr.extract_row(
        run,
        metrics=["val/loss", "val/ppl", "missing"],
        extra_config_keys=["config.algo.name", "config.block_size"],
    )
    assert row["run_id"] == "r0"
    assert row["run_name"] == "abd3-rdr"
    assert row["run_state"] == "finished"
    assert row["tags"] == "bs=4,rdr"  # sorted, comma-joined
    assert row["val/loss"] == 1.23
    assert row["val/ppl"] == 3.45
    assert row["missing"] is None
    assert row["config.algo.name"] == "abd3"
    assert row["config.block_size"] == 4


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def test_write_csv_headers_union_all_row_keys(tmp_path):
    rows = [
        {"a": 1, "b": 2},
        {"a": 3, "c": 4},  # missing b, extra c
    ]
    out = tmp_path / "o.csv"
    cr.write_csv(rows, out)
    text = out.read_text().splitlines()
    assert text[0].split(",") == ["a", "b", "c"]
    assert text[1] == "1,2,"
    assert text[2] == "3,,4"


def test_write_csv_is_noop_on_empty(tmp_path):
    out = tmp_path / "o.csv"
    cr.write_csv([], out)
    assert not out.exists()


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def test_fmt_cell_handles_missing_values():
    assert cr._fmt_cell(None) == "—"
    assert cr._fmt_cell("") == "—"


def test_fmt_cell_switches_to_scientific_for_tiny_floats():
    assert cr._fmt_cell(1e-5) == "1e-05"
    assert cr._fmt_cell(1500.5) == "1.5e+03"


def test_fmt_cell_default_floats():
    assert cr._fmt_cell(0.12345) == "0.123"
    assert cr._fmt_cell(42.0) == "42.000"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_by_group_averages_numeric_and_counts_members():
    rows = [
        {"algo": "abd3", "val/loss": 1.0},
        {"algo": "abd3", "val/loss": 2.0},
        {"algo": "baseline", "val/loss": 4.0},
        {"algo": "baseline", "val/loss": None},
    ]
    agg = cr._aggregate_by_group(rows, "algo", ["val/loss"])
    by_key = {r["algo"]: r for r in agg}
    assert by_key["abd3"]["val/loss"] == pytest.approx(1.5)
    assert by_key["abd3"]["n_runs"] == 2
    assert by_key["baseline"]["val/loss"] == 4.0
    assert by_key["baseline"]["n_runs"] == 2


def test_aggregate_by_group_reports_none_when_no_numeric_values():
    rows = [
        {"algo": "foo", "val/loss": None},
        {"algo": "foo", "val/loss": "N/A"},
    ]
    agg = cr._aggregate_by_group(rows, "algo", ["val/loss"])
    assert agg[0]["val/loss"] is None


# ---------------------------------------------------------------------------
# Markdown / LaTeX rendering
# ---------------------------------------------------------------------------


def test_render_markdown_emits_header_and_dashes():
    rows = [{"name": "a", "ppl": 5.0}, {"name": "b", "ppl": None}]
    md = cr.render_markdown(rows, ["name", "ppl"])
    lines = md.strip().splitlines()
    assert lines[0] == "| name | ppl |"
    assert "---" in lines[1]
    assert "5.000" in lines[2]
    assert "—" in lines[3]


def test_render_markdown_handles_empty():
    assert "(no rows)" in cr.render_markdown([], ["a"])


def test_render_latex_escapes_underscores_in_run_names():
    rows = [{"name": "abd3_rdr_only", "val": 1.0}]
    out = cr.render_latex(rows, ["name", "val"])
    assert r"abd3\_rdr\_only" in out
    assert r"\toprule" in out
    assert r"\bottomrule" in out


def test_render_latex_inserts_caption_and_label():
    out = cr.render_latex(
        [{"x": 1}],
        ["x"],
        caption="My Caption",
        label="tab:ablation",
    )
    assert r"\caption{My Caption}" in out
    assert r"\label{tab:ablation}" in out


def test_render_latex_handles_empty():
    assert "(no rows)" in cr.render_latex([], ["x"])


# ---------------------------------------------------------------------------
# collect_and_write orchestration
# ---------------------------------------------------------------------------


def test_collect_and_write_writes_all_requested_files(tmp_path):
    rows = [
        {"run_name": "a", "algo": "abd3", "val/loss": 1.1},
        {"run_name": "b", "algo": "abd3", "val/loss": 1.2},
        {"run_name": "c", "algo": "baseline", "val/loss": 2.0},
    ]
    csv_out = tmp_path / "out.csv"
    md_out = tmp_path / "out.md"
    tex_out = tmp_path / "out.tex"
    cr.collect_and_write(
        rows,
        metrics=["val/loss"],
        group_by="algo",
        csv_out=csv_out,
        markdown_out=md_out,
        latex_out=tex_out,
        latex_caption="Results",
    )
    assert csv_out.exists() and md_out.exists() and tex_out.exists()
    assert "val/loss" in md_out.read_text()
    assert r"\toprule" in tex_out.read_text()


def test_collect_and_write_without_group_by_uses_run_name_column(tmp_path):
    rows = [{"run_name": "a", "val/loss": 1.0}]
    md = tmp_path / "out.md"
    cr.collect_and_write(
        rows,
        metrics=["val/loss"],
        group_by=None,
        csv_out=None,
        markdown_out=md,
        latex_out=None,
    )
    text = md.read_text()
    assert "run_name" in text.splitlines()[0]


# ---------------------------------------------------------------------------
# WandBFilter -> Mongo
# ---------------------------------------------------------------------------


def test_filter_to_mongo_omits_empty_fields():
    assert cr.WandBFilter().to_mongo() == {}


def test_filter_to_mongo_populates_tags_state_regex():
    f = cr.WandBFilter(tags=["rdr", "mixed-bs"], state="finished", name_regex="^abd3-")
    m = f.to_mongo()
    assert m["tags"] == {"$in": ["rdr", "mixed-bs"]}
    assert m["state"] == "finished"
    assert m["display_name"] == {"$regex": "^abd3-"}


# ---------------------------------------------------------------------------
# Panels YAML loader
# ---------------------------------------------------------------------------


def _panels_dict():
    return {
        "project": "abd3",
        "entity": "arrdel",
        "sections": [
            {
                "name": "Training",
                "panels": [
                    {"title": "Loss", "type": "line", "metrics": ["train/loss", "val/loss"]},
                ],
            },
            {
                "name": "Quality",
                "panels": [
                    {"title": "MAUVE", "type": "bar", "metrics": ["quality/mauve"]},
                ],
            },
        ],
    }


def test_load_panels_yaml_accepts_valid_file(tmp_path):
    import yaml

    p = tmp_path / "panels.yaml"
    p.write_text(yaml.safe_dump(_panels_dict()))
    doc = cr.load_panels_yaml(p)
    assert doc["project"] == "abd3"
    assert len(doc["sections"]) == 2


def test_load_panels_yaml_rejects_non_mapping_root(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just a list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        cr.load_panels_yaml(p)


def test_load_panels_yaml_rejects_missing_sections(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("project: abd3\n")
    with pytest.raises(ValueError, match="`sections`"):
        cr.load_panels_yaml(p)


def test_load_panels_yaml_rejects_section_without_panels(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("project: x\nsections:\n  - {name: broken}\n")
    with pytest.raises(ValueError, match=r"section\[0\]"):
        cr.load_panels_yaml(p)


def test_summarize_panels_lists_every_panel():
    summary = cr.summarize_panels(_panels_dict())
    assert "Training" in summary
    assert "Quality" in summary
    assert "MAUVE" in summary
    assert "Loss" in summary


def test_apply_panels_dry_run_is_deterministic():
    out = cr.apply_panels(_panels_dict(), dry_run=True)
    assert out.startswith("(dry-run)")


def test_default_panels_file_is_valid():
    default = _ROOT / "configs" / "wandb_panels" / "default.yaml"
    if not default.exists():
        pytest.skip("default panels yaml missing; run the infra task first.")
    doc = cr.load_panels_yaml(default)
    assert doc["project"] == "abd3"
    assert any("Training" in s.get("name", "") for s in doc["sections"])


# ---------------------------------------------------------------------------
# CLI (fetch_runs mocked)
# ---------------------------------------------------------------------------


def test_cli_fetches_runs_and_writes_tables(tmp_path, monkeypatch, capsys):
    fake_runs = [
        _fake_run(
            id="r0",
            name="rdr-run",
            state="finished",
            tags=["rdr"],
            summary={"val/loss": 1.1, "val/ppl": 3.0},
            config={"algo": {"name": "abd3"}, "block_size": 4},
        ),
        _fake_run(
            id="r1",
            name="baseline-run",
            state="finished",
            tags=["baseline"],
            summary={"val/loss": 1.5, "val/ppl": 4.5},
            config={"algo": {"name": "baseline"}, "block_size": 4},
        ),
    ]

    def fake_fetch(**kw):
        return fake_runs

    monkeypatch.setattr(cr, "fetch_runs", fake_fetch)

    csv_out = tmp_path / "o.csv"
    md_out = tmp_path / "o.md"
    rc = cr.main(
        [
            "--entity",
            "arrdel",
            "--project",
            "abd3",
            "--metrics",
            "val/loss",
            "val/ppl",
            "--extra-config-keys",
            "config.algo.name",
            "--group-by",
            "config.algo.name",
            "--csv-out",
            str(csv_out),
            "--markdown-out",
            str(md_out),
        ]
    )
    assert rc == 0
    assert csv_out.exists()
    assert md_out.exists()
    md_text = md_out.read_text()
    # Group-by was applied → one row per algo, plus header.
    assert "abd3" in md_text and "baseline" in md_text
    out_blob = capsys.readouterr().out
    # CLI prints a pretty-printed JSON summary at the tail; grab everything
    # from the first `{` to the last `}` and parse it.
    start = out_blob.index("{")
    end = out_blob.rindex("}") + 1
    parsed = json.loads(out_blob[start:end])
    assert parsed["n_rows"] == 2


def test_cli_apply_panels_path_exits_cleanly(tmp_path, capsys):
    import yaml

    p = tmp_path / "panels.yaml"
    p.write_text(yaml.safe_dump(_panels_dict()))
    rc = cr.main(["--apply-panels", str(p), "--dry-run-panels"])
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
