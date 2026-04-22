"""Unified quality-evaluation runner.

Given a JSONL of generated samples (from :mod:`eval.generate`), compute
*all* quality metrics the project cares about and dump a single combined
report:

* **Diversity** (always-on, no extra deps): self-BLEU, distinct-n,
  repetition ratio. See :mod:`eval.diversity`.
* **Generation PPL** (opt-out via ``--no-gen-ppl``): perplexity of the
  generations under a frozen GPT-2. See :mod:`eval.gen_ppl`.
* **MAUVE** (opt-in via ``--mauve-refs-*``): distributional divergence
  between generated and reference texts. See :mod:`eval.mauve`.

The runner degrades gracefully: a failure in any one metric is logged
and flagged in the output, but the other metrics still run. This matters
in practice — MAUVE crashes if the ``mauve-text`` pip is missing, GPT-2
download fails behind firewalls, etc. We never want one broken dep to
wipe out a 30-minute sampling job's report.

Output schema (JSON)::

    {
      "samples_path": "...",
      "n_records": 512,
      "diversity": { "report": {...}, "error": null },
      "gen_ppl":  { "report": {...}, "error": null },
      "mauve":    { "report": {...}, "error": null }    // or null if not run
    }

CLI example::

    python -m eval.quality \\
        --samples report/samples/feasibility_step4000.jsonl \\
        --gen-ppl-scorer gpt2 --device cuda \\
        --mauve-refs-hf "wikitext|wikitext-103-raw-v1|validation|text" \\
        --mauve-max-refs 1000 \\
        --json-out report/quality/feasibility_step4000.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import traceback
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import diversity as diversity_mod  # noqa: E402
from eval import gen_ppl as gen_ppl_mod  # noqa: E402
from eval import mauve as mauve_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Section runners (each returns (report_dict_or_None, error_str_or_None))
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Section:
    """Small wrapper so the JSON output is uniform across metrics."""

    name: str
    report: dict | None
    error: str | None

    def as_dict(self) -> dict:
        return {"report": self.report, "error": self.error}


def _run_diversity(records: list[dict], source: str, max_refs: int | None, seed: int) -> Section:
    try:
        if source == "token_ids":
            token_sequences = [list(r["token_ids"]) for r in records if "token_ids" in r]
            if not token_sequences:
                # Fall back without surfacing an error — common in CLI pipelines
                # where someone passed a JSONL that only has `text`.
                source = "text"
        if source == "text":
            token_sequences = [
                diversity_mod._whitespace_tokenize(r.get("text", "")) for r in records
            ]
        rep = diversity_mod.compute_diversity(
            token_sequences=token_sequences,
            max_refs=max_refs,
            seed=seed,
            tokenization_label=source,
        )
        return Section("diversity", rep.as_dict(), None)
    except Exception as e:  # noqa: BLE001
        return Section("diversity", None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _run_gen_ppl(
    records: list[dict], scorer: str, device: str, batch_size: int, max_length: int, min_tokens: int
) -> Section:
    try:
        texts = [r.get("text", "") for r in records if r.get("text", "").strip()]
        if not texts:
            return Section("gen_ppl", None, "no non-empty text fields to score")
        rep, per_sample = gen_ppl_mod.compute_gen_ppl(
            texts,
            scorer_name=scorer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            min_tokens=min_tokens,
        )
        # Keep per_sample rows on the section so callers can post-process; we
        # keep it off the summary to avoid multi-megabyte reports.
        payload = rep.as_dict() | {"_per_sample_available": True}
        section = Section("gen_ppl", payload, None)
        section._per_sample = per_sample  # type: ignore[attr-defined]
        return section
    except Exception as e:  # noqa: BLE001
        return Section("gen_ppl", None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def _run_mauve(
    records: list[dict],
    refs_source: dict[str, Any],
    featurize_model: str,
    device: str,
    max_samples: int,
    max_refs: int,
    num_buckets: int | str,
    scaling_factor: float,
    max_text_length: int,
    seed: int,
) -> Section:
    try:
        gen_texts = [r.get("text", "") for r in records if r.get("text", "").strip()]
        if max_samples and len(gen_texts) > max_samples:
            gen_texts = gen_texts[:max_samples]
        if not gen_texts:
            return Section("mauve", None, "no non-empty generations to score")

        if "file" in refs_source:
            refs = mauve_mod.load_refs_from_file(refs_source["file"], max_refs=max_refs)
        elif "hf" in refs_source:
            refs = mauve_mod.load_refs_from_hf(refs_source["hf"], max_refs=max_refs)
        else:
            return Section("mauve", None, "no reference source configured")
        if not refs:
            return Section("mauve", None, "reference source produced no usable rows")

        import os

        device_id = (
            int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            if device == "cuda"
            else -1
        )

        rep = mauve_mod.compute_mauve(
            gen_texts,
            refs,
            featurize_model=featurize_model,
            device_id=device_id,
            max_text_length=max_text_length,
            scaling_factor=scaling_factor,
            num_buckets=num_buckets,
            seed=seed,
        )
        return Section("mauve", rep.as_dict(), None)
    except Exception as e:  # noqa: BLE001
        return Section("mauve", None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QualityReport:
    """Roll-up of all section reports. Flat enough to ``json.dumps`` directly."""

    samples_path: str
    n_records: int
    diversity: Section
    gen_ppl: Section | None
    mauve: Section | None

    def as_dict(self) -> dict:
        return {
            "samples_path": self.samples_path,
            "n_records": self.n_records,
            "diversity": self.diversity.as_dict(),
            "gen_ppl": self.gen_ppl.as_dict() if self.gen_ppl else None,
            "mauve": self.mauve.as_dict() if self.mauve else None,
        }


def run_quality(
    samples_path: str | pathlib.Path,
    *,
    diversity_source: str = "token_ids",
    diversity_max_refs: int | None = 100,
    diversity_seed: int = 42,
    run_gen_ppl: bool = True,
    gen_ppl_scorer: str = "gpt2",
    gen_ppl_device: str = "cpu",
    gen_ppl_batch_size: int = 4,
    gen_ppl_max_length: int = 1024,
    gen_ppl_min_tokens: int = 2,
    mauve_refs_file: str | None = None,
    mauve_refs_hf: str | None = None,
    mauve_max_samples: int = 1000,
    mauve_max_refs: int = 1000,
    mauve_featurize_model: str = "gpt2-large",
    mauve_device: str = "cpu",
    mauve_num_buckets: int | str = "auto",
    mauve_scaling_factor: float = 5.0,
    mauve_max_text_length: int = 1024,
    mauve_seed: int = 42,
) -> QualityReport:
    """Programmatic entry point — useful from notebooks and the paper-writer."""
    records = diversity_mod.load_samples(samples_path)
    n = len(records)

    diversity_section = _run_diversity(
        records,
        diversity_source,
        diversity_max_refs,
        diversity_seed,
    )

    gen_ppl_section: Section | None = None
    if run_gen_ppl:
        gen_ppl_section = _run_gen_ppl(
            records,
            gen_ppl_scorer,
            gen_ppl_device,
            gen_ppl_batch_size,
            gen_ppl_max_length,
            gen_ppl_min_tokens,
        )

    mauve_section: Section | None = None
    if mauve_refs_file or mauve_refs_hf:
        refs_source: dict[str, Any] = (
            {"file": mauve_refs_file} if mauve_refs_file else {"hf": mauve_refs_hf}
        )
        mauve_section = _run_mauve(
            records,
            refs_source,
            mauve_featurize_model,
            mauve_device,
            mauve_max_samples,
            mauve_max_refs,
            mauve_num_buckets,
            mauve_scaling_factor,
            mauve_max_text_length,
            mauve_seed,
        )

    return QualityReport(
        samples_path=str(samples_path),
        n_records=n,
        diversity=diversity_section,
        gen_ppl=gen_ppl_section,
        mauve=mauve_section,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("Run diversity + Gen-PPL + optional MAUVE on a SampleRecord JSONL."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--samples", required=True)
    ap.add_argument("--json-out", default=None)

    # Diversity
    ap.add_argument("--diversity-source", choices=["text", "token_ids"], default="token_ids")
    ap.add_argument("--diversity-max-refs", type=int, default=100)
    ap.add_argument("--diversity-seed", type=int, default=42)

    # Gen-PPL
    ap.add_argument("--no-gen-ppl", action="store_true", help="Skip the GPT-2 fluency scorer.")
    ap.add_argument("--gen-ppl-scorer", default="gpt2")
    ap.add_argument("--gen-ppl-batch-size", type=int, default=4)
    ap.add_argument("--gen-ppl-max-length", type=int, default=1024)
    ap.add_argument("--gen-ppl-min-tokens", type=int, default=2)

    # MAUVE (opt-in)
    refs = ap.add_mutually_exclusive_group()
    refs.add_argument("--mauve-refs-file", default=None)
    refs.add_argument("--mauve-refs-hf", default=None)
    ap.add_argument("--mauve-max-samples", type=int, default=1000)
    ap.add_argument("--mauve-max-refs", type=int, default=1000)
    ap.add_argument("--mauve-featurize-model", default="gpt2-large")
    ap.add_argument("--mauve-num-buckets", default="auto")
    ap.add_argument("--mauve-scaling-factor", type=float, default=5.0)
    ap.add_argument("--mauve-max-text-length", type=int, default=1024)
    ap.add_argument("--mauve-seed", type=int, default=42)

    # Shared
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return ap


def _print_section_summary(section: Section) -> None:
    name = section.name
    if section.error:
        # Truncate the traceback in the stdout summary — the full string is in the JSON.
        first_line = section.error.splitlines()[0]
        print(f"  [{name}] ERROR: {first_line}")
        return
    rep = section.report or {}
    if name == "diversity":
        print(
            f"  [diversity] dist-1={rep.get('distinct_1', 0):.3f}  "
            f"dist-4={rep.get('distinct_4', 0):.3f}  "
            f"self-BLEU-4={rep.get('self_bleu_4', 0):.3f}  "
            f"rep={rep.get('repetition_ratio', 0):.3f}"
        )
    elif name == "gen_ppl":
        print(
            f"  [gen_ppl]   corpus-PPL={rep.get('corpus_ppl', float('nan')):.2f}  "
            f"median-PPL={rep.get('median_sample_ppl', float('nan')):.2f}  "
            f"(scorer={rep.get('scorer')})"
        )
    elif name == "mauve":
        print(
            f"  [mauve]     MAUVE={rep.get('mauve', float('nan')):.4f}  "
            f"FI={rep.get('frontier_integral', float('nan')):.4f}  "
            f"(K={rep.get('cluster_k')})"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    report = run_quality(
        args.samples,
        diversity_source=args.diversity_source,
        diversity_max_refs=(args.diversity_max_refs if args.diversity_max_refs > 0 else None),
        diversity_seed=args.diversity_seed,
        run_gen_ppl=not args.no_gen_ppl,
        gen_ppl_scorer=args.gen_ppl_scorer,
        gen_ppl_device=args.device,
        gen_ppl_batch_size=args.gen_ppl_batch_size,
        gen_ppl_max_length=args.gen_ppl_max_length,
        gen_ppl_min_tokens=args.gen_ppl_min_tokens,
        mauve_refs_file=args.mauve_refs_file,
        mauve_refs_hf=args.mauve_refs_hf,
        mauve_max_samples=args.mauve_max_samples,
        mauve_max_refs=args.mauve_max_refs,
        mauve_featurize_model=args.mauve_featurize_model,
        mauve_device=args.device,
        mauve_num_buckets=(
            int(args.mauve_num_buckets)
            if args.mauve_num_buckets.isdigit()
            else args.mauve_num_buckets
        ),
        mauve_scaling_factor=args.mauve_scaling_factor,
        mauve_max_text_length=args.mauve_max_text_length,
        mauve_seed=args.mauve_seed,
    )

    print("=" * 100)
    print(f"quality report for {args.samples}  (n={report.n_records})")
    print("=" * 100)
    _print_section_summary(report.diversity)
    if report.gen_ppl is not None:
        _print_section_summary(report.gen_ppl)
    if report.mauve is not None:
        _print_section_summary(report.mauve)
    print("=" * 100)

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2))
        print(f"[quality] wrote {out}")

    # Non-zero exit if every metric errored — useful for CI/cron to flag silent failures.
    errors = [
        s for s in (report.diversity, report.gen_ppl, report.mauve) if s is not None and s.error
    ]
    if errors and len(errors) == sum(
        1 for s in (report.diversity, report.gen_ppl, report.mauve) if s is not None
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
