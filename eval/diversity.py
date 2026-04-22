"""Diversity metrics for generated samples.

Consumes the JSONL produced by :mod:`eval.generate` (one ``SampleRecord`` per
line, with ``text`` and ``token_ids`` fields) and emits three standard
diversity/repetition signals:

distinct-n
    Count of distinct n-grams divided by total n-grams across the whole
    corpus of generations. Higher = more diverse. Reported for n in
    {1, 2, 3, 4} by default.

self-BLEU-n
    For each sample, BLEU-n against a random subset of the other samples
    used as references. Averaged across the corpus. Lower = more diverse
    (higher = generations repeat each other). Reported for n in {3, 4}
    since BLEU-1/2 saturate on small vocabularies.

repetition ratio
    Fraction of consecutive token pairs that are identical (a proxy for
    degenerate "aaaa" or "the the the" outputs). Complements n-gram
    metrics at the local scale.

Everything is pure Python + ``collections.Counter`` so adding this module
doesn't introduce any new runtime dependencies — sacrebleu / nltk are
nice but overkill for a self-contained diversity report.

CLI::

    python -m eval.diversity \\
        --samples report/samples/feasibility_step4000.jsonl \\
        --json-out report/diversity/feasibility_step4000.json

Library::

    from eval.diversity import compute_diversity, load_samples
    recs = load_samples("report/samples/foo.jsonl")
    report = compute_diversity([r["text"] for r in recs])
    print(report.as_dict())
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _whitespace_tokenize(text: str) -> list[str]:
    """Whitespace tokenization.

    The canonical self-BLEU implementation (Zhu et al., Texygen) splits on
    whitespace. We match that here so numbers are directly comparable to
    published results; callers can pass pre-tokenized token id lists if
    they want subword-level diversity instead.
    """
    return text.strip().split()


def _ngrams(tokens: Sequence, n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Distinct-n
# ---------------------------------------------------------------------------


def distinct_n(token_sequences: Sequence[Sequence], n: int) -> float:
    """Return (unique n-grams) / (total n-grams) across the corpus.

    Corpus-level distinct-n (Li et al., 2016). Returns 0.0 when no sample
    is long enough to contain a single n-gram — not NaN, because the
    downstream JSON dashboards should never see NaN from this function.
    """
    total = 0
    unique: set[tuple] = set()
    for toks in token_sequences:
        ngs = _ngrams(toks, n)
        total += len(ngs)
        unique.update(ngs)
    if total == 0:
        return 0.0
    return len(unique) / total


# ---------------------------------------------------------------------------
# Self-BLEU
# ---------------------------------------------------------------------------


def _modified_precision(hyp: Sequence, refs: Sequence[Sequence], n: int) -> float:
    """BLEU modified n-gram precision. 0.0 if hyp has no n-grams of order n."""
    hyp_counts = Counter(_ngrams(hyp, n))
    if not hyp_counts:
        return 0.0

    # Multi-reference max: for each n-gram, take the biggest count any single
    # reference has. This is the standard BLEU clipping rule.
    ref_max: Counter[tuple] = Counter()
    for ref in refs:
        rc = Counter(_ngrams(ref, n))
        for ng, c in rc.items():
            if c > ref_max[ng]:
                ref_max[ng] = c

    overlap = 0
    for ng, c in hyp_counts.items():
        overlap += min(c, ref_max[ng])
    total = sum(hyp_counts.values())
    return overlap / total if total > 0 else 0.0


def _brevity_penalty(hyp_len: int, ref_lens: Iterable[int]) -> float:
    """BLEU brevity penalty using the closest ref length (Papineni et al.)."""
    if hyp_len == 0:
        return 0.0
    # "closest" = minimal absolute diff, tiebroken by shorter ref.
    best_ref = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))
    if hyp_len > best_ref:
        return 1.0
    return math.exp(1.0 - best_ref / hyp_len)


def sentence_bleu(
    hyp: Sequence,
    refs: Sequence[Sequence],
    max_n: int = 4,
    smoothing_eps: float = 1e-9,
) -> float:
    """Compute BLEU-max_n of ``hyp`` against ``refs``.

    Uses additive smoothing (``smoothing_eps``) rather than the harsher
    ``+1`` scheme so a single zero-count high-order n-gram doesn't collapse
    the whole score, which matters for short (< ~20 token) generations.
    """
    if not refs or not hyp:
        return 0.0
    weights = [1.0 / max_n] * max_n
    precisions = [_modified_precision(hyp, refs, n) for n in range(1, max_n + 1)]
    precisions = [p if p > 0 else smoothing_eps for p in precisions]
    log_score = sum(w * math.log(p) for w, p in zip(weights, precisions, strict=True))
    bp = _brevity_penalty(len(hyp), (len(r) for r in refs))
    return bp * math.exp(log_score)


def self_bleu(
    token_sequences: Sequence[Sequence],
    max_n: int = 4,
    max_refs: int = 100,
    seed: int = 42,
) -> float:
    """Mean self-BLEU-max_n across the corpus.

    For each sample we score it against up to ``max_refs`` *other* samples
    chosen uniformly at random. ``max_refs=None`` uses every other sample
    (O(N²); only feasible for small corpora). With ``max_refs=100`` the
    estimator is ~O(N·max_refs) and matches published self-BLEU numbers
    within the Monte-Carlo noise floor.
    """
    n = len(token_sequences)
    if n < 2:
        return 0.0

    rng = random.Random(seed)
    scores: list[float] = []
    all_indices = list(range(n))
    for i, hyp in enumerate(token_sequences):
        others = all_indices[:i] + all_indices[i + 1 :]
        if max_refs is not None and len(others) > max_refs:
            others = rng.sample(others, max_refs)
        refs = [token_sequences[j] for j in others]
        scores.append(sentence_bleu(hyp, refs, max_n=max_n))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Repetition ratio
# ---------------------------------------------------------------------------


def repetition_ratio(token_sequences: Sequence[Sequence]) -> float:
    """Fraction of adjacent-token pairs that repeat, averaged over samples.

    A fast, interpretable proxy for degenerate local repetition ("the the
    the…"). Complements distinct-n at the scale of a single transition.
    """
    ratios = []
    for toks in token_sequences:
        if len(toks) < 2:
            continue
        repeats = sum(1 for a, b in zip(toks[:-1], toks[1:], strict=True) if a == b)
        ratios.append(repeats / (len(toks) - 1))
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DiversityReport:
    """Flat, JSON-serializable container for all diversity signals."""

    n_samples: int
    mean_length: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    distinct_4: float
    self_bleu_3: float
    self_bleu_4: float
    repetition_ratio: float
    tokenization: str  # "whitespace" or "tokenizer:<name>" etc.
    max_refs: int | None
    seed: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        return (
            f"n={self.n_samples}  len={self.mean_length:.1f}  "
            f"dist-1={self.distinct_1:.3f}  dist-2={self.distinct_2:.3f}  "
            f"dist-3={self.distinct_3:.3f}  dist-4={self.distinct_4:.3f}  "
            f"self-BLEU-3={self.self_bleu_3:.3f}  self-BLEU-4={self.self_bleu_4:.3f}  "
            f"rep={self.repetition_ratio:.3f}  "
            f"[{self.tokenization}, max_refs={self.max_refs}]"
        )


def compute_diversity(
    texts: Sequence[str] | None = None,
    *,
    token_sequences: Sequence[Sequence] | None = None,
    max_refs: int | None = 100,
    seed: int = 42,
    tokenization_label: str = "whitespace",
) -> DiversityReport:
    """Compute all diversity metrics from either raw texts or pre-tokenized ids.

    Pass either ``texts`` (whitespace-tokenized here) or pre-tokenized
    ``token_sequences`` (e.g. ``SampleRecord.token_ids``). Passing token ids
    gives subword-level diversity — useful when you want to penalise
    repeated sub-word junk that whitespace tokenization would miss.
    """
    if token_sequences is None:
        if texts is None:
            raise ValueError("Provide either `texts` or `token_sequences`")
        token_sequences = [_whitespace_tokenize(t) for t in texts]
    if len(token_sequences) == 0:
        raise ValueError("Empty corpus — no samples to score")

    lens = [len(t) for t in token_sequences]
    return DiversityReport(
        n_samples=len(token_sequences),
        mean_length=sum(lens) / len(lens),
        distinct_1=distinct_n(token_sequences, 1),
        distinct_2=distinct_n(token_sequences, 2),
        distinct_3=distinct_n(token_sequences, 3),
        distinct_4=distinct_n(token_sequences, 4),
        self_bleu_3=self_bleu(token_sequences, max_n=3, max_refs=max_refs, seed=seed),
        self_bleu_4=self_bleu(token_sequences, max_n=4, max_refs=max_refs, seed=seed),
        repetition_ratio=repetition_ratio(token_sequences),
        tokenization=tokenization_label,
        max_refs=max_refs,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Sample-file loader
# ---------------------------------------------------------------------------


def load_samples(path: str | pathlib.Path) -> list[dict]:
    """Load one ``SampleRecord`` per line from a JSONL produced by ``eval.generate``."""
    records: list[dict] = []
    with pathlib.Path(path).open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: malformed JSON — {e}") from e
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Corpus-level diversity metrics for an eval.generate JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--samples", required=True, help="Path to a SampleRecord JSONL.")
    ap.add_argument(
        "--source",
        choices=["text", "token_ids"],
        default="token_ids",
        help=(
            "Which field to score. ``token_ids`` gives subword-level "
            "diversity (recommended — whitespace on undertrained models is "
            "noisy); ``text`` uses whitespace splits of the decoded text."
        ),
    )
    ap.add_argument(
        "--max-refs",
        type=int,
        default=100,
        help="Self-BLEU ref subset per hyp (None/0 = all other samples).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None, help="Optional path to dump the report.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    records = load_samples(args.samples)
    if not records:
        print(f"[diversity] {args.samples}: empty; nothing to score", file=sys.stderr)
        return 1

    if args.source == "token_ids":
        token_sequences = [list(r["token_ids"]) for r in records if "token_ids" in r]
        if not token_sequences:
            print(
                "[diversity] --source=token_ids but no records have token_ids;"
                " falling back to text",
                file=sys.stderr,
            )
            args.source = "text"
    if args.source == "text":
        token_sequences = [_whitespace_tokenize(r.get("text", "")) for r in records]

    max_refs = args.max_refs if args.max_refs and args.max_refs > 0 else None
    report = compute_diversity(
        token_sequences=token_sequences,
        max_refs=max_refs,
        seed=args.seed,
        tokenization_label=args.source,
    )

    print("=" * 100)
    print(report.summary_line())
    print("=" * 100)

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "samples_path": str(args.samples),
            "source": args.source,
            "report": report.as_dict(),
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"[diversity] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
