"""Tests for eval/diversity.py.

Locks down the numerical contract of every public helper so refactors can't
silently shift self-BLEU / distinct-n numbers — those appear in the paper's
tables and even ~0.01 drift on a hand-picked corpus is suspicious. All
values below were recomputed by pen/paper on the tiny corpora used here.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import diversity as div  # noqa: E402

# ---------------------------------------------------------------------------
# n-gram helpers
# ---------------------------------------------------------------------------


def test_ngrams_returns_empty_when_sequence_shorter_than_n():
    assert div._ngrams([1, 2], 3) == []
    assert div._ngrams([], 1) == []


def test_ngrams_slides_over_sequence():
    assert div._ngrams([1, 2, 3, 4], 2) == [(1, 2), (2, 3), (3, 4)]
    assert div._ngrams(list("abc"), 3) == [("a", "b", "c")]


def test_whitespace_tokenize_splits_and_strips():
    assert div._whitespace_tokenize("  hello   world  ") == ["hello", "world"]
    assert div._whitespace_tokenize("") == []


# ---------------------------------------------------------------------------
# distinct-n
# ---------------------------------------------------------------------------


def test_distinct_n_everything_unique_returns_one():
    seqs = [[1, 2, 3], [4, 5, 6]]
    # 1-grams: {1,2,3,4,5,6}/6 = 1.0; 2-grams: 4 unique / 4 total = 1.0
    assert div.distinct_n(seqs, 1) == 1.0
    assert div.distinct_n(seqs, 2) == 1.0


def test_distinct_n_fully_repeated_returns_min_fraction():
    # 4 copies of [1,1,1,1] -> 1-gram "1" appears 16 times; unique=1 -> 1/16
    seqs = [[1, 1, 1, 1]] * 4
    assert div.distinct_n(seqs, 1) == pytest.approx(1 / 16)


def test_distinct_n_returns_zero_when_no_samples_long_enough():
    # asking for 4-grams on samples of length 2
    assert div.distinct_n([[1, 2], [3, 4]], 4) == 0.0


# ---------------------------------------------------------------------------
# Self-BLEU
# ---------------------------------------------------------------------------


def test_self_bleu_is_high_when_all_samples_identical():
    """Identical corpus should have self-BLEU ≈ 1.0 (perfect match)."""
    seqs = [list("the quick brown fox jumps over".split())] * 5
    bleu = div.self_bleu(seqs, max_n=4, max_refs=None)
    assert bleu > 0.95, f"expected ≈1.0 on identical corpus, got {bleu}"


def test_self_bleu_is_low_when_samples_fully_disjoint():
    """Disjoint vocab across samples → all n-gram precisions are ~0 → BLEU tiny."""
    seqs = [
        list("aaa bbb ccc ddd eee fff".split()),
        list("ggg hhh iii jjj kkk lll".split()),
        list("mmm nnn ooo ppp qqq rrr".split()),
    ]
    bleu = div.self_bleu(seqs, max_n=4, max_refs=None)
    assert bleu < 1e-6, f"expected ≈0 on disjoint corpus, got {bleu}"


def test_self_bleu_returns_zero_for_single_sample():
    """Nothing to compare against: must gracefully return 0, not crash."""
    assert div.self_bleu([[1, 2, 3, 4, 5]], max_n=4) == 0.0


def test_self_bleu_is_deterministic_with_seed():
    """Same seed → same subsample of refs → identical score."""
    seqs = [[i, i + 1, i + 2, i + 3, i + 4] for i in range(30)]
    a = div.self_bleu(seqs, max_n=4, max_refs=5, seed=7)
    b = div.self_bleu(seqs, max_n=4, max_refs=5, seed=7)
    assert a == b


def test_self_bleu_changes_with_max_refs_subsample():
    """Distinct max_refs subsamples should produce a different mean score
    on a corpus where the first few samples are much more repetitive
    than the rest — guards against a bug where max_refs is silently ignored."""
    # First 5 samples identical, last 25 all disjoint.
    common = list("one two three four five".split())
    seqs = [common] * 5
    for i in range(25):
        seqs.append([f"w{i}{j}" for j in range(5)])

    low = div.self_bleu(seqs, max_n=4, max_refs=1, seed=0)
    full = div.self_bleu(seqs, max_n=4, max_refs=None, seed=0)
    # They should differ; which is larger depends on the random draw.
    assert low != full


# ---------------------------------------------------------------------------
# Brevity penalty / precision helpers
# ---------------------------------------------------------------------------


def test_brevity_penalty_is_one_when_hyp_longer():
    assert div._brevity_penalty(hyp_len=10, ref_lens=[5, 6, 7]) == 1.0


def test_brevity_penalty_penalises_short_hyps():
    bp = div._brevity_penalty(hyp_len=3, ref_lens=[10])
    assert 0 < bp < 1


def test_modified_precision_clips_to_max_ref_count():
    """'the' appears 3× in hyp but max 2× in any ref → clipped to 2/3."""
    hyp = ["the", "the", "the"]
    refs = [["the", "the", "cat"], ["the", "dog"]]
    p = div._modified_precision(hyp, refs, n=1)
    assert p == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Repetition ratio
# ---------------------------------------------------------------------------


def test_repetition_ratio_all_different_is_zero():
    assert div.repetition_ratio([[1, 2, 3, 4]]) == 0.0


def test_repetition_ratio_all_same_is_one():
    assert div.repetition_ratio([[5, 5, 5, 5, 5]]) == 1.0


def test_repetition_ratio_mixed():
    # [1,1,2,2,3] -> pairs: (1,1), (1,2), (2,2), (2,3) -> 2/4 = 0.5
    assert div.repetition_ratio([[1, 1, 2, 2, 3]]) == 0.5


def test_repetition_ratio_ignores_too_short_samples():
    # Samples of length 0 or 1 contribute nothing; average is over the valid rest.
    assert div.repetition_ratio([[], [7], [1, 1, 1]]) == 1.0


# ---------------------------------------------------------------------------
# compute_diversity top-level
# ---------------------------------------------------------------------------


def test_compute_diversity_from_texts_populates_every_field():
    texts = [
        "the cat sat on the mat",
        "a quick brown fox jumped over the log",
        "diffusion models can generate diverse text",
    ]
    report = div.compute_diversity(texts)
    assert report.n_samples == 3
    assert 0.0 < report.distinct_1 <= 1.0
    assert 0.0 < report.distinct_2 <= 1.0
    assert 0.0 <= report.self_bleu_3 <= 1.0
    assert 0.0 <= report.self_bleu_4 <= 1.0
    assert report.tokenization == "whitespace"


def test_compute_diversity_requires_some_input():
    with pytest.raises(ValueError, match="Provide either"):
        div.compute_diversity()


def test_compute_diversity_rejects_empty_corpus():
    with pytest.raises(ValueError, match="Empty corpus"):
        div.compute_diversity(texts=[])


def test_compute_diversity_accepts_token_sequences():
    seqs = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 9, 10]]
    report = div.compute_diversity(token_sequences=seqs, tokenization_label="subword")
    assert report.n_samples == 3
    assert report.tokenization == "subword"


def test_report_summary_line_contains_all_metric_prefixes():
    report = div.compute_diversity(
        token_sequences=[[1, 2, 3], [4, 5, 6]], tokenization_label="test"
    )
    line = report.summary_line()
    for marker in ["n=", "dist-1=", "dist-4=", "self-BLEU-4=", "rep="]:
        assert marker in line


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def test_load_samples_reads_records_and_skips_blank_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"sample_id": 0, "text": "hi"})
        + "\n"
        + "\n"  # blank line → skipped
        + json.dumps({"sample_id": 1, "text": "there"})
        + "\n"
    )
    records = div.load_samples(path)
    assert [r["sample_id"] for r in records] == [0, 1]


def test_load_samples_errors_on_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": 0}) + "\n{not_json}\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        div.load_samples(path)


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_writes_json_output(tmp_path, capsys):
    samples_path = tmp_path / "samples.jsonl"
    records = [
        {"sample_id": i, "text": f"alpha beta gamma {i}", "token_ids": [1, 2, 3, i + 10]}
        for i in range(4)
    ]
    with samples_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out_path = tmp_path / "report.json"
    rc = div.main(
        [
            "--samples",
            str(samples_path),
            "--source",
            "token_ids",
            "--max-refs",
            "0",
            "--json-out",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["source"] == "token_ids"
    assert payload["report"]["n_samples"] == 4

    captured = capsys.readouterr()
    assert "self-BLEU-4" in captured.out


def test_cli_returns_1_on_empty_jsonl(tmp_path):
    samples_path = tmp_path / "empty.jsonl"
    samples_path.write_text("")
    rc = div.main(["--samples", str(samples_path)])
    assert rc == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
