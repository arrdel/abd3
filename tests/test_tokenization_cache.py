"""Tests for abd3/tokenization_cache.py.

We exercise the full cache round-trip against a tiny in-memory
``datasets.Dataset`` so there's no network / no HF Hub download. Uses
a fake tokenizer that just returns character-level ids — simple,
deterministic, and enough to prove the cached tensor survives a
save/load cycle unchanged.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from abd3 import tokenization_cache as tc  # noqa: E402

# ---------------------------------------------------------------------------
# build_cache_key
# ---------------------------------------------------------------------------


def test_build_cache_key_is_deterministic():
    k1 = tc.build_cache_key(
        dataset_name="wikitext",
        subset="wikitext-103-raw-v1",
        split="train",
        tokenizer_name="gpt2",
        tokenizer_vocab_size=50257,
        seq_len=512,
        filter_min_chars=10,
    )
    k2 = tc.build_cache_key(
        dataset_name="wikitext",
        subset="wikitext-103-raw-v1",
        split="train",
        tokenizer_name="gpt2",
        tokenizer_vocab_size=50257,
        seq_len=512,
        filter_min_chars=10,
    )
    assert k1 == k2
    assert len(k1) == 16


def test_build_cache_key_changes_for_every_axis():
    base = dict(
        dataset_name="wikitext",
        subset="x",
        split="train",
        tokenizer_name="gpt2",
        tokenizer_vocab_size=50257,
        seq_len=512,
        filter_min_chars=10,
    )
    base_key = tc.build_cache_key(**base)
    variants = [
        {**base, "dataset_name": "ptb"},
        {**base, "subset": "y"},
        {**base, "subset": None},
        {**base, "split": "validation"},
        {**base, "tokenizer_name": "gpt2-medium"},
        {**base, "tokenizer_vocab_size": 50258},
        {**base, "seq_len": 1024},
        {**base, "filter_min_chars": 0},
    ]
    keys = {tc.build_cache_key(**v) for v in variants}
    assert base_key not in keys
    assert len(keys) == len(variants)  # all distinct


def test_build_cache_key_includes_extra_context():
    k0 = tc.build_cache_key(
        dataset_name="d",
        subset=None,
        split=None,
        tokenizer_name="t",
        tokenizer_vocab_size=1,
        seq_len=1,
        filter_min_chars=0,
    )
    k1 = tc.build_cache_key(
        dataset_name="d",
        subset=None,
        split=None,
        tokenizer_name="t",
        tokenizer_vocab_size=1,
        seq_len=1,
        filter_min_chars=0,
        extra={"pack": True},
    )
    assert k0 != k1


def test_build_cache_key_ignores_extra_dict_ordering():
    k0 = tc.build_cache_key(
        dataset_name="d",
        subset=None,
        split=None,
        tokenizer_name="t",
        tokenizer_vocab_size=1,
        seq_len=1,
        filter_min_chars=0,
        extra={"b": 2, "a": 1},
    )
    k1 = tc.build_cache_key(
        dataset_name="d",
        subset=None,
        split=None,
        tokenizer_name="t",
        tokenizer_vocab_size=1,
        seq_len=1,
        filter_min_chars=0,
        extra={"a": 1, "b": 2},
    )
    assert k0 == k1


def test_cache_path_uses_custom_dir(tmp_path):
    p = tc.cache_path_for_key("abcdef1234567890", cache_dir=tmp_path)
    assert p.parent == tmp_path
    assert p.name == "abd3_abcdef1234567890"


def test_cache_path_defaults_under_data_dir():
    p = tc.cache_path_for_key("deadbeefdeadbeef")
    assert p.parts[-2] == "_tokenized_cache"


# ---------------------------------------------------------------------------
# load_if_cached / save_to_cache / invalidate_cache with real HF Datasets
# ---------------------------------------------------------------------------


def _small_dataset():
    import datasets

    return datasets.Dataset.from_dict({"text": [f"line number {i}" for i in range(20)]})


def _fake_tokenize_fn(examples):
    ids = [[ord(c) % 64 for c in t][:8] for t in examples["text"]]
    lens = [len(x) for x in ids]
    pad_to = max(lens) if lens else 0
    padded = [row + [0] * (pad_to - len(row)) for row in ids]
    attn = [[1] * ln + [0] * (pad_to - ln) for ln in lens]
    return {"input_ids": padded, "attention_mask": attn}


def test_load_if_cached_returns_none_for_missing_path(tmp_path):
    assert tc.load_if_cached(tmp_path / "does_not_exist") is None


def test_save_and_load_roundtrip(tmp_path):
    ds = _small_dataset().map(_fake_tokenize_fn, batched=True, remove_columns=["text"])
    path = tmp_path / "abd3_roundtrip"
    tc.save_to_cache(ds, path)
    assert path.exists()

    loaded = tc.load_if_cached(path)
    assert loaded is not None
    assert list(loaded.column_names) == list(ds.column_names)
    assert len(loaded) == len(ds)
    assert loaded[0]["input_ids"] == ds[0]["input_ids"]


def test_save_overwrites_existing(tmp_path):
    path = tmp_path / "abd3_overwrite"
    first = (
        _small_dataset()
        .select(range(5))
        .map(_fake_tokenize_fn, batched=True, remove_columns=["text"])
    )
    second = (
        _small_dataset()
        .select(range(12))
        .map(_fake_tokenize_fn, batched=True, remove_columns=["text"])
    )
    tc.save_to_cache(first, path)
    tc.save_to_cache(second, path)
    loaded = tc.load_if_cached(path)
    assert len(loaded) == 12


def test_invalidate_cache_removes_entry(tmp_path):
    ds = _small_dataset().map(_fake_tokenize_fn, batched=True, remove_columns=["text"])
    path = tmp_path / "abd3_to_kill"
    tc.save_to_cache(ds, path)
    assert tc.invalidate_cache(path) is True
    assert not path.exists()
    # idempotent second call
    assert tc.invalidate_cache(path) is False


def test_load_if_cached_returns_none_on_corrupted_cache(tmp_path):
    # Create an existing-but-invalid directory — HF will refuse to load it.
    bad = tmp_path / "abd3_corrupt"
    bad.mkdir()
    (bad / "not_a_dataset.txt").write_text("corrupted")
    assert tc.load_if_cached(bad) is None


# ---------------------------------------------------------------------------
# tokenize_with_cache full flow
# ---------------------------------------------------------------------------


def test_tokenize_with_cache_hits_on_second_call(tmp_path, capsys):
    ds = _small_dataset()

    kwargs = dict(
        dataset_name="demo",
        subset=None,
        split=None,
        tokenizer_name="fake",
        tokenizer_vocab_size=64,
        seq_len=8,
        filter_min_chars=0,
        cache_dir=tmp_path,
    )
    first = tc.tokenize_with_cache(ds, _fake_tokenize_fn, **kwargs)
    out1 = capsys.readouterr().out
    assert "MISS" in out1

    second = tc.tokenize_with_cache(ds, _fake_tokenize_fn, **kwargs)
    out2 = capsys.readouterr().out
    assert "HIT" in out2

    # Outputs are equivalent.
    assert len(first) == len(second)
    assert first[0]["input_ids"] == second[0]["input_ids"]


def test_tokenize_with_cache_force_rebuild(tmp_path):
    ds = _small_dataset()
    kwargs = dict(
        dataset_name="demo",
        subset=None,
        split=None,
        tokenizer_name="fake",
        tokenizer_vocab_size=64,
        seq_len=8,
        filter_min_chars=0,
        cache_dir=tmp_path,
        verbose=False,
    )
    # Fill the cache
    tc.tokenize_with_cache(ds, _fake_tokenize_fn, **kwargs)
    path = tc.cache_path_for_key(
        tc.build_cache_key(
            dataset_name="demo",
            subset=None,
            split=None,
            tokenizer_name="fake",
            tokenizer_vocab_size=64,
            seq_len=8,
            filter_min_chars=0,
        ),
        cache_dir=tmp_path,
    )
    mtime_before = path.stat().st_mtime

    import time as _t

    _t.sleep(0.01)  # ensure filesystem tick
    tc.tokenize_with_cache(ds, _fake_tokenize_fn, force=True, **kwargs)
    assert path.stat().st_mtime >= mtime_before  # rebuilt


def test_tokenize_with_cache_applies_filter_before_tokenizing(tmp_path):
    import datasets as hfd

    ds = hfd.Dataset.from_dict({"text": ["short", "this is a long enough string"]})

    def _nonempty(x):
        return len(x["text"]) > 10

    kwargs = dict(
        dataset_name="demo",
        subset=None,
        split=None,
        tokenizer_name="fake",
        tokenizer_vocab_size=64,
        seq_len=8,
        filter_min_chars=10,
        cache_dir=tmp_path,
        filter_fn=_nonempty,
        verbose=False,
    )
    out = tc.tokenize_with_cache(ds, _fake_tokenize_fn, **kwargs)
    assert len(out) == 1  # "short" was filtered before tokenization


def test_tokenize_with_cache_key_depends_on_vocab_size(tmp_path):
    """Swapping the tokenizer vocab must produce a fresh cache entry."""
    ds = _small_dataset()
    tc.tokenize_with_cache(
        ds,
        _fake_tokenize_fn,
        dataset_name="demo",
        subset=None,
        split=None,
        tokenizer_name="fake",
        tokenizer_vocab_size=64,
        seq_len=8,
        filter_min_chars=0,
        cache_dir=tmp_path,
        verbose=False,
    )
    tc.tokenize_with_cache(
        ds,
        _fake_tokenize_fn,
        dataset_name="demo",
        subset=None,
        split=None,
        tokenizer_name="fake",
        tokenizer_vocab_size=128,  # <-- changed
        seq_len=8,
        filter_min_chars=0,
        cache_dir=tmp_path,
        verbose=False,
    )
    # Two distinct directories should exist now.
    entries = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("abd3_"))
    assert len(entries) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
