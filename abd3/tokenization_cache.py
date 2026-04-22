"""On-disk cache for tokenized HF datasets.

Why this exists
---------------
Every eval script (perplexity, zero-shot, soon MAUVE) re-tokenizes the
same WikiText-103 / PTB / LAMBADA splits from scratch. Tokenizing
WT-103 takes ~2 min on a single core; doing it five times in a row is
10 min of pure overhead that also blocks GPU utilization during
interactive dev. This module memoizes the tokenized
:class:`datasets.Dataset` (or :class:`DatasetDict`) to disk using HF's
native ``save_to_disk`` / ``load_from_disk``, keyed by every input
that can change the output.

Cache key policy
----------------
The key is a short SHA-1 of a JSON blob containing:

* dataset name, subset, split (``None`` → ``"_"``)
* tokenizer name (``AutoTokenizer.from_pretrained`` argument)
* tokenizer vocab size (guards against silent model updates on HF hub)
* sequence length
* the minimum-length filter threshold
* any caller-provided ``extra`` dict (for custom preprocessing flavors)

If the tokenizer author bumps vocab on the hub, the key changes and we
re-tokenize. That's the whole point — silent cache corruption is worse
than paying the tokenization cost again.

Layout
------
``{cache_dir}/abd3_{16-char-hex}/`` — a complete HF ``save_to_disk``
directory. You can ``rm -rf`` to invalidate, or pass ``force=True``.

Intended entry points
---------------------
* :func:`tokenize_with_cache` — the thin wrapper, takes a dataset + a
  ``tokenize_fn``, does a ``.map(...)`` only when the cache is cold.
* :func:`build_cache_key` — exposed for tests and for eval scripts that
  want a stable per-config cache path.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from typing import Any

_DEFAULT_CACHE_DIR = pathlib.Path("data") / "_tokenized_cache"


# ---------------------------------------------------------------------------
# Key + path helpers
# ---------------------------------------------------------------------------


def build_cache_key(
    *,
    dataset_name: str,
    subset: str | None,
    split: str | None,
    tokenizer_name: str,
    tokenizer_vocab_size: int,
    seq_len: int,
    filter_min_chars: int,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a short, deterministic cache key for the (dataset, tokenizer) combo.

    Fresh keys are generated for every meaningful axis including
    ``tokenizer_vocab_size`` — that catches the one subtle failure mode
    where someone swaps ``gpt2`` for ``gpt2-xl`` (both named ``"gpt2"``
    in config if they're sloppy) and the cache serves the wrong tensors.
    """
    material: dict[str, Any] = {
        "ds": dataset_name,
        "sub": subset if subset else "_",
        "split": split if split else "_",
        "tok": tokenizer_name,
        "tok_vocab": int(tokenizer_vocab_size),
        "L": int(seq_len),
        "fmin": int(filter_min_chars),
    }
    if extra:
        material["extra"] = {k: extra[k] for k in sorted(extra.keys())}
    blob = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def cache_path_for_key(key: str, cache_dir: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the absolute path for a given cache key."""
    root = pathlib.Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    return root / f"abd3_{key}"


# ---------------------------------------------------------------------------
# Load / save primitives
# ---------------------------------------------------------------------------


def load_if_cached(path: pathlib.Path):
    """Return the cached dataset (or ``DatasetDict``) at ``path``, or ``None``.

    We soft-swallow :class:`FileNotFoundError` and the generic errors HF
    raises on corrupted caches — those translate to a cold miss, which
    triggers a rebuild. The only hard failure is an unexpected
    ``ImportError``, which we surface.
    """
    if not path.exists():
        return None
    try:
        import datasets
    except ImportError:  # pragma: no cover — hard dep at project level
        raise
    try:
        return datasets.load_from_disk(str(path))
    except (FileNotFoundError, OSError, ValueError):
        return None


def save_to_cache(dataset, path: pathlib.Path) -> None:
    """Persist ``dataset`` atomically: write to a sibling, then rename.

    Avoids half-written cache dirs if the save is killed mid-way, which
    would otherwise poison every future load until a user notices.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    dataset.save_to_disk(str(tmp))
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def invalidate_cache(path: pathlib.Path) -> bool:
    """Remove a cache entry. Returns True if something was deleted."""
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Main convenience wrapper
# ---------------------------------------------------------------------------


def tokenize_with_cache(
    dataset,
    tokenize_fn,
    *,
    dataset_name: str,
    subset: str | None,
    split: str | None,
    tokenizer_name: str,
    tokenizer_vocab_size: int,
    seq_len: int,
    filter_min_chars: int,
    cache_dir: str | pathlib.Path | None = None,
    force: bool = False,
    remove_columns: list[str] | None = None,
    filter_fn=None,
    batched: bool = True,
    num_proc: int = 1,
    extra_key: dict[str, Any] | None = None,
    verbose: bool = True,
):
    """Tokenize ``dataset`` with cache memoization.

    Parameters
    ----------
    dataset
        A ``datasets.Dataset`` or ``datasets.DatasetDict`` to tokenize.
    tokenize_fn
        The batched callable passed to ``dataset.map(tokenize_fn, batched=True, …)``.
    filter_fn
        Optional row filter applied *before* tokenization (matches the
        current dataloader's "drop near-empty lines" behavior). Must be
        pure so the cache key stays meaningful.
    force
        Skip the cache lookup and overwrite. Useful for CI that wants
        to validate the tokenization pipeline end to end.
    extra_key
        Extra caller-provided context that should segment the cache
        (e.g. ``{"pack": True}`` for a pack-docs variant).

    Returns
    -------
    A :class:`datasets.Dataset` / :class:`DatasetDict` with the
    tokenized columns — same type as the input.
    """
    key = build_cache_key(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        tokenizer_name=tokenizer_name,
        tokenizer_vocab_size=tokenizer_vocab_size,
        seq_len=seq_len,
        filter_min_chars=filter_min_chars,
        extra=extra_key,
    )
    path = cache_path_for_key(key, cache_dir=cache_dir)

    if not force:
        cached = load_if_cached(path)
        if cached is not None:
            if verbose:
                print(f"[tok-cache] HIT  {path}")
            return cached

    if verbose:
        print(f"[tok-cache] MISS {path}  (tokenizing from scratch)")

    import datasets

    prepared = dataset
    if filter_fn is not None:
        if isinstance(prepared, datasets.DatasetDict):
            for s in list(prepared.keys()):
                prepared[s] = prepared[s].filter(filter_fn)
        else:
            prepared = prepared.filter(filter_fn)

    if isinstance(prepared, datasets.DatasetDict):
        cols = remove_columns or prepared[list(prepared.keys())[0]].column_names
    else:
        cols = remove_columns or prepared.column_names

    tokenized = prepared.map(
        tokenize_fn,
        batched=batched,
        remove_columns=cols,
        num_proc=num_proc,
    )

    save_to_cache(tokenized, path)
    return tokenized
