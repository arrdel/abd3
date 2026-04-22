"""
ABD3 Data Loading with mixed block-size support.

Tokenization is memoized to ``data/_tokenized_cache/`` via
:mod:`abd3.tokenization_cache` so re-runs and eval scripts don't
re-tokenize from scratch every time. Set ``config.data.tokenization_cache=false``
to disable, or ``config.data.tokenization_cache_dir`` to point elsewhere.
"""

import datasets
import torch

from abd3.tokenization_cache import tokenize_with_cache

_FILTER_MIN_CHARS = 10


def get_dataloaders(config, tokenizer):
    """Create train/val dataloaders."""
    # Support subset (e.g. wikitext-2-raw-v1)
    subset = getattr(config.data, "subset", None)
    kwargs = {"trust_remote_code": True}
    if subset:
        dataset = datasets.load_dataset(config.data.name, subset, **kwargs)
    else:
        split = getattr(config.data, "split", None)
        if split:
            kwargs["split"] = split
        dataset = datasets.load_dataset(config.data.name, **kwargs)

    seq_len = config.model.length

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

    def _nonempty(x):
        return len(x["text"].strip()) > _FILTER_MIN_CHARS

    use_cache = bool(getattr(config.data, "tokenization_cache", True))
    cache_dir = getattr(config.data, "tokenization_cache_dir", None)
    tokenizer_name = getattr(
        config.data, "tokenizer_name_or_path", getattr(tokenizer, "name_or_path", "unknown")
    )
    split_name = getattr(config.data, "split", None)

    if use_cache:
        tokenized = tokenize_with_cache(
            dataset,
            tokenize_fn,
            dataset_name=config.data.name,
            subset=subset,
            split=split_name,
            tokenizer_name=str(tokenizer_name),
            tokenizer_vocab_size=int(getattr(tokenizer, "vocab_size", 0)),
            seq_len=int(seq_len),
            filter_min_chars=_FILTER_MIN_CHARS,
            cache_dir=cache_dir,
            filter_fn=_nonempty,
            extra_key={"padding": "max_length", "truncation": True},
        )
    else:
        # Preserve the original path for debugging / CI-time validation.
        if isinstance(dataset, datasets.DatasetDict):
            for sname in dataset:
                dataset[sname] = dataset[sname].filter(_nonempty)
            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset[list(dataset.keys())[0]].column_names,
                num_proc=1,
            )
        else:
            dataset = dataset.filter(_nonempty)
            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=1,
            )

    if isinstance(tokenized, datasets.DatasetDict):
        train_ds = tokenized["train"]
        val_ds = tokenized.get("validation", tokenized.get("test", tokenized["train"]))
    else:
        split = tokenized.train_test_split(test_size=0.05, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.loader.eval_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: {config.data.name} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_dl, val_dl
