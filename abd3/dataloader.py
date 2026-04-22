"""
ABD3 Data Loading with mixed block-size support.
"""

import datasets
import torch


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

    if isinstance(dataset, datasets.DatasetDict):
        # Filter empty texts before tokenizing
        for split_name in dataset:
            dataset[split_name] = dataset[split_name].filter(lambda x: len(x["text"].strip()) > 10)

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset[list(dataset.keys())[0]].column_names,
            num_proc=1,
        )
        train_ds = tokenized["train"]
        val_ds = tokenized.get("validation", tokenized.get("test", tokenized["train"]))
    else:
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
        tokenized = dataset.map(
            tokenize_fn, batched=True, remove_columns=dataset.column_names, num_proc=1
        )
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
