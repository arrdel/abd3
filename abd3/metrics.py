"""
ABD3 Metrics.

Evaluation metrics for language modeling quality.
"""

import torch
import torchmetrics
from torch import Tensor

LOG2 = torch.log(torch.tensor(2.0))


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        return torch.exp(self.mean_value / self.weight)


class Metrics:
    def __init__(self, config=None):
        self.config = config
        metrics = torchmetrics.MetricCollection({"nll": NLL(), "bpd": BPD(), "ppl": Perplexity()})
        self.block_size = getattr(config, "block_size", config.model.length)
        self.train_nlls = metrics.clone(prefix="train/")
        self.valid_nlls = metrics.clone(prefix="val/")
        self.gen_ppl = Perplexity()

    def to(self, *args, **kwargs):
        self.train_nlls = self.train_nlls.to(*args, **kwargs)
        self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
        self.gen_ppl = self.gen_ppl.to(*args, **kwargs)

    def reset(self):
        self.train_nlls.reset()
        self.valid_nlls.reset()
        self.gen_ppl.reset()
