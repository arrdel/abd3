"""
ABD3 EMA (Exponential Moving Average).

Inherited from BD3-LMs.
"""

import copy
import torch


class ExponentialMovingAverage:
    """Maintains exponential moving averages of model parameters."""

    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = None

    def _align_device(self, parameters):
        parameters = list(parameters)
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if s_param.device != param.device:
                self.shadow_params[i] = s_param.to(param.device)
        return parameters

    def update(self, parameters):
        parameters = self._align_device(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, parameters):
        parameters = self._align_device(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [p.clone() for p in parameters]

    def restore(self, parameters):
        if self.collected_params is None:
            raise RuntimeError("No parameters stored for restore")
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        self.collected_params = None
