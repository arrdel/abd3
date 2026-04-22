"""ABD3 EMA (Exponential Moving Average).

Originally inherited from BD3-LMs. Extended with ``state_dict`` /
``load_state_dict`` so that EMA shadow params survive a checkpoint round-trip
— without this, ``ABD3Diffusion.load_from_checkpoint`` silently re-initialises
the shadows to the model's random-init weights (because ``shadow_params`` is
not a registered module buffer), producing near-uniform predictions at eval
time.
"""

from __future__ import annotations

import torch


class ExponentialMovingAverage:
    """Maintains exponential moving averages of model parameters.

    Persistence: ``state_dict()`` returns ``{'decay', 'shadow_params'}`` as
    plain tensors on CPU; ``load_state_dict`` is tolerant of the shapes
    already being correct and restores them in-place. ``collected_params``
    (the temporary snapshot used by ``store``/``restore``) is intentionally
    **not** persisted — it only exists during an eval swap.
    """

    def __init__(self, parameters, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------ io
    def state_dict(self) -> dict:
        # .clone() is critical: without it, the returned tensors share storage
        # with the live shadow_params when they already live on CPU, so any
        # downstream code that mutates the serialised dict silently corrupts
        # the in-memory EMA.
        return {
            "decay": self.decay,
            "shadow_params": [p.detach().cpu().clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state: dict) -> None:
        if "shadow_params" not in state:
            raise KeyError("EMA state_dict missing 'shadow_params'")
        incoming = state["shadow_params"]
        if len(incoming) != len(self.shadow_params):
            raise ValueError(
                f"EMA param count mismatch: checkpoint has {len(incoming)}, "
                f"model has {len(self.shadow_params)}. Did the architecture change?"
            )
        for i, (tgt, src) in enumerate(zip(self.shadow_params, incoming, strict=False)):
            if tgt.shape != src.shape:
                raise ValueError(
                    f"EMA shadow #{i} shape mismatch: checkpoint {tuple(src.shape)} "
                    f"vs model {tuple(tgt.shape)}."
                )
            # Preserve device/dtype of the live shadow; just copy values.
            self.shadow_params[i] = src.to(dtype=tgt.dtype, device=tgt.device).clone()
        if "decay" in state:
            self.decay = float(state["decay"])
        # collected_params is a transient store-restore buffer, never persisted.
        self.collected_params = None

    # ---------------------------------------------------------- mechanics
    def _align_device(self, parameters):
        parameters = list(parameters)
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters, strict=False)):
            if s_param.device != param.device:
                self.shadow_params[i] = s_param.to(param.device)
        return parameters

    def update(self, parameters):
        parameters = self._align_device(parameters)
        for s_param, param in zip(self.shadow_params, parameters, strict=False):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, parameters):
        parameters = self._align_device(parameters)
        for s_param, param in zip(self.shadow_params, parameters, strict=False):
            param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [p.clone() for p in parameters]

    def restore(self, parameters):
        if self.collected_params is None:
            raise RuntimeError("No parameters stored for restore")
        for c_param, param in zip(self.collected_params, parameters, strict=False):
            param.data.copy_(c_param.data)
        self.collected_params = None
