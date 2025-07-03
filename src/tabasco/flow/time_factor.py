import torch
from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseTimeFactor(nn.Module, ABC):
    """Return a scalar weight per sample based on the interpolation time.

    Args:
        max_value: Upper clamp for the time-factor value.
        min_value: Lower clamp for the time-factor value.
        zero_before: If *t* ≤ this threshold the factor is forced to zero.
        eps: Small constant to avoid division by zero.
    """

    def __init__(
        self,
        max_value: float,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.zero_before = zero_before
        self.eps = eps

    @abstractmethod
    def forward(self, t: Tensor) -> Tensor:
        """Return the time-factor for each element in `t`."""
        pass


class InverseTimeFactor(BaseTimeFactor):
    """Weight ~ 1 / (1 - t)^2 as used in the Proteina paper."""

    def __init__(
        self,
        max_value: float = 100.0,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(max_value, min_value, zero_before, eps)

    def forward(self, t: Tensor) -> Tensor:  # noqa: D401
        """Return `1 / (1 - t)^2` clamped to [`min_value`, `max_value`]."""
        norm_scale = 1 / ((1 - t + self.eps) ** 2)
        norm_scale = torch.clamp(norm_scale, min=self.min_value, max=self.max_value)
        return norm_scale * (t > self.zero_before)


class SignalToNoiseTimeFactor(BaseTimeFactor):
    """Weight ~ t / (1 - t) (signal-to-noise ratio)."""

    def __init__(
        self,
        max_value: float = 1.5,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(max_value, min_value, zero_before, eps)

    def forward(self, t: Tensor) -> Tensor:  # noqa: D401
        """Return `t / (1 - t)` clamped to [`min_value`, `max_value`]."""
        clamped = torch.clamp(
            t / (1 - t + self.eps), min=self.min_value, max=self.max_value
        )
        return clamped * (t > self.zero_before)


class SquaredSignalToNoiseTimeFactor(BaseTimeFactor):
    """Weight ~ (t / (1 - t))^2 (squared SNR)."""

    def __init__(
        self,
        max_value: float = 1.5,
        min_value: float = 0.05,
        zero_before: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(max_value, min_value, zero_before, eps)

    def forward(self, t: Tensor) -> Tensor:  # noqa: D401
        """Return `(t / (1 - t))^2` clamped to [`min_value`, `max_value`]."""
        clamped = torch.clamp(
            t / (1 - t + self.eps) ** 2, min=self.min_value, max=self.max_value
        )
        return clamped * (t > self.zero_before)


class SquareTimeFactor(BaseTimeFactor):
    """Weight ~ 1 / (t^2 + 1e-2)"""

    def forward(self, t):
        raw_scale = 1 / (t**2 + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class ZeroTimeFactor(BaseTimeFactor):
    """Weight = 0 for all t"""

    def forward(self, t):
        return torch.zeros_like(t)
