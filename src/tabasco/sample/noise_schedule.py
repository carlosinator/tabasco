import torch
from abc import ABC, abstractmethod


class BaseNoiseSchedule(ABC):
    """Interface for time-dependent noise scaling."""

    def __init__(self, cutoff: float = 0.9):
        """Args:
        cutoff: Timesteps above this value return zero noise.
        """
        self.cutoff = cutoff

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return per-sample noise multipliers for timesteps `t`."""
        pass


class SampleNoiseSchedule(BaseNoiseSchedule):
    """Inverse schedule: `scale = 1 / (t + eps)`."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute inverse noise scaling with small numerical stabilizer."""
        raw_scale = 1 / (t + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class RatioSampleNoiseSchedule(BaseNoiseSchedule):
    """Ratio schedule: `scale = (1 - t) / (t + eps)`."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Decrease noise roughly linearly while keeping divergence at early steps."""
        raw_scale = (1 - t) / (t + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class SquareSampleNoiseSchedule(BaseNoiseSchedule):
    """Inverse-square schedule: `scale = 1 / (t**2 + eps)`."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Sharper decay than inverse schedule for very early timesteps."""
        raw_scale = 1 / (t**2 + 1e-2)
        return torch.where(t < self.cutoff, raw_scale, torch.zeros_like(t))


class ZeroSampleNoiseSchedule(BaseNoiseSchedule):
    """Schedule that always returns zero (disables additional noise)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Return a tensor of zeros with same shape as `t`."""
        return torch.zeros_like(t)
