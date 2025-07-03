import torch


class HistogramTimeDistribution:
    def __init__(self, probs: torch.Tensor):
        assert probs.ndim == 1, "probs must be 1D"
        assert torch.all(probs >= 0), "probs must be non-negative"
        self.probs = probs / probs.sum()
        self.categorical = torch.distributions.Categorical(probs=self.probs)
        self.uniform = torch.distributions.Uniform(0, 1)
        self.bin_width = 1 / len(probs)

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        # sample offset from uniform
        offset = self.uniform.sample(sample_shape)

        # sample bin index
        bin_idx = self.categorical.sample(sample_shape)

        # compute time
        time = (bin_idx + offset) * self.bin_width

        return time
