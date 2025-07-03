import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x
