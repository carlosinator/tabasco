import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
# TODO: chore: clean this file


class PositionalEncoding(ABC, nn.Module):
    """Abstract interface for modules that add positional information to tensors."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Return positional encodings."""
        pass

    @abstractmethod
    def out_dim(self):
        """Embedding dimension produced by this encoder."""
        pass


class SinusoidEncoding(PositionalEncoding):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, posenc_dim, max_len=100, random_permute=False):
        """Initialize the positional encoding.

        Args:
            posenc_dim: Size of the embedding dimension.
            max_len: Maximum sequence length supported. `seq_len` passed to
                `forward` must not exceed this value.
            random_permute: If `True`, the positions are randomly permuted for each
                sample (useful as lightweight data augmentation but destroys absolute
                ordering).
        """
        super().__init__()

        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

        pos_embed = torch.zeros(max_len, self.posenc_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.posenc_dim, 2).float()
            * (-math.log(10 * self.max_len) / self.posenc_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, batch_size: int, seq_len: int):
        """Return positional embeddings of shape `(batch_size, seq_len, posenc_dim)`.

        Args:
            batch_size: Number of samples in the batch.
            seq_len: Length of the sequence.

        Note:
            `seq_len` must not exceed the `max_len` passed at construction.
        """
        pos_embed = self.pos_embed[:, :seq_len, :]
        pos_embed = pos_embed.expand(batch_size, -1, -1)

        if self.random_permute:
            # a fast way to do batched random permutations
            batch_size, seq_len, dim = pos_embed.shape
            perm = torch.argsort(
                torch.rand(batch_size, seq_len, device=pos_embed.device), dim=1
            )
            pos_embed = pos_embed.gather(
                1, perm.unsqueeze(-1).expand(batch_size, seq_len, dim)
            )

        return pos_embed

    def out_dim(self):
        return self.posenc_dim


class TimeFourierEncoding(PositionalEncoding):
    """Encoder for continuous timesteps in `[0, 1]`"""

    def __init__(self, posenc_dim, max_len=100, random_permute=False):
        super().__init__()
        self.posenc_dim = posenc_dim
        self.random_permute = random_permute
        self.max_len = max_len

    def forward(self, t: Tensor):
        """Encode a tensor of timesteps.

        Args:
            t: 1-D tensor with values in `[0, 1]`.

        Returns:
            Tensor of shape `(B, posenc_dim)` with sine/cosine features.
        """
        t_scaled = t * self.max_len
        half_dim = self.posenc_dim // 2
        emb = math.log(self.max_len) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb
        )
        emb = torch.outer(t_scaled.float(), emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.posenc_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")

        assert emb.shape == (t.shape[0], self.posenc_dim), (
            f"Expected shape ({t.shape[0], self.posenc_dim}), got {emb.shape}"
        )
        return emb

    def out_dim(self):
        return self.posenc_dim
