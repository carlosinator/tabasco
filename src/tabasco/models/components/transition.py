import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type


class FeedForward(nn.Module):
    """
    Feed Forward Network with optional activation and dropout.

    This is a standard feed forward network used in transformer architectures,
    with a configurable activation function and dropout rate.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim: Input and output dimension
            hidden_dim: Hidden dimension (typically 4x the input dimension)
            dropout: Dropout probability
            activation: Activation function to use
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        return self.net(x)


class Transition(nn.Module):
    """
    Modern Transition MLP block with SwiGLU or other activation variants.

    This implements a more modern version of the feed forward network used in
    transformers, with options for different activation functions including SwiGLU
    which is used in models like PaLM and LLaMA.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation_type: str = "swiglu",
        layer_norm: bool = True,
    ):
        """
        Initialize the Transition module.

        Args:
            dim: Input and output dimension
            hidden_dim: Hidden dimension (defaults to 4x input dim)
            dropout: Dropout probability
            activation_type: Type of activation to use ('swiglu', 'geglu', 'gelu', 'relu', 'silu')
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.activation_type = activation_type

        if activation_type == "swiglu":
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        elif activation_type == "geglu":
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, dim, bias=False)

            if activation_type == "gelu":
                self.act = F.gelu
            elif activation_type == "relu":
                self.act = F.relu
            elif activation_type == "silu":
                self.act = F.silu
            else:
                raise ValueError(f"Unsupported activation type: {activation_type}")

        self.dropout = nn.Dropout(dropout)

        if layer_norm:
            self.norm = nn.LayerNorm(dim, eps=1e-5)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        if self.norm is not None:
            x = self.norm(x)

        if self.activation_type == "swiglu":
            x1 = self.w1(x)
            x2 = self.w2(x)
            hidden = F.silu(x1) * x2
        elif self.activation_type == "geglu":
            x1 = self.w1(x)
            x2 = self.w2(x)
            hidden = F.gelu(x1) * x2
        else:
            hidden = self.act(self.w1(x))

        output = self.w3(hidden)
        return self.dropout(output)
