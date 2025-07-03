import torch
import torch.nn as nn
from typing import Optional

from tabasco.models.components.attention import AttentionBlock
from tabasco.models.components.transition import Transition


class TransformerBlock(nn.Module):
    """
    A transformer block with layer normalization and residual connections.

    This implements a standard transformer block with self-attention followed by
    a feed-forward network, with layer normalization and residual connections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int = None,
        dropout: float = 0.0,
        activation_type: str = "swiglu",
        norm_eps: float = 1e-5,
    ):
        """
        Initialize the TransformerBlock module.

        Args:
            dim: Input and output dimension
            num_heads: Number of attention heads
            mlp_dim: Hidden dimension for the feed-forward network (defaults to 4x input dim)
            dropout: Dropout probability
            activation_type: Type of activation to use in the feed-forward network
            norm_eps: Epsilon value for layer normalization
        """
        super().__init__()

        # Self-attention block
        self.attn_block = AttentionBlock(
            dim=dim, num_heads=num_heads, dropout=dropout, norm_eps=norm_eps
        )

        # Feed-forward network
        self.ff_block = Transition(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            activation_type=activation_type,
            layer_norm=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        attn_output = self.attn_block(
            x, key_padding_mask=padding_mask, attn_mask=attn_mask
        )
        x = x + attn_output

        # Feed-forward network with residual connection
        ff_output = self.ff_block(x)
        x = x + ff_output

        return x


class Transformer(nn.Module):
    """
    A standard Transformer model with multiple layers.

    This implements a sequence of transformer blocks, each containing
    self-attention and feed-forward components with residual connections.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation_type: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        """
        Initialize the Transformer module.

        Args:
            dim: Model dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_dim: Hidden dimension for feed-forward networks (defaults to 4x dim)
            dropout: Dropout probability
            activation_type: Type of activation to use in feed-forward networks
            norm_eps: Epsilon value for layer normalization
        """
        super().__init__()

        if mlp_dim is None:
            mlp_dim = 4 * dim

        # Create a sequence of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    activation_type=activation_type,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Pass through each transformer block
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, attn_mask=attn_mask)

        # Apply final normalization
        x = self.norm(x)

        return x
