"""
This module provides utility classes for the transformer model,
including projection, layer normalization, feed-forward, and residual connection layers.
"""

import torch
import torch.nn as nn
from typing import Callable


class ProjectionLayer(nn.Module):
    """Projection layer to map d_model dimensions to vocabulary size."""

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the ProjectionLayer.

        Args:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ProjectionLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, vocab_size).
        """
        return self.proj(x)


class LayerNormalization(nn.Module):
    """Custom implementation of Layer Normalization."""

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initialize the LayerNormalization.

        Args:
            features (int): The number of features in the input.
            eps (float): A small number to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LayerNormalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Feed-forward block with two linear transformations and a ReLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initialize the FeedForwardBlock.

        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForwardBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    """Residual connection with layer normalization and dropout."""

    def __init__(self, features: int, dropout: float) -> None:
        """
        Initialize the ResidualConnection.

        Args:
            features (int): The number of features in the input.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)


    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the ResidualConnection.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable[[torch.Tensor], torch.Tensor]): A callable representing the sublayer.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))