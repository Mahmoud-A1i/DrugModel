"""
This module implements the Multi-Head Attention mechanism for transformer models.
It contains the MultiHeadAttentionBlock class, which is a key component in the 
transformer architecture for processing sequential data.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class MultiHeadAttentionBlock(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.

    This block performs parallel attention operations and combines their results.
    It's a key component in transformer architectures for capturing different aspects
    of the input sequence.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Initialize the Multi-Head Attention Block.

        Args:
            d_model (int): The dimension of the model (embedding size).
            h (int): The number of attention heads.
            dropout (float): Dropout rate.
        
        Raises:
            AssertionError: If d_model is not divisible by h.
        """
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Attention mask.
            dropout (nn.Dropout): Dropout layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and attention scores.
        """
        d_k = query.shape[-1]
        
        # Just apply the formula from the paper
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Multi-Head Attention Block.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output of the multi-head attention operation.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape and transpose for multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        return self.w_o(x)