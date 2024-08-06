"""
This module implements the core transformer blocks: Encoder and Decoder.
It includes the EncoderBlock, Encoder, DecoderBlock, and Decoder classes.
"""

import torch
import torch.nn as nn
from typing import List, Callable

from .attention import MultiHeadAttentionBlock
from .utils import LayerNormalization, FeedForwardBlock, ResidualConnection


class EncoderBlock(nn.Module):
    """
    Implements a single block of the transformer encoder.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float):
        """
        Initialize the EncoderBlock.

        Args:
            features (int): Number of input features.
            self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention mechanism.
            feed_forward_block (FeedForwardBlock): Feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(6)])


    def forward(self, src_seq: torch.Tensor, scaffold: torch.Tensor, properties: torch.Tensor, 
                src_mask: torch.Tensor, scaffold_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the EncoderBlock.

        Args:
            src_seq (torch.Tensor): Source sequence.
            scaffold (torch.Tensor): Scaffold tensor.
            properties (torch.Tensor): Properties tensor.
            src_mask (torch.Tensor): Source mask.
            scaffold_mask (torch.Tensor): Scaffold mask.

        Returns:
            torch.Tensor: Output of the encoder block.
        """
        src_seq = self.residual_connections[0](src_seq, lambda x: self.self_attention_block(x, x, x, src_mask))
        src_seq = self.residual_connections[1](src_seq, self.feed_forward_block)
        src_seq = self.residual_connections[2](src_seq, lambda x: self.cross_attention_block(x, scaffold, scaffold, scaffold_mask))
        src_seq = self.residual_connections[3](src_seq, self.feed_forward_block)
        src_seq = self.residual_connections[4](src_seq, lambda x: self.cross_attention_block(x, properties, properties, None))
        src_seq = self.residual_connections[5](src_seq, self.feed_forward_block)
        return src_seq


class Encoder(nn.Module):
    """
    Implements the full encoder of the transformer.
    """

    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Initialize the Encoder.

        Args:
            features (int): Number of input features.
            layers (nn.ModuleList): List of EncoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)


    def forward(self, src_seq: torch.Tensor, scaffold: torch.Tensor, properties: torch.Tensor, 
                src_mask: torch.Tensor, scaffold_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Encoder.

        Args:
            src_seq (torch.Tensor): Source sequence.
            scaffold (torch.Tensor): Scaffold tensor.
            properties (torch.Tensor): Properties tensor.
            src_mask (torch.Tensor): Source mask.
            scaffold_mask (torch.Tensor): Scaffold mask.

        Returns:
            torch.Tensor: Output of the encoder.
        """
        for layer in self.layers:
            src_seq = layer(src_seq, scaffold, properties, src_mask, scaffold_mask)
        return self.norm(src_seq)


class DecoderBlock(nn.Module):
    """
    Implements a single block of the transformer decoder.
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float):
        """
        Initialize the DecoderBlock.

        Args:
            features (int): Number of input features.
            self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention mechanism.
            feed_forward_block (FeedForwardBlock): Feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(4)])


    def forward(self, tgt_seq: torch.Tensor, encoder_output: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the DecoderBlock.

        Args:
            tgt_seq (torch.Tensor): Target sequence.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Output of the decoder block.
        """
        tgt_seq = self.residual_connections[0](tgt_seq, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        tgt_seq = self.residual_connections[1](tgt_seq, self.feed_forward_block)
        tgt_seq = self.residual_connections[2](tgt_seq, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        tgt_seq = self.residual_connections[3](tgt_seq, self.feed_forward_block)
        return tgt_seq


class Decoder(nn.Module):
    """
    Implements the full decoder of the transformer.
    """

    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Initialize the Decoder.

        Args:
            features (int): Number of input features.
            layers (nn.ModuleList): List of DecoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)


    def forward(self, tgt_seq: torch.Tensor, encoder_output: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the Decoder.

        Args:
            tgt_seq (torch.Tensor): Target sequence.
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Output of the decoder.
        """
        for layer in self.layers:
            tgt_seq = layer(tgt_seq, encoder_output, src_mask, tgt_mask)
        return self.norm(tgt_seq)