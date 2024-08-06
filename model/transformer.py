"""
This module implements the Transformer model for SMILES generation.
It includes the Transformer class and a function to build the transformer.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple

from .embeddings import InputEmbeddings, PositionalEncoding
from .utils import ProjectionLayer, FeedForwardBlock
from .transformer_blocks import Encoder, EncoderBlock, Decoder, DecoderBlock
from .attention import MultiHeadAttentionBlock


class Transformer(nn.Module):
    """
    Implements the Transformer model for SMILES generation.
    """

    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_vocab_size: int, 
                 d_model: int, 
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer
                ):
        """
        Initialize the Transformer model.

        Args:
            encoder (Encoder): The encoder component.
            decoder (Decoder): The decoder component.
            src_embed (InputEmbeddings): Source embeddings.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimension of the model.
            src_pos (PositionalEncoding): Positional encoding for source.
            tgt_pos (PositionalEncoding): Positional encoding for target.
            projection_layer (ProjectionLayer): Output projection layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer


    def forward(self, 
                src_seq: torch.Tensor, 
                src_scaffold: torch.Tensor, 
                tgt: torch.Tensor, 
                scaffold_mask: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor, 
                mw: torch.Tensor, logp: torch.Tensor, hbd: torch.Tensor, hba: torch.Tensor, tpsa: torch.Tensor, rotatable_bonds: torch.Tensor, qed: torch.Tensor, sa_score: torch.Tensor
                ) -> torch.Tensor:
        """
        Perform the forward pass of the Transformer.

        Args:
            src_seq (torch.Tensor): Source SMILES input tensor.
            src_scaffold (torch.Tensor): Source scaffold input tensor.
            tgt (torch.Tensor): Target sequence tensor.
            scaffold_mask (torch.Tensor): Mask tensor for scaffold input.
            src_mask (torch.Tensor): Mask tensor for SMILES input.
            tgt_mask (torch.Tensor): Mask tensor for target sequence.
            mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score (torch.Tensor): Molecular property inputs.

        Returns:
            torch.Tensor: Output logits from the transformer.
        """
        src_embed, scaffold_embed, property_embeds = self.src_embed(src_seq, src_scaffold, mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score)
        src_embed = self.src_pos(src_embed)
        scaffold_embed = self.src_pos(scaffold_embed)
        properties = torch.cat(property_embeds, dim=1)
        
        encoder_output = self.encoder(src_embed, scaffold_embed, properties, src_mask, scaffold_mask)
        
        tgt = self.tgt_embed(tgt) * math.sqrt(self.src_embed.d_model)
        tgt = self.tgt_pos(tgt)
        
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.projection_layer(decoder_output)
        return output


    def encode(self, 
               scaffold_mask: torch.Tensor, 
               src_mask: torch.Tensor, 
               src_seq: torch.Tensor, 
               scaffolds: torch.Tensor, 
               mw: torch.Tensor, logp: torch.Tensor, hbd: torch.Tensor, hba: torch.Tensor, tpsa: torch.Tensor, rotatable_bonds: torch.Tensor, qed: torch.Tensor, sa_score: torch.Tensor
              ) -> torch.Tensor:
        """
        Encode the input sequence.

        Args:
            scaffold_mask (torch.Tensor): Mask tensor for scaffold input.
            src_mask (torch.Tensor): Mask tensor for SMILES input.
            src_seq (torch.Tensor): Source SMILES input tensor.
            scaffolds (torch.Tensor): Input tensors.
            mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score (torch.Tensor): Molecular property inputs.

        Returns:
            torch.Tensor: Encoded representation.
        """
        src_embed, scaffold_embed, property_embeds = self.src_embed(src_seq, scaffolds, mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score)
        src_embed = self.src_pos(src_embed)
        scaffold_embed = self.src_pos(scaffold_embed)
        properties = torch.cat(property_embeds, dim=1)
        
        return self.encoder(src_embed, scaffold_embed, properties, src_mask, scaffold_mask)


    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode the target sequence given the encoder output.

        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            src_mask (torch.Tensor): Source mask.
            tgt (torch.Tensor): Target sequence.
            tgt_mask (torch.Tensor): Target mask.

        Returns:
        """   
        tgt = self.tgt_embed(tgt) * math.sqrt(self.src_embed.d_model)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the decoder output to vocabulary size.

        Args:
            x (torch.Tensor): Decoder output.

        Returns:
            torch.Tensor: Projected output.
        """
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      scaffold_vocab_size: int, 
                      d_model: int = 512, 
                      N: int = 6, 
                      h: int = 8, 
                      dropout: float = 0.1, 
                      d_ff: int = 2048
                     ) -> Transformer:
    """
    Build a Transformer model.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum source sequence length.
        tgt_seq_len (int): Maximum target sequence length.
        scaffold_vocab_size (int): Size of the scaffold vocabulary.
        d_model (int, optional): Dimension of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder layers. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimension of the feed-forward network. Defaults to 2048.

    Returns:
        Transformer: The built Transformer model.
    """
    src_embed = InputEmbeddings(d_model, src_vocab_size, scaffold_vocab_size, dropout)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_cross_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_vocab_size, d_model, src_pos, tgt_pos, projection_layer)

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer