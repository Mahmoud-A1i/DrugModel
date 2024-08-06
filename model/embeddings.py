"""
This module implements the embedding layers for the transformer model.
It includes InputEmbeddings for processing input sequences and molecular properties,
and PositionalEncoding for adding positional information to the embeddings.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple

from .utils import LayerNormalization


class InputEmbeddings(nn.Module):
    """
    Implements the input embeddings for the transformer model.
    
    This class handles embeddings for source sequences, scaffolds, and continuous molecular properties.
    """

    def __init__(self, d_model: int, src_vocab_size: int, scaffold_vocab_size: int, dropout: float):
        """
        Initialize the InputEmbeddings module.

        Args:
            d_model (int): The dimension of the model (embedding size).
            src_vocab_size (int): Size of the source vocabulary.
            scaffold_vocab_size (int): Size of the scaffold vocabulary.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.scaffold_embedding = nn.Embedding(scaffold_vocab_size, d_model)
        
        # Linear layers for continuous molecular properties
        self.property_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(8)  # 8 properties
        ])
        
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, 
                src_seq: torch.Tensor, 
                scaffolds: torch.Tensor, 
                mw: torch.Tensor, logp: torch.Tensor, hbd: torch.Tensor, hba: torch.Tensor, tpsa: torch.Tensor, rotatable_bonds: torch.Tensor, qed: torch.Tensor, sa_score: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Perform the forward pass of the InputEmbeddings module.

        Args:
            src_seq (torch.Tensor): Source sequence tensor.
            scaffolds (torch.Tensor): Scaffold tensor.
            mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score (torch.Tensor): Molecular property tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]: 
                Embedded source sequence, embedded scaffolds, and list of embedded properties.
        """
        src_embed = self.layer_norm(self.dropout(self.src_embedding(src_seq) * math.sqrt(self.d_model)))
        scaffold_embed = self.layer_norm(self.dropout(self.scaffold_embedding(scaffolds) * math.sqrt(self.d_model)))
        
        # Embed continuous properties
        properties = [mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score]
        property_embeds = [self.layer_norm(self.dropout(embed(prop.unsqueeze(-1)))) for embed, prop in zip(self.property_embeddings, properties)]

        # Apply dropout and layer norm to sequence embeddings
        src_embed = self.layer_norm(self.dropout(src_embed))
        scaffold_embed = self.layer_norm(self.dropout(scaffold_embed))
        
        return src_embed, scaffold_embed, property_embeds


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for the transformer model.
    
    This class adds positional information to the input embeddings.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model (embedding size).
            seq_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Input tensor with added positional encoding.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)