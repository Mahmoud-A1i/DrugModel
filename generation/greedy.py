"""
This module implements the greedy decode algorithm used in training and generation.
"""

import torch
from tokenizers import Tokenizer
from data import causal_mask


def greedy_decode(model: torch.nn.Module,
                  tokenizer_smiles_tgt: Tokenizer, 
                  max_len: int,
                  device: torch.device,
                  smiles_input: torch.Tensor,
                  scaffold_input: torch.Tensor,
                  smiles_mask: torch.Tensor,
                  scaffold_mask: torch.Tensor,
                  mw: torch.Tensor, logp: torch.Tensor, hbd: torch.Tensor, hba: torch.Tensor, tpsa: torch.Tensor, rotatable_bonds: torch.Tensor, qed: torch.Tensor, sa_score: torch.Tensor
                 ) -> torch.Tensor:
    """
    Perform greedy decoding on the input.

    Args:
        model (torch.nn.Module): The transformer model.
        smiles_input, scaffold_input, smiles_mask, scaffold_mask (torch.Tensor): Input tensors.
        tokenizer_smiles_tgt (Tokenizer): Target SMILES tokenizer.
        max_len (int): Maximum length of the generated sequence.
        device (torch.device): The device to run the model on.
        mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score (torch.Tensor): Molecular property inputs.

    Returns:
        torch.Tensor: The decoded sequence.
    """
    sos_idx = tokenizer_smiles_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_smiles_tgt.token_to_id("[EOS]")
    
    # Encode all inputs
    encoder_output = model.encode(scaffold_mask, smiles_mask, smiles_input, scaffold_input, mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score)
    
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(smiles_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(smiles_mask).to(device)
        out = model.decode(encoder_output, smiles_mask, decoder_input, decoder_mask)
        
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(smiles_input).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)