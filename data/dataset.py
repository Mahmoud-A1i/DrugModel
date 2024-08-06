"""
This module defines the SMILESDataset class, which is used for processing and preparing
SMILES (Simplified Molecular Input Line Entry System) data for machine learning models.
It handles tokenization, padding, and preparation of input tensors for encoder-decoder
models working with molecular data.
"""

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import Dict, Any, Union, List, Tuple

class SMILESDataset(Dataset):
    """
    Dataset class for SMILES (Simplified Molecular Input Line Entry System) data.
    
    This dataset handles tokenization of SMILES strings and scaffolds, and prepares
    input tensors for encoder-decoder models.
    """

    def __init__(self, ds: List[Dict[str, Any]], 
                 tokenizer_smiles_src: Tokenizer, 
                 tokenizer_smiles_tgt: Tokenizer, 
                 tokenizer_scaffold: Tokenizer, 
                 seq_len: int, 
                 train: bool = True):
        """
        Initialize the SMILESDataset.

        Args:
            ds (List[Dict[str, Any]]): List of dictionaries containing the dataset items.
            tokenizer_smiles_src (Tokenizer): Tokenizer for source SMILES strings.
            tokenizer_smiles_tgt (Tokenizer): Tokenizer for target SMILES strings.
            tokenizer_scaffold (Tokenizer): Tokenizer for scaffold strings.
            seq_len (int): Maximum sequence length for input tensors.
            train (bool): Whether this dataset is for training (True) or inference (False).
        """
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.train = train
        self.tokenizer_smiles_src = tokenizer_smiles_src
        self.tokenizer_smiles_tgt = tokenizer_smiles_tgt
        self.tokenizer_scaffold = tokenizer_scaffold
        self.sos_token = torch.tensor([tokenizer_smiles_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_smiles_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_smiles_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.ds)


    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Union[torch.Tensor, str]]: A dictionary containing the processed item data.

        Raises:
            ValueError: If the sequence is too long for the specified seq_len.
        """
        item = self._preprocess_item(self.ds[idx])
        
        src_smiles = item['input']['SMILES']
        src_scaffold = item['input'].get('Scaffold', '')
        
        encoder_smiles_input, encoder_smiles_mask = self._tokenize_and_pad(
            src_smiles, self.tokenizer_smiles_src)
        
        encoder_scaffold_input, encoder_scaffold_mask = self._tokenize_and_pad(
            src_scaffold, self.tokenizer_scaffold)
        
        properties = self._get_molecular_properties(item['input'])
        
        result = {
            "encoder_smiles_input": encoder_smiles_input,
            "encoder_scaffold_input": encoder_scaffold_input,
            "encoder_smiles_mask": encoder_smiles_mask,
            "encoder_scaffold_mask": encoder_scaffold_mask,
            "src_smiles": src_smiles,
            "src_scaffold": src_scaffold,
            **properties
        }
        
        if self.train:
            tgt_text = item['target']
            decoder_input, label, decoder_mask = self._prepare_decoder_inputs(tgt_text)
            result.update({
                "decoder_input": decoder_input,
                "decoder_mask": decoder_mask,
                "label": label,
                "tgt_text": tgt_text
            })
        
        return result


    def _preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single item from the dataset."""
        if 'input' not in item:
            item = {'input': item}
        
        item['input'] = {k: round(v, 4) if isinstance(v, float) else v 
                         for k, v in item['input'].items()}
        
        return item


    def _tokenize_and_pad(self, text: str, tokenizer: Tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pad the input text."""
        if text == '':
            input_tensor  = torch.full((self.seq_len,), self.pad_token.item(), dtype=torch.long)
            mask = (input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int()
            return input_tensor, mask
        
        tokens = tokenizer.encode(text).ids
        padding = self.seq_len - len(tokens) - 2  # Add SOS and EOS
        
        if padding < 0:
            raise ValueError("Sequence is too long")
        
        input_tensor = torch.cat([
            self.sos_token,
            torch.tensor(tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * padding, dtype=torch.int64),
        ], dim=0)
        
        mask = (input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        
        return input_tensor, mask


    def _get_molecular_properties(self, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract molecular properties from the input data."""
        properties = {'MolecularWeight': 'mw', 'LogP': 'logp', 'HBD': 'hbd', 'HBA': 'hba', 'TPSA': 'tpsa', 
                      'RotatableBonds': 'rotatable_bonds', 'QED': 'qed', 'SA_Score': 'sa_score'}
        return {
            name: torch.tensor([input_data.get(prop, float('-inf'))], dtype=torch.float)
            for prop, name in properties.items()
        }


    def _prepare_decoder_inputs(self, tgt_text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare decoder inputs for training."""
        dec_input_tokens = self.tokenizer_smiles_tgt.encode(tgt_text).ids
        dec_padding = self.seq_len - len(dec_input_tokens) - 1  # Add SOS
        
        if dec_padding < 0:
            raise ValueError("Sequence is too long")
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_padding, dtype=torch.int64),
        ], dim=0)
        
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_padding, dtype=torch.int64),
        ], dim=0)
        
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        
        return decoder_input, label, decoder_mask


def causal_mask(size: int) -> torch.Tensor:
    """
    Create a causal mask for the decoder.

    Args:
        size (int): Size of the square mask.

    Returns:
        torch.Tensor: A boolean tensor representing the causal mask.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0