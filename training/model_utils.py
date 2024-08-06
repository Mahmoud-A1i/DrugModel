"""
This module provides utility functions for model operations, including decoding,
data augmentation, and model initialization.
"""

import torch
import random
from rdkit import Chem, RDLogger
from typing import Dict, List, Any, Tuple
from tokenizers import Tokenizer

from model import build_transformer
from utils import SMILESLoss
from copy import deepcopy

RDLogger.DisableLog('rdApp.*')


def partial_smiles(item: Dict[str, Any], min_percentage: int = 25, max_percentage: int = 65) -> Dict[str, Any]:
    """
    Create a partial SMILES string from the input.

    Args:
        item (Dict[str, Any]): Input data dictionary.
        min_percentage (int): Minimum percentage of SMILES to keep.
        max_percentage (int): Maximum percentage of SMILES to keep.

    Returns:
        Dict[str, Any]: Modified input data with partial SMILES.
    """
    percentage = random.randint(min_percentage, max_percentage)
    cutoff = max(5, int(len(item['input']['SMILES']) * percentage / 100))  # Ensure at least five character
    item['input']['SMILES'] = item['input']['SMILES'][:cutoff]
    return item


def randomize_smiles(smiles: str) -> str:
    """
    Randomize the SMILES string.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        str: Randomized SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    return smiles  # Return original if conversion fails


def augment_data(config: Dict[str, Any], data: List[Dict[str, Any]], property_removal_probability: float = 0.2) -> List[Dict[str, Any]]:
    """
    Augment the input data based on the configuration settings.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing augmentation settings.
        data (List[Dict[str, Any]]): List of input data items.
        property_removal_probability (float, optional): Probability of removing a property during augmentation. Defaults to 0.2.

    Returns:
        List[Dict[str, Any]]: Augmented data list.
    """
    augmented_data = []
    property_keys = ['MolecularWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds', 'QED', 'SA_Score']
    random.seed(42)
    
    for item in data:
        augmentation = config['augmentation']
        
        # Original data
        if 1 in augmentation:
            augmented_data.append(partial_smiles(item, min_percentage=config['min_percentage'], max_percentage=config['max_percentage']))
        
        # Augmented data with property masking
        if 2 in augmentation:
            augmented_item = deepcopy(item)
            augmented_item['input'] = deepcopy(item['input'])
            
            if 'Scaffold' in augmented_item['input'] and random.random() < 0.5:
                del augmented_item['input']['Scaffold']
            
            for i, prop in enumerate(property_keys):
                if random.random() < property_removal_probability:
                    del augmented_item['input'][prop]
            
            augmented_data.append(partial_smiles(augmented_item))
        
        # Augmented data with randomized SMILES
        if 3 in augmentation:
            randomized_item = deepcopy(item)
            randomized_item['input'] = deepcopy(item['input'])
            randomized_item['input']['SMILES'] = randomize_smiles(item['input']['SMILES'])
            augmented_data.append(randomized_item)
    
    return augmented_data


def get_model(config: Dict[str, Any], vocab_src: Tokenizer, vocab_tgt: Tokenizer, scaffold_src: Tokenizer) -> Tuple[torch.nn.Module, SMILESLoss]:
    """
    Initialize the model and loss function.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        vocab_src (Tokenizer): Source vocabulary tokenizer.
        vocab_tgt (Tokenizer): Target vocabulary tokenizer.
        scaffold_src (Tokenizer): Scaffold vocabulary tokenizer.

    Returns:
        Tuple[torch.nn.Module, SMILESLoss]: Initialized model and loss function.
    """
    src_vocab_size = vocab_src.get_vocab_size()
    tgt_vocab_size = vocab_tgt.get_vocab_size()
    scaffold_vocab_size = scaffold_src.get_vocab_size()
    
    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config['seq_len'],
        config['seq_len'],
        scaffold_vocab_size,
        config['d_model'],
        N=config['number_of_layers'],
        h=config['number_of_attention_heads'],
        dropout=config['dropout'],
        d_ff=config['feed_forward']
    )
    loss_fn = SMILESLoss(
        vocab_tgt, 
        vocab_tgt.token_to_id("[PAD]"), 
        initial_alpha=config['initial_loss_alpha'], 
        initial_beta=config['initial_loss_beta'],
        label_smoothing=config['label_smoothing'],
        alpha_adjust_rate=config['loss_adjust_rate'],
        beta_adjust_rate=config['loss_adjust_rate']
    )
    return model, loss_fn