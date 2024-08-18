"""
Generate complete SMILES strings from partial inputs using a pre-trained model.

This module provides the 'generate' function, which takes a list of partial SMILES 
strings and their properties as input, and returns a list of complete, generated 
SMILES strings.

Requires a pre-trained model and configuration files to function.
"""

from pathlib import Path
import torch
from tokenizers import Tokenizer
from typing import List, Dict, Any

from config import latest_weights_file_path 
from model import build_transformer
from data import SMILESDataset, causal_mask
from .greedy import greedy_decode


def generate(config: Dict[str, Any], input_data: List[str], weights_path: str = None, best: bool = False) -> List[Dict[str, str]]:
    """
    Generate SMILES strings using a pre-trained transformer model.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        input_data (List[str]): List of dictionaries containing partial 'SMILES', optional 'Scaffold', 
                           and properties like 'MolecularWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds', 
                           'QED', 'SA_Score'.
                           'Scaffold' can be omitted if not available.
                           'SMILES' should be partial SMILES strings.
                           Example: [
                               {'SMILES': 'CC', 'Scaffold': 'C', 'MolecularWeight': 30.07, ...},
                               {'SMILES': 'CN', 'MolecularWeight': 31.06, ...}  # scaffold omitted
                           ]
        weights_path (str, optional): Path to model weights file. If None, latest weights will be used.
        best (bool): If True, use the best model, else, use the latest weights.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing 'input_smiles' (partial SMILES) and 'output_smiles' (generated SMILES) keys.
    """
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizers
    tokenizers = {
        'src': Tokenizer.from_file(str(Path(config['tokenizer_file'].format('smiles_src')))),
        'tgt': Tokenizer.from_file(str(Path(config['tokenizer_file'].format('smiles_tgt')))),
        'scaffold': Tokenizer.from_file(str(Path(config['tokenizer_file'].format('scaffold'))))
    }
    
    # Build and load model
    model = build_transformer(
        tokenizers['src'].get_vocab_size(),
        tokenizers['tgt'].get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        tokenizers['scaffold'].get_vocab_size(),
        config['d_model'],
        N=config['number_of_layers'],
        h=config['number_of_attention_heads'],
        dropout=config['dropout'],
        d_ff=config['feed_forward']
    ).to(device)

    weights_file = weights_path if weights_path else latest_weights_file_path(config, best)
    model.load_state_dict(torch.load(weights_file)['model_state_dict'])
    print(f"Currently using {weights_file.split('/')[-1]}")
    
    # Prepare dataset
    dataset = SMILESDataset(input_data, tokenizers['src'], tokenizers['tgt'], tokenizers['scaffold'], config['seq_len'], train=False)
    
    generated_data = []
    model.eval()

    with torch.no_grad():
        for item in input_data:
            batch = dataset[input_data.index(item)]
            model_out = greedy_decode(model, tokenizers['tgt'], config['seq_len'], device, 
                                      *(batch[k].unsqueeze(0).to(device) for k in ['encoder_smiles_input', 'encoder_scaffold_input', 'encoder_scaffold_mask', 'encoder_smiles_mask', 'mw', 'logp', 'hbd', 'hba',
                                                                                   'tpsa', 'rotatable_bonds', 'qed', 'sa_score']))
            
            generated_smiles = ''.join(tokenizers['tgt'].decode(model_out.squeeze().tolist()).split())
            
            generated_data.append({
                'input_smiles': item['input']['SMILES'],
                'output_smiles': generated_smiles
            })

    return generated_data
