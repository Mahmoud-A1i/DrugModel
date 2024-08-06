"""
This module provides functions for validating the model during training.
"""

import torch
from typing import List, Callable, Any, Dict
from tokenizers import Tokenizer
from tqdm import tqdm
import torchmetrics
from moses.metrics.metrics import fraction_valid

from generation import greedy_decode
from utils import SMILESLoss, calculate_similarity


def run_validation(config: Dict[str, Any],
                   model: torch.nn.Module, 
                   validation_ds: torch.utils.data.Dataset, 
                   tokenizer_smiles_tgt: Tokenizer, 
                   max_len: int, 
                   device: torch.device, 
                   print_msg: Callable[[str], None], 
                   global_step: int, 
                   writer: Any,
                   loss_fn: SMILESLoss,
                   epoch: int
                  ) -> float:
    """
    Run validation on the model and print results.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (torch.nn.Module): The model to validate.
        validation_ds (torch.utils.data.Dataset): The validation dataset.
        tokenizer_smiles_tgt (Tokenizer): The target SMILES tokenizer.
        max_len (int): Maximum length for generated sequences.
        device (torch.device): The device to run validation on.
        print_msg (Callable[[str], None]): Function to print messages.
        global_step (int): Current global step in training.
        writer (Any): TensorBoard writer object.
        loss_fn (SMILESLoss): The loss function to calculate the average validation loss.
        epoch (int): The current epoch number.
    Returns:
        float: Average validation loss
    """
    model.eval()
    
    count = 0
    total_val_loss = 0.0
    
    expected, generated = [], []
    
    with torch.no_grad():
        for batch in validation_ds:
            assert batch['encoder_smiles_input'].size(0) == 1, "Batch size must be 1 for validation"

            if epoch % config['test_metrics_freq'] == 0:
                model_out = greedy_decode(model, tokenizer_smiles_tgt, max_len, device, *(batch[k].to(device) for k in ['encoder_smiles_input', 'encoder_scaffold_input', 'encoder_scaffold_mask', 'encoder_smiles_mask', 'mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'qed', 'sa_score']))
                model_out_text = tokenizer_smiles_tgt.decode(model_out.detach().cpu().numpy())
                model_out_text = ''.join(model_out_text.split())
                generated.append(model_out_text)
                target_text = batch['tgt_text'][0]
                expected.append(target_text)
                
            outputs = model(*(batch[k].to(device) for k in ['encoder_smiles_input', 'encoder_scaffold_input', 'decoder_input', 'encoder_scaffold_mask', 'encoder_smiles_mask', 'decoder_mask', 'mw', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'qed', 'sa_score']))

            val_loss = loss_fn.compute_ce_loss(outputs, batch['label'].to(device))
            total_val_loss += val_loss.item()
            count += 1

    avg_val_loss = total_val_loss / count

    if epoch % config['test_metrics_freq'] == 0:
        cer = torchmetrics.text.CharErrorRate()
        print_msg(f"TEST: Validity: {fraction_valid(generated):.5f}, Similarity: {calculate_similarity(generated, expected):.5f}, CER: {cer(generated, expected).item():.5f}")

    return avg_val_loss
