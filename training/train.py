"""
This module contains the main training loop for the SMILES generation model.
"""

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

from config import get_weights_file_path, latest_weights_file_path
from .validation import run_validation
from .tokenizer import get_ds
from .model_utils import get_model


def train_model(config: Dict[str, Any]) -> None:
    """
    Train the SMILES generation model.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model and training parameters.
    """
    torch.manual_seed(42)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    best_performance = float('inf')
    
    train_dataloader, val_dataloader, tokenizer_smiles_src, tokenizer_smiles_tgt, tokenizer_scaffold = get_ds(config)
    model, loss_fn = get_model(config, tokenizer_smiles_src, tokenizer_smiles_tgt, tokenizer_scaffold)
    loss_fn = loss_fn.to(device)
    model = model.to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch, global_step = 1, 0
    
    if config['preload']:
        state = torch.load(latest_weights_file_path(config))
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    update_count = config['update_loss_freq']
    update_max = config['update_loss_freq']
    previous_similarity, previous_validity = 0, 0
    update = True
        
    for epoch in range(initial_epoch, config['num_epochs'] + 1):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        validity_scores = []
        similarity_scores = []
        
        for batch in batch_iterator:
            # Move all inputs to the correct device
            src_smiles = batch['encoder_smiles_input'].to(device)
            src_scaffold = batch['encoder_scaffold_input'].to(device)
            tgt = batch['decoder_input'].to(device)
            smiles_mask = batch['encoder_smiles_mask'].to(device)
            scaffold_mask = batch['encoder_scaffold_mask'].to(device)
            tgt_mask = batch['decoder_mask'].to(device)
            
            # Move molecular properties to device
            mw = batch['mw'].to(device)
            logp = batch['logp'].to(device)
            hbd = batch['hbd'].to(device)
            hba = batch['hba'].to(device)
            tpsa = batch['tpsa'].to(device)
            rotatable_bonds = batch['rotatable_bonds'].to(device)
            qed = batch['qed'].to(device)
            sa_score = batch['sa_score'].to(device)
                        
            label = batch['label'].to(device)
    
            optimizer.zero_grad()                

            total_loss, ce_loss, validity_loss, similarity_loss, validity_score, similarity_score = loss_fn(
                model,
                src_smiles,
                src_scaffold,
                tgt,
                scaffold_mask,
                smiles_mask,
                tgt_mask,
                mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score,
                label, previous_validity, previous_similarity, update=update
            )

            if update_count == update_max:
                update = True
                update_count = 1

                validity_scores.append(validity_score)
                similarity_scores.append(similarity_score)

                previous_validity, previous_similarity = validity_loss, similarity_loss
            else:
                update = False
                update_count += 1              
            
            batch_iterator.set_postfix({
                'total_loss': f"{total_loss.item():6.3f}",
                'ce_loss': f"{ce_loss.item():6.3f}",
                'validity_loss': f"{validity_loss.item():6.3f}",
                'similarity_loss': f"{similarity_loss.item():6.3f}"
            })
            
            writer.add_scalar('train total_loss', total_loss.item(), global_step)
            writer.add_scalar('train ce_loss', ce_loss.item(), global_step)
            writer.add_scalar('train validity_loss', validity_loss.item(), global_step)
            writer.add_scalar('train equivalence_loss', similarity_loss.item(), global_step)
            writer.flush()
            total_loss.backward()

            optimizer.step()
            global_step += 1

        # Printing training results
        avg_validity_score = sum(validity_scores) / len(validity_scores)
        avg_similarity_score = sum(similarity_scores) / len(similarity_scores)
        loss_fn.update_coefficients(avg_validity_score, avg_similarity_score, config['validity_threshold'], config['similarity_threshold'])
        
        current_alpha, current_beta = loss_fn.get_current_coefficients()
        print(f"Avg Validity Score:   {avg_validity_score:.4f} -> Alpha: {current_alpha:.4f}\nAvg Similarity Score: {avg_similarity_score:.4f} ->  Beta: {current_beta:.4f}")

        # Printing validation results
        avg_val_loss = run_validation(config, model, val_dataloader, tokenizer_smiles_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, loss_fn, epoch)

        if avg_val_loss < best_performance:
            best_performance = avg_val_loss
            best_model_filename = get_weights_file_path(config, 'best')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_performance': best_performance
            }, best_model_filename)
            print(f"Average Validation loss: {best_performance:.4f} - Best Model Saved")
        else:
            print(f"Average Validation loss: {avg_val_loss:.4f}")
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'global_step': global_step}, model_filename)
