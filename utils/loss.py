"""
Custom loss module for SMILES-based molecule generation.
Implements a combined loss function incorporating cross-entropy, validity, and similarity losses.
"""

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem
from tokenizers import Tokenizer
from typing import List, Tuple

RDLogger.DisableLog('rdApp.*')


class SMILESLoss(nn.Module):
    """
    Custom loss function for SMILES-based molecule generation.
    Combines cross-entropy loss with chemical validity and similarity losses.
    """
    
    def __init__(self, 
                 tokenizer: Tokenizer,
                 ignore_index: int,
                 initial_alpha: float = 0.1,
                 initial_beta: float = 0.1,
                 label_smoothing: float = 0.1,
                 alpha_adjust_rate: float = 0.001,
                 beta_adjust_rate: float = 0.001,
                 min_alpha: float = 0.01,
                 max_alpha: float = 1.0,
                 min_beta: float = 0.01,
                 max_beta: float = 1.0):
        """
        Initialize the SMILESLoss module.

        Args:
            tokenizer (Tokenizer): Tokenizer object for SMILES tokens.
            ignore_index (int): Index to ignore in loss calculation (usually padding).
            initial_alpha (float): Initial weight for validity loss.
            initial_beta (float): Initial weight for similarity loss.
            label_smoothing (float): Label smoothing factor for cross-entropy loss.
            alpha_adjust_rate (float): Rate to adjust alpha coefficient.
            beta_adjust_rate (float): Rate to adjust beta coefficient.
            min_alpha (float): Minimum value for alpha coefficient.
            max_alpha (float): Maximum value for alpha coefficient.
            min_beta (float): Minimum value for beta coefficient.
            max_beta (float): Maximum value for beta coefficient.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.alpha_adjust_rate = alpha_adjust_rate
        self.beta_adjust_rate = beta_adjust_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing, reduction='none')
        self.epoch = 0


    def forward(self, 
                model: nn.Module, 
                src_seq: torch.Tensor, 
                src_scaffold: torch.Tensor, 
                tgt: torch.Tensor, 
                scaffold_mask: torch.Tensor, 
                smiles_mask: torch.Tensor, 
                tgt_mask: torch.Tensor, 
                mw: torch.Tensor, logp: torch.Tensor, hbd: torch.Tensor, hba: torch.Tensor, tpsa: torch.Tensor, rotatable_bonds: torch.Tensor, qed: torch.Tensor, sa_score: torch.Tensor, 
                label: torch.Tensor, 
                previous_validity: torch.Tensor, 
                previous_similarity: torch.Tensor, 
                update: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Compute the combined loss for the model output.

        Args:
            model (nn.Module): The model to evaluate.
            src_seq (torch.Tensor): Source SMILES input tensor.
            src_scaffold (torch.Tensor): Source scaffold input tensor.
            tgt (torch.Tensor): Target sequence tensor.
            scaffold_mask (torch.Tensor): Mask tensor for scaffold input.
            smiles_mask (torch.Tensor): Mask tensor for SMILES input.
            tgt_mask (torch.Tensor): Mask tensor for target sequence.
            mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score (torch.Tensor)s: Molecular property inputs.
            label (torch.Tensor): True labels tensor.
            previous_validity (torch.Tensor): Previous validity loss tensor.
            previous_similarity (torch.Tensor): Previous similarity loss tensor.
            update (bool): Whether to update the loss coefficients.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]: Contains total loss, cross-entropy loss, validity loss, similarity loss,
                   validity score, and similarity score.
        """
        outputs = model(src_seq, src_scaffold, tgt, scaffold_mask, smiles_mask, tgt_mask, mw, logp, hbd, hba, tpsa, rotatable_bonds, qed, sa_score)
        ce_loss = self.compute_ce_loss(outputs, label)

        if update:
            validity_loss, similarity_loss, validity_score, similarity_score = self.chemical_losses(outputs, label)
        else:
            validity_loss, similarity_loss, validity_score, similarity_score = previous_validity, previous_similarity, 0, 0

        total_loss = ce_loss + self.alpha * validity_loss + self.beta * similarity_loss
        
        return total_loss, ce_loss, validity_loss, similarity_loss, validity_score, similarity_score 


    def update_coefficients(self, validity_score: float, similarity_score: float, validity_threshold: float, similarity_threshold: float) -> None:
        """
        Update alpha and beta coefficients based on validity and similarity scores.

        Args:
            validity_score (float): Current validity score.
            similarity_score (float): Current similarity score.
            validity_threshold (float): Alpha is modified based on this threshold.
            similarity_threshold (float): Beta is modified based on this threshold.
        """
        if validity_score < validity_threshold:
            self.alpha = min(self.alpha + self.alpha_adjust_rate, self.max_alpha)
        else:
            self.alpha = max(self.alpha - self.alpha_adjust_rate, self.min_alpha)
    
        if similarity_score < similarity_threshold:
            self.beta = min(self.beta + self.beta_adjust_rate, self.max_beta)
        else:
            self.beta = max(self.beta - self.beta_adjust_rate, self.min_beta)


    def compute_ce_loss(self, outputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss for the model outputs.
    
        This method calculates the cross-entropy loss, applies masking to ignore
        padded elements, and returns the mean loss across the batch.
    
        Args:
            outputs (torch.Tensor): The model output tensor of shape (batch_size, seq_len, vocab_size).
            label (torch.Tensor): The target label tensor of shape (batch_size, seq_len).
    
        Returns:
            torch.Tensor: The mean cross-entropy loss across the batch.
        """
        batch_size, seq_len, vocab_size = outputs.shape
        
        # Calculate CrossEntropyLoss
        ce_loss = self.ce_loss(outputs.view(-1, vocab_size), label.view(-1))
        ce_loss = ce_loss.view(batch_size, seq_len)
        
        # Create a mask for non-padded elements
        mask = (label != self.ignore_index).float()
        
        # Apply mask and calculate mean CE loss
        ce_loss = (ce_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return ce_loss.mean()


    def get_current_coefficients(self) -> tuple:
        """
        Get the current values of alpha and beta coefficients.

        Returns:
            tuple: Current alpha and beta values.
        """
        return self.alpha, self.beta


    def chemical_losses(self, outputs: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Compute chemical validity and similarity losses.

        Args:
            outputs: Model output logits.
            label: True labels.

        Returns:
            tuple: Contains validity loss, similarity loss, validity score, and similarity score.
        """
        pred_smiles = self.decode_smiles(outputs.argmax(dim=-1))
        target_smiles = self.decode_smiles(label)
    
        validity_losses = []
        similarity_losses = []
        valid_count = 0
        total_similarity = 0
    
        for pred, target in zip(pred_smiles, target_smiles):
            pred_mol = Chem.MolFromSmiles(pred)
            target_mol = Chem.MolFromSmiles(target)
    
            if pred_mol is None:
                validity_losses.append(torch.tensor(1.0, device=outputs.device))
            else:
                validity_losses.append(torch.tensor(0.0, device=outputs.device))
                valid_count += 1
            
            if pred_mol is not None and target_mol is not None:
                pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=1024)
                target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)
                similarity = DataStructs.TanimotoSimilarity(pred_fp, target_fp)
                similarity_losses.append(torch.tensor(1 - similarity, device=outputs.device))
                total_similarity += similarity
            else:
                similarity_losses.append(torch.tensor(1.0, device=outputs.device))
    
        validity_loss = torch.stack(validity_losses).mean()
        similarity_loss = torch.stack(similarity_losses).mean()
    
        validity_score = valid_count / len(pred_smiles)
        similarity_score = total_similarity / len(pred_smiles)
    
        return validity_loss, similarity_loss, validity_score, similarity_score


    def decode_smiles(self, tensor: torch.Tensor) -> List[str]:
        """
        Decode a tensor of token indices into SMILES strings.

        Args:
            tensor: Tensor of token indices.

        Returns:
            list: List of decoded SMILES strings.
        """
        smiles_list = []
        
        for seq in tensor:
            smiles = ""
            for idx in seq:
                token = self.tokenizer.id_to_token(idx.item())
                if token == "[EOS]":
                    break
                smiles += token
            smiles_list.append(smiles)
            
        return smiles_list