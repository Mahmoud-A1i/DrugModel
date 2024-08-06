"""
This module handles the creation and training of tokenizers for SMILES strings and scaffolds.
It also prepares the dataset for training and validation.
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import pandas as pd
from torch.utils.data import DataLoader, random_split
from typing import List, Dict, Any, Tuple, Iterator
import random
import math

from .model_utils import augment_data
from data import SMILESDataset


def create_character_tokenizer() -> Tuple[Tokenizer, trainers.WordLevelTrainer]:
    """
    Create a character-level tokenizer.

    Returns:
        Tuple[Tokenizer, trainers.WordLevelTrainer]: The tokenizer and its trainer.
    """
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.normalizer = normalizers.NFKC()
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
    return tokenizer, trainer


def train_character_tokenizer(config: Dict[str, Any], 
                              tokenizer_type: str, 
                              sentences: Iterator[str], 
                              tokenizer: Tokenizer, 
                              trainer: trainers.WordLevelTrainer
                             ) -> Tokenizer:
    """
    Train a character tokenizer on the given sentences.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        tokenizer_type (str): Type of tokenizer (e.g., 'smiles_src', 'smiles_tgt', 'scaffold').
        sentences (Iterator[str]): Iterator of sentences to train on.
        tokenizer (Tokenizer): The tokenizer to train.
        trainer (trainers.WordLevelTrainer): The trainer for the tokenizer.

    Returns:
        Tokenizer: The trained tokenizer.
    """
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(str(config['tokenizer_file'].format(tokenizer_type)))
    return tokenizer


def get_all_sentences(data: List[Dict[str, Any]], key1: str, key2: str = None) -> Iterator[str]:
    """
    Extract sentences from the data based on the given keys.

    Args:
        data (List[Dict[str, Any]]): List of data dictionaries.
        key1 (str): Primary key to access in each dictionary.
        key2 (str, optional): Secondary key to access within the primary key's value. Defaults to None.

    Yields:
        str: Extracted sentences.
    """
    if key2 is None:
        for item in data:
            yield item[key1]
    else:
        for item in data:
            if key2 in item[key1] and isinstance(item[key1][key2], str):
                yield item[key1][key2]


def get_ds(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer, Tokenizer]:
    """
    Prepare the dataset and tokenizers for training and validation.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer, Tokenizer]: 
            Train dataloader, validation dataloader, source SMILES tokenizer, 
            target SMILES tokenizer, and scaffold tokenizer.
    """
    df = pd.read_csv(config['datasource'])
    smiles_data = df.to_dict('records')

    data = []

    for smiles in smiles_data:
        if isinstance(smiles['Scaffold'], float) and math.isnan(smiles['Scaffold']):
            del smiles['Scaffold']
        data.append({'input': smiles, 'target': smiles['SMILES']})

    data = augment_data(config, data)

    tokenizer_src, trainer_src = create_character_tokenizer()
    tokenizer_tgt, trainer_tgt = create_character_tokenizer()
    
    tokenizer_smiles_src = train_character_tokenizer(config, 'smiles_src', get_all_sentences(data, 'input', 'SMILES'), tokenizer_src, trainer_src)
    tokenizer_smiles_tgt = train_character_tokenizer(config, 'smiles_tgt', get_all_sentences(data, 'target'), tokenizer_tgt, trainer_tgt)

    tokenizer_scaffold, trainer_scaffold = create_character_tokenizer()
    
    tokenizer_scaffold = train_character_tokenizer(config, 'scaffold', get_all_sentences(data, 'input', 'Scaffold'), tokenizer_scaffold, trainer_scaffold)
    
    train_ds_size = int(config['validation_split'] * len(data))
    val_ds_size = len(data) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(data, [train_ds_size, val_ds_size])

    train_ds = SMILESDataset(train_ds_raw, tokenizer_smiles_src, tokenizer_smiles_tgt, tokenizer_scaffold, config['seq_len'])
    val_ds = SMILESDataset(val_ds_raw, tokenizer_smiles_src, tokenizer_smiles_tgt, tokenizer_scaffold, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_smiles_src, tokenizer_smiles_tgt, tokenizer_scaffold
