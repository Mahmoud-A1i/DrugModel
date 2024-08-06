"""
Configuration module for the model.
Provides functions to get configuration parameters and manage weight files.
"""

from pathlib import Path
from typing import Union, Dict


def get_config(large: bool = False) -> Dict[str, Union[str, int, float]]:
    """
    Returns a dictionary containing default configuration parameters for the model.

    Args:
        large (bool): If True, updates the model's hyperparameters by increasing the model dimensions and layers. Defaults to False.
    
    Returns:
        dict: A dictionary of configuration parameters.
    """
    config = {
        # Model hyperparameters
        "d_model": 512,
        "number_of_layers": 6,
        "number_of_attention_heads": 8,
        "feed_forward": 2048,
        "dropout": 0.1,
        
        # Training parameters
        "batch_size": 32, # 8
        "num_epochs": 50,
        "lr": 10**-4,
        "seq_len": 350,
        "min_percentage": 25,
        "max_percentage": 65,
        "augmentation": [1], # Can be [1,2] or [3] for example
        "validation_split": 0.9,
        
        # Data and model paths
        "datasource": 'data/processed_smiles-10000.csv',
        "model_folder": "weights",
        "model_basename": "model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/model",
        
        # Loss adjustment parameters
        "initial_loss_alpha": 0.1,
        "initial_loss_beta": 0.1,
        "loss_adjust_rate": 0.001,
        "update_loss_freq": 1,
        "validity_threshold": 0.8,
        "similarity_threshold": 0.5,
        "label_smoothing": 0.1,
        "test_metrics_freq": 5
    }

    if large:
        config["d_model"] = 1024
        config["number_of_layers"] = 8
        config["number_of_attention_heads"] = 16
        config["feed_forward"] = 4096

    return config


def get_weights_file_path(config: dict, epoch: str) -> str:
    """
    Generates the file path for model weights based on the given configuration and epoch.
    
    Args:
        config (dict): The configuration dictionary.
        epoch (str): The epoch number.
    
    Returns:
        str: The file path for the model weights.
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config: dict, best: bool = False) -> Union[str, None]:
    """
    Finds the latest weights file in the specified folder.
    
    Args:
        config (dict): The configuration dictionary.
        best (bool): If True, use the best model.
    
    Returns:
        str or None: The path to the latest weights file, or None if no files are found.
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = sorted(list(Path(model_folder).glob(model_filename)))
    
    if len(weights_files) == 0:
        return None
    
    if best:
        best_weights = [str(weight) for weight in weights_files if 'best' in str(weight)]
        return best_weights[0] if best_weights else None

    return str(weights_files[-2 if 'best' in weights_files[-1].name else -1])