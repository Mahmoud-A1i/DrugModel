"""
Main script for the SMILES Drug Generation Model.

This script provides a command-line interface for various operations:
- Generating drug SMILES strings
- Running model tests
- Training the model
- Training and then testing the model

It uses argparse for command-line argument parsing and integrates with other modules
in the project for specific functionalities.
"""

import argparse
import pandas as pd
import math
from typing import List, Dict, Any, Union

from config import get_config
from training import train_model
from generation import generate
from utils import test, partial_smiles


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for different modes of operation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='SMILES Drug Generation Model')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Generate mode
    generate_parser = subparsers.add_parser('generate', help='Generate Drug SMILES strings')
    generate_parser.add_argument('--large', action='store_true', help='Use large model configuration')
    generate_parser.add_argument('--weights_path', type=str, help='Path to model weights file')
    generate_parser.add_argument('--best', action='store_true', help='Use the best model weights')
    generate_parser.add_argument('--input_file', type=str, required=True, help='Path to input file')
    generate_parser.add_argument('--output_file', type=str, required=True, help='Path to output file')

    # Test mode
    test_parser = subparsers.add_parser('test', help='Run model tests')
    test_parser.add_argument('--large', action='store_true', help='Use large model configuration')
    test_parser.add_argument('--test_file', type=str, default='data/test_smiles-1000.csv', help='Path to test SMILES file')
    test_parser.add_argument('--train_file', type=str, default='data/processed_smiles-10000.csv', help='Path to train SMILES file')
    test_parser.add_argument('--min_percentage', type=int, default=25, help='Minimum percentage for partial SMILES')
    test_parser.add_argument('--max_percentage', type=int, default=65, help='Maximum percentage for partial SMILES')
    test_parser.add_argument('--weights_path', type=str, default=None, help='Path to model weights file')
    test_parser.add_argument('--best', action='store_true', help='Use the best model weights')
    test_parser.add_argument('--output_file', type=str, default=None, help='Path to output file for test cases')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--large', action='store_true', help='Use large model configuration')
    train_parser.add_argument('--train_file', type=str, default='data/processed_smiles-10000.csv', help='Path to train SMILES file')

    # Train and test mode
    train_test_parser = subparsers.add_parser('pipeline', help='Train and then test the model')
    train_test_parser.add_argument('--large', action='store_true', help='Use large model configuration')
    train_test_parser.add_argument('--test_file', type=str, default='data/test_smiles-1000.csv', help='Path to test SMILES file')
    train_test_parser.add_argument('--train_file', type=str, default='data/processed_smiles-10000.csv', help='Path to train SMILES file')
    train_test_parser.add_argument('--min_percentage', type=int, default=25, help='Minimum percentage for partial SMILES')
    train_test_parser.add_argument('--max_percentage', type=int, default=65, help='Maximum percentage for partial SMILES')
    train_test_parser.add_argument('--best', action='store_true', help='Use the best model weights')
    train_test_parser.add_argument('--output_file', type=str, default=None, help='Path to output file for test cases')

    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the selected mode of operation based on command-line arguments.
    """
    args = parse_args()

    if args.mode == 'generate':
        config = get_config(large=args.large)
        input_data = load_input_data(args.input_file)
        generated_smiles = generate(config, input_data, args.weights_path, args.best)
        save_output_data(generated_smiles, args.output_file)

    elif args.mode == 'test':
        config = get_config(large=args.large)
        generated_smiles = test(
            config=config,
            test_file=args.test_file,
            train_file=args.train_file,
            min_percentage=args.min_percentage,
            max_percentage=args.max_percentage,
            weights=args.weights_path,
            best=args.best
        )
        save_output_data(generated_smiles, args.output_file)

    elif args.mode == 'train':
        config = get_config(large=args.large)
        config['datasource'] = args.train_file
        train_model(config)

    elif args.mode == 'pipeline':
        # Train
        config = get_config(large=args.large)
        config['datasource'] = args.train_file
        train_model(config)

        # Test
        generated_smiles = test(
            config=config,
            test_file=args.test_file,
            train_file=args.train_file,
            min_percentage=args.min_percentage,
            max_percentage=args.max_percentage,
            best=args.best
        )
        save_output_data(generated_smiles, args.output_file)

    else:
        print("Invalid mode. Use 'generate', 'test', 'train', or 'pipeline'.")


def load_input_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and preprocess input data from a CSV file.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        List[Dict[str, Any]]: Preprocessed input data.
    """
    df = pd.read_csv(file_path, header=None, names=['SMILES', 'MolecularWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds', 'Scaffold', 'QED', 'SA_Score'])
    smiles_data = df.to_dict('records')

    data = []
    for smiles in smiles_data:
        if isinstance(smiles['Scaffold'], float) and math.isnan(smiles['Scaffold']):
            del smiles['Scaffold']
        data.append(partial_smiles({'input': smiles}))
    return data


def save_output_data(generated_smiles: List[str], file_path: Union[str, None]) -> None:
    """
    Save generated SMILES strings to a CSV file.

    Args:
        generated_smiles (List[str]): List of generated SMILES strings.
        file_path (Union[str, None]): Path to the output CSV file.
    """
    if file_path is not None:
        df = pd.DataFrame(generated_smiles)
        df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()