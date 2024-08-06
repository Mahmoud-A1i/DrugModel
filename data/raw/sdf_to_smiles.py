"""
SDF to SMILES Converter

This module provides functionality to convert SDF (Structure-Data File) files
to SMILES (Simplified Molecular Input Line Entry System) strings. It uses RDKit
to process molecular structures and can handle multiple SDF files in a directory.

Usage:
    python script_name.py --sdf_dir <path_to_sdf_directory> --output_file <path_to_output_file>
"""

import os
import argparse
from rdkit import Chem
from typing import List

def convert_sdf_to_smiles(sdf_directory: str, output_file_path: str) -> None:
    """
    Convert SDF files in a directory to SMILES strings and write them to an output file.

    This function iterates through all .sdf files in the given directory, converts each
    molecule to its SMILES representation, and writes it to the output file.

    Args:
        sdf_directory (str): Path to the directory containing SDF files.
        output_file_path (str): Path to the output file where SMILES strings will be written.

    Returns:
        None
    """
    if not os.path.exists(sdf_directory):
        raise FileNotFoundError(f"The directory {sdf_directory} does not exist.")

    for sdf_file in os.listdir(sdf_directory):
        if sdf_file.endswith('.sdf'):
            sdf_file_path = os.path.join(sdf_directory, sdf_file)
            supplier = Chem.SDMolSupplier(sdf_file_path)
            try:
                with open(output_file_path, 'a') as output_file:
                    for mol in supplier:
                        if mol is not None:
                            try:
                                # Convert molecule to SMILES
                                Chem.SanitizeMol(mol)
                                smiles = Chem.MolToSmiles(mol)
                                output_file.write(f"{smiles}\n")
                            except Chem.rdchem.AtomValenceException:
                                # Skip molecules that cause valence errors
                                continue
                            except Exception as e:
                                # Log other unexpected errors
                                print(f"Error processing molecule in {sdf_file}: {str(e)}")
                                continue
            except PermissionError:
                print(f"Permission denied: Cannot write to {output_file_path}")
                return
            print(f"Conversion completed for {sdf_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SDF files to SMILES format.")
    parser.add_argument("--sdf_dir", type=str, required=True, help="Directory containing SDF files.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path for SMILES strings.")
    
    args = parser.parse_args()
    
    try:
        convert_sdf_to_smiles(args.sdf_dir, args.output_file)
    except Exception as e:
        print(f"An error occurred: {str(e)}")