"""
SMILES Filtering and Property Calculation Module

This module provides functionality to filter SMILES (Simplified Molecular Input Line Entry System) 
strings based on molecular properties and drug-likeness rules. It calculates various molecular 
properties using RDKit and applies filters based on Lipinski's Rule of Five, Veber's rules, 
and additional criteria.

The module can process large datasets of SMILES strings in parallel, leveraging multiprocessing 
for improved performance. It outputs the filtered SMILES along with their calculated properties 
to CSV files, including separate train and test sets.

Usage:
    python script_name.py [arguments]

Arguments:
    --qed_threshold: Quantitative Estimate of Drug-likeness threshold (default: 0.5)
    --sa_score: Synthetic Accessibility score threshold (default: 5.0)
    --min_smiles_length: Minimum number of atoms in SMILES (default: 5)
    --input_file: Input file containing SMILES strings (default: smiles.txt)
    --output_file: Output CSV file (default: processed_smiles.csv)
    --n_cores: Number of CPU cores to use (default: all available)
    --split_ratio: Ratio for train/test split (default: 0.8)
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import QED
from moses.metrics.utils import SA
import pandas as pd
from tqdm import tqdm
import multiprocessing
import argparse
from typing import List, Tuple, Optional

RDLogger.DisableLog('rdApp.*')

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Filter SMILES strings based on molecular properties.")
    parser.add_argument("--qed_threshold", type=float, default=0.5,
                        help="Quantitative Estimate of Drug-likeness threshold (default: 0.5)")
    parser.add_argument("--sa_score", type=float, default=5.0,
                        help="Synthetic Accessibility score threshold (default: 5.0)")
    parser.add_argument("--min_smiles_length", type=int, default=5,
                        help="Minimum number of atoms in SMILES (default: 5)")
    parser.add_argument("--input_file", type=str, default="smiles.txt",
                        help="Input file containing SMILES strings (default: smiles.txt)")
    parser.add_argument("--output_file", type=str, default="processed_smiles.csv",
                        help="Output CSV file (default: processed_smiles.csv)")
    parser.add_argument("--n_cores", type=int, default=multiprocessing.cpu_count(),
                        help="Number of CPU cores to use (default: all available)")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Ratio for train/test split (default: 0.8)")
    return parser.parse_args()

def calculate_properties(smiles):
    """Calculate molecular properties for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol)
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        psa = Descriptors.TPSA(mol)
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        qed = QED.qed(mol)
        sa_score = SA(mol)
    
        return smiles, mw, logp, hbd, hba, psa, rotatable_bonds, scaffold, qed, sa_score
    
    except Chem.rdchem.KekulizeException:
        print(f"Kekulization error for SMILES: {smiles}")
        return None
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(f"Error: {str(e)}")
        return None

def apply_filters(props: Optional[Tuple[str, float, float, int, int, float, int, str, float, float]]) -> bool:
    """
    Apply extended drug-likeness rules to molecular properties.

    Args:
        props (Optional[Tuple[str, float, float, int, int, float, int, str, float, float]]): 
            Tuple of molecular properties.

    Returns:
        bool: True if the molecule passes all filters, False otherwise.
    """
    if props is None:
        return False
    
    smiles, mw, logp, hbd, hba, psa, rotatable_bonds, _, qed, sa_score = props
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return False
    
    # Lipinski's Rule of Five
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])
    lipinski = lipinski_violations <= 1 # allows breaking one rule

    # Veber's rules
    veber = (rotatable_bonds <= 10) and (psa <= 140)

    return all([
        lipinski,
        veber,
        qed > QED_THRESHOLD,
        sa_score < SA_SCORE,
        mol.GetNumAtoms() > MIN_SMILES_LENGTH
    ])

def process_chunk(chunk: List[str]) -> List[Tuple[str, float, float, int, int, float, int, str, float, float]]:
    """
    Process a chunk of SMILES strings.

    Args:
        chunk (List[str]): A list of SMILES strings to process.

    Returns:
        List[Tuple[str, float, float, int, int, float, int, str, float, float]]: 
            A list of tuples containing properties of molecules that pass the filters.
    """
    results = []
    for smiles in chunk:
        props = calculate_properties(smiles)
        if props is not None and apply_filters(props):
            results.append(props)
    return results

def filter_smiles(smiles_list: List[str], n_cores: int = multiprocessing.cpu_count()) -> List[Tuple[str, float, float, int, int, float, int, str, float, float]]:
    """
    Filter SMILES strings using multiprocessing.

    Args:
        smiles_list (List[str]): A list of SMILES strings to filter.
        n_cores (int, optional): Number of CPU cores to use. Defaults to all available cores.

    Returns:
        List[Tuple[str, float, float, int, int, float, int, str, float, float]]: 
            A list of tuples containing properties of molecules that pass the filters.
    """
    chunk_size = len(smiles_list) // n_cores
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    with multiprocessing.Pool(n_cores) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))
    
    return [item for sublist in results for item in sublist]

if __name__ == "__main__":
    args = parse_arguments()

    QED_THRESHOLD = args.qed_threshold
    SA_SCORE = args.sa_score
    MIN_SMILES_LENGTH = args.min_smiles_length

    df = pd.read_csv(args.input_file, header=None, names=['SMILES'])
    smiles_list = df['SMILES'].tolist()

    # Apply filters and remove duplicates
    filtered_data = filter_smiles(smiles_list, n_cores=args.n_cores)

    columns = ['SMILES', 'MolecularWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds', 'Scaffold', 'QED', 'SA_Score']
    df_filtered = pd.DataFrame(filtered_data, columns=columns)

    df_filtered = df_filtered.drop_duplicates(subset=['SMILES'])

    # Shuffle
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split and save
    split_index = int(len(df_filtered) * args.split_ratio)

    df_train = df_filtered[:split_index]
    df_test = df_filtered[split_index:]

    train_file = args.output_file.replace('.csv', '_train.csv')
    test_file = args.output_file.replace('.csv', '_test.csv')

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_filtered.to_csv(args.output_file, index=False)

    print(f"Total SMILES processed: {len(smiles_list)}")
    print(f"SMILES passing filters: {len(df_filtered)}")
    print(f"Training set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    print(f"Training set saved to: {train_file}")
    print(f"Test set saved to: {test_file}")
