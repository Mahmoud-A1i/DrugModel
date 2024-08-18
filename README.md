# SMILES Drug Generation Model

This project implements a transformer-based model for generating drug-like molecules using SMILES (Simplified Molecular Input Line Entry System) strings. The model incorporates molecular properties and scaffolds to guide the generation process.

## Project Structure
```
DrugModel/
│
├── model/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer_blocks.py
│   ├── transformer.py
│   └── utils.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── validation.py
│   ├── tokenizer.py
│   └── model_utils.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── raw/
│   │   ├── run.sh
│   │   └── sdf_to_smiles.py
│   └── processing/
│       ├── preprocess.py
│       └── run.sh
│
├── utils/
│   ├── __init__.py
│   ├── sascorer.py
│   ├── metrics.py
│   └── loss.py
│
├── config.py
├── requirements.txt
└── main.py
```

## Installation

1. Clone the repository:
```
git clone https://github.com/Mahmoud-A1i/DrugModel.git
cd DrugModel
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## How to Train with Your Own Data

1. Prepare your data:
Place your SMILES strings in a text file (one per line) in the `data/processing/` directory.

2. Preprocess the data:
```
cd data/processing
./run.sh
```
This will run the `preprocess.py` script, which filters the SMILES strings based on drug-likeness criteria and calculates molecular properties. The output files will be saved in the `data/` directory.

3. Train the model:
```
python main.py train --train_file data/processed_smiles_train.csv
```

## Usage

To generate new SMILES strings using a trained model:
```
python main.py generate --input_file path/to/input.csv --output_file path/to/output.csv --weights_path path/to/model_weights.pt
```

To evaluate the model:
```
python main.py test --test_file data/processed_smiles_test.csv --weights_path path/to/model_weights.pt
```

## Model Architecture

Our model is based on the transformer architecture with some modifications tailored for SMILES string generation:

1. Encoder:
   - The encoder processes the input partial SMILES string using self-attention mechanisms.
   - It then applies cross-attention to incorporate information from the molecular properties and scaffold.
   
2. Decoder:
   - The decoder generates the output SMILES string token by token.
   - It uses self-attention on the generated sequence and cross-attention to attend to the encoder's output.

3. Property and Scaffold Integration:
   - Molecular properties are embedded and added to the encoder's output.
   - The scaffold is processed separately and integrated via cross-attention in the encoder.

This architecture allows the model to consider not just the partial SMILES string, but also the desired molecular properties and scaffold information during generation. The self-attention mechanisms help the model understand the relationships between different parts of the SMILES string, while the cross-attention mechanisms allow it to incorporate the additional property and scaffold information effectively.

## Custom Loss Function

Our model uses a custom loss function that combines cross-entropy loss with chemical validity and similarity losses. This helps ensure that the generated molecules are not only syntactically correct but also chemically valid and similar to the target molecules.

The total loss is calculated as follows:

$$ L_{total} = L_{CE} + \alpha \cdot L_{validity} + \beta \cdot L_{similarity} $$

Where:
- $L_{CE}$ is the cross-entropy loss
- $L_{validity}$ is the validity loss
- $L_{similarity}$ is the similarity loss
- $\alpha$ and $\beta$ are dynamically adjusted coefficients

### Cross-Entropy Loss ($L_{CE}$)

The cross-entropy loss is calculated using a standard cross-entropy function with label smoothing:

$$ L_{CE} = - \sum_{i} y_i \log(p_i) $$

Where $y_i$ are the smoothed target probabilities and $p_i$ are the predicted probabilities.

### Validity Loss ($L_{validity}$)

The validity loss is binary, penalizing invalid SMILES strings:

$$ L_{validity} = \begin{cases} 
1, & \text{if SMILES is invalid} \\
0, & \text{if SMILES is valid}
\end{cases} $$

### Similarity Loss ($L_{similarity}$)

The similarity loss is based on the Tanimoto similarity between the generated and target molecules:

$$ L_{similarity} = 1 - \text{TanimotoSimilarity}(FP_{generated}, FP_{target}) $$

Where $FP_{generated}$ and $FP_{target}$ are Morgan fingerprints of the generated and target molecules, respectively.

### Dynamic Coefficient Adjustment

The coefficients $\alpha$ and $\beta$ are adjusted during training based on the model's performance at the end of every epoch:

$$ \alpha_{new} = \begin{cases} 
\min(\alpha + \alpha_\text{adjust rate}, \alpha_{max}), & \text{if } \text{validity score} < \text{validity threshold} \\
\max(\beta - \beta_\text{adjust rate}, \beta_{min}), & \text{if } \text{similarity score} < \text{similarity threshold}
\end{cases} $$

## Acknowledgements

I acknowledge the use of the base transformer architecture from https://github.com/hkproj/pytorch-transformer, which served as a starting point for the model developed in this project. The architecture was substantially modified and adapted to suit the specific requirements of SMILES generation and molecular property prediction.

I also utilized code from the Molecular Sets repository (https://github.com/molecularsets/moses), specifically the methods found in utils/sascorer.py, including calculateScore, novelty, and fraction_valid. Due to the outdated nature of the library, direct usage was not feasible; therefore, I integrated and adapted the necessary components into this project to ensure compatibility and functionality.

This dynamic adjustment helps balance the different components of the loss function throughout the training process.

