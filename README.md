# Stable Cellulase Sequence Generation using Machine Learning

Deep generative design of bacterial/fungal cellulase sequences conditioned on physicochemical metadata.

This project builds a Stability Predictor and a Conditional Variational Autoencoder (CVAE) to generate novel cellulase sequences predicted to be stable (Instability Index < 40).

---

## Overview

This repository implements an end-to-end pipeline to:

1. Clean and preprocess cellulase sequence datasets and associated physicochemical metadata.
2. Train a Stability Predictor (sequence + metadata → instability index).
3. Train a Conditional Variational Autoencoder (CVAE) to generate amino-acid sequences conditioned on metadata.
4. Generate candidate sequences and filter them using the Stability Predictor.

The instability index (Instability Index < 40) is used as the stability threshold.

---

## Requirements

- Python 3.8+
- TensorFlow (2.x)
- NumPy
- Pandas
- Scikit-learn

Install dependencies:

pip install numpy pandas scikit-learn tensorflow

---

## Input Data Format

The script expects a CSV file in the project directory (update filename in the script if needed).

### Required Columns

- seq — amino acid sequence (string)
- mol_wt — molecular weight
- aromaticity — aromaticity fraction
- pi — isoelectric point
- chrg — net charge
- gravy — grand average of hydropathicity
- molar extinction coefficient — two numeric values (list or parseable string)
- ss — secondary structure fractions [alpha, beta, random_coil]
- flexibility — list of values or mean-convertible values
- instability index — target stability metric

Notes:
- List/tuple columns must be parseable using ast.literal_eval.
- Sequences are padded/truncated to MAX_LEN = 500.

---

## How to Run

1. Place your CSV file in the repository directory.
2. (Optional) Run statistics helper to compute stable medians:

python statistics.py

3. Run the full pipeline:

python cellulase_generator.py

The script will:
- Preprocess data
- Train the Stability Predictor
- Train the CVAE
- Generate sequences
- Filter sequences based on predicted instability
- Save stable candidates to a text file

---

## Configuration

Open cellulase_generator.py and adjust:

- MAX_LEN
- LATENT_DIM
- EPOCHS_PREDICTOR
- EPOCHS_CVAE
- BATCH_SIZE
- KL_WEIGHT
- STABILITY_THRESHOLD
- num_sequences_to_generate

### Target Metadata Order

The metadata order must match:

['mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
 'molar_extinct1', 'molar_extinct2',
 'alpha', 'beta', 'random_coil',
 'flexibility_mean']

Choose realistic target values (e.g., median of stable proteins).

---

## Model Architecture

### Stability Predictor

- Conv1D → MaxPooling
- Conv1D → GlobalMaxPooling
- Dense (metadata branch)
- Concatenation
- Dense
- Linear output (regression)

Loss: Mean Squared Error (MSE)
Metric: Mean Absolute Error (MAE)

---

### Conditional Variational Autoencoder (CVAE)

Encoder:
- Conv1D blocks
- Flatten
- Concatenate with metadata
- Dense
- Latent mean and log variance
- Reparameterization trick

Decoder:
- Dense → reshape
- Conv1DTranspose layers
- Final Conv1D with softmax activation

Loss:
Reconstruction Loss + β * KL Divergence

---

## Output

- Console logs showing training progress
- Predicted instability scores for generated sequences
- bacterial_sequences.txt containing filtered stable sequences

---

## Limitations

- Instability index is an in silico proxy for stability.
- Generated sequences require:
  - BLAST homology validation
  - Structural modeling (e.g., AlphaFold)
  - Experimental testing

Argmax decoding reduces diversity; probabilistic sampling may improve exploration.

---

## Suggested Improvements

- Add activity prediction model
- Use attention-based encoders
- Incorporate structural embeddings
- Add diversity metrics
- Save trained models explicitly using model.save()

---

## License

Add an MIT License (recommended) or appropriate license file.

