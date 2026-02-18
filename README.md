# Stable Cellulase Sequence Generation using Machine Learning

Deep generative modeling of cellulase enzymes using a Stability Predictor and a Conditional Variational Autoencoder (CVAE).

This project designs novel cellulase sequences predicted to be stable (Instability Index < 40) using supervised and generative deep learning.

---

## Project Overview

This pipeline performs:

1. Data preprocessing and metadata extraction  
2. Training a Stability Predictor (regression model)  
3. Training a Conditional Variational Autoencoder (CVAE)  
4. Conditional sequence generation  
5. Filtering generated sequences by predicted stability  

The instability index (threshold < 40) is used as a computational proxy for protein stability.

---

## Model Architecture

### Stability Predictor

Multi-input neural network:

Sequence branch:
- Conv1D → MaxPooling  
- Conv1D → GlobalMaxPooling  

Metadata branch:
- Dense layer (ReLU)

Combined:
- Dense layer  
- Linear output (Instability Index regression)

Loss: Mean Squared Error (MSE)  
Metric: Mean Absolute Error (MAE)

---

### Conditional Variational Autoencoder (CVAE)

Encoder:
- Conv1D blocks  
- Flatten  
- Concatenate with metadata  
- Dense layer  
- Latent mean + log variance  
- Reparameterization trick  

Decoder:
- Dense → reshape  
- Conv1DTranspose layers  
- Final Conv1D with softmax activation  

Loss Function:

Reconstruction Loss + β × KL Divergence

---

## Repository Structure

- `cellulase_generator.py` — Main training and generation pipeline  
- `statistics.py` — Stable median metadata calculation  
- `methodology_22.4.15.docx` — Detailed methodology and results  

---

## Dataset Requirements

Place your dataset CSV file in the repository root.

### Required Columns

- `seq` — amino acid sequence (string)  
- `mol_wt` — molecular weight  
- `aromaticity`  
- `pi`  
- `chrg`  
- `gravy`  
- `molar extinction coefficient` — two numeric values (list or parseable string)  
- `ss` — secondary structure fractions `[alpha, beta, random_coil]`  
- `flexibility` — list or mean-convertible values  
- `instability index` — target stability metric  

### Notes

- List/tuple columns must be parseable using `ast.literal_eval`  
- Sequences are padded/truncated to `MAX_LEN = 500`  
- Metadata is standardized using `StandardScaler`  

---

## Environment (Tested Configuration)

This project was tested with:

- Python 3.9  
- TensorFlow 2.10.1  
- NumPy 1.23.5  
- Pandas 1.5.3  
- Scikit-learn 1.1.3  
- h5py 3.8.0  
- protobuf 3.19.6  

Using these exact versions is strongly recommended to avoid binary incompatibility issues.

---

## Installation

Create a clean conda environment:

(conda create -n cellulase_env python=3.9)
(conda activate cellulase_env)

Install dependencies:

(pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.1.3 tensorflow==2.10.1 h5py==3.8.0 protobuf==3.19.6)

---

## How to Run

### 1. (Optional) Compute Median Stable Metadata

(python statistics.py)

This calculates median values for proteins with Instability Index < 40.

### 2. Run Full Pipeline

(python cellulase_generator.py)

The script will:

- Clean and preprocess data  
- Train the Stability Predictor  
- Train the CVAE  
- Generate new sequences  
- Filter stable candidates  
- Save results to `fungal_sequences.txt` (or the defined output file)  

---

## Configure Target Generation Properties

In `cellulase_generator.py`, locate:

target_metadata_example = [...]


The order must match:

['mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
'molar_extinct1', 'molar_extinct2',
'alpha', 'beta', 'random_coil',
'flexibility_mean']


Using median values from stable proteins is recommended for realistic conditioning.

---

## Output

- Training logs in console  
- Predicted instability scores  
- Stable sequences saved to text file  
- Top candidates printed in terminal  

---

## Scientific Considerations

The instability index is an in silico proxy and does not guarantee experimental stability.

Generated sequences should be validated using:

- BLAST homology search  
- AlphaFold structure prediction  
- Active-site conservation analysis  
- Experimental expression and stability assays  

---

## Limitations

- No explicit activity prediction  
- No structural constraints in the generative model  
- Argmax decoding reduces sequence diversity  
- Stability predictor performance depends on dataset quality  

---

## Future Improvements

- Add enzyme activity predictor  
- Integrate structure-aware embeddings  
- Use probabilistic decoding instead of argmax  
- Add sequence diversity metrics  
- Save trained models for reuse  

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
