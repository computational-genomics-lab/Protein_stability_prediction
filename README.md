# 🧬 Stable Cellulase Sequence Generation using Machine Learning

Deep generative modeling of cellulase enzymes using a Stability Predictor and a Conditional Variational Autoencoder (CVAE).

This project designs **novel cellulase sequences predicted to be stable** (Instability Index < 40) using supervised and generative deep learning.

---

## 🚀 What This Project Does

This pipeline performs:

1. 🔬 Data preprocessing and metadata extraction  
2. 🧠 Training a Stability Predictor (regression model)  
3. 🧬 Training a Conditional Variational Autoencoder (CVAE)  
4. 🎯 Conditional sequence generation  
5. 🛡 Filtering generated sequences by predicted stability  

The instability index (threshold < 40) is used as a proxy for protein stability.

---

## 🏗 Model Architecture Overview

### 🔹 Stability Predictor

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

### 🔹 Conditional Variational Autoencoder (CVAE)

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

## 📂 Repository Structure

- `cellulase_generator.py` → Main training + generation pipeline  
- `statistics.py` → Stable median metadata calculation  
- `methodology_22.4.15.docx` → Detailed methodology and results  

---

## 📊 Input Dataset Requirements

Place your dataset CSV in the repository root.

### Required Columns

- `seq` — amino acid sequence (string)
- `mol_wt` — molecular weight
- `aromaticity`
- `pi`
- `chrg`
- `gravy`
- `molar extinction coefficient` — two numeric values (list or parseable string)
- `ss` — secondary structure fractions [alpha, beta, random_coil]
- `flexibility` — list or mean-convertible values
- `instability index` — target stability metric

### Notes

- List/tuple columns must be parseable using `ast.literal_eval`
- Sequences are padded/truncated to `MAX_LEN = 500`
- Metadata is standardized using `StandardScaler`

---

## ⚙ Installation

Requirements:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn

Install dependencies:

pip install numpy pandas scikit-learn tensorflow

---

## ▶ How To Run

### 1️⃣ (Optional) Compute Median Stable Metadata

python statistics.py

This calculates median values for proteins with Instability Index < 40.

---

### 2️⃣ Run Full Pipeline

python cellulase_generator.py

The script will:

- Clean and preprocess data
- Train Stability Predictor
- Train CVAE
- Generate new sequences
- Filter stable candidates
- Save results to `bacterial_sequences.txt`

---

## 🎯 Configure Target Generation Properties

In `cellulase_generator.py`, locate:

target_metadata_example = [...]

The order must match:

['mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
 'molar_extinct1', 'molar_extinct2',
 'alpha', 'beta', 'random_coil',
 'flexibility_mean']

Tip: Use median values from stable proteins for realistic conditioning.

---

## 📈 Output

- Training logs in console
- Predicted instability scores
- Stable sequences saved to text file
- Top candidates printed in terminal

---

## 🧪 Scientific Considerations

⚠ Instability index is an in silico proxy.

Generated sequences should be validated using:

- BLAST homology search
- AlphaFold structure prediction
- Active-site conservation analysis
- Experimental expression and stability assays

---

## 🔧 Limitations

- No explicit activity prediction
- No structural constraints in model
- Argmax decoding reduces diversity
- Stability predictor accuracy depends on dataset quality

---

## 🚀 Future Improvements

- Add enzyme activity predictor
- Integrate structure-aware embeddings
- Use probabilistic decoding instead of argmax
- Add diversity metrics
- Save trained models explicitly

---

## 📜 License

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
