# Stable Cellulase Generator

**Repository purpose:** reproducible pipeline to generate candidate cellulase sequences using a Conditional VAE (CVAE) combined with a Stability Predictor (CNN).

Phase-2 implements guided generation: experimentally validated peptides are used as anchors in latent space, nearby latent points are sampled, decoded to sequences, filtered by predicted stability, diversified, and the top candidates are selected for synthesis.

---

# Repository Structure

```
.
├── cellulase_generator.py
├── phase_2.py
├── statistics.py
├── models/
├── experimental_peptides.csv
└── results/
```

**File descriptions**

- `cellulase_generator.py`  
  Trains the CVAE generative model and stability predictor.

- `phase_2.py`  
  Performs guided sequence generation using experimentally validated anchor peptides.

- `statistics.py`  
  Utility script for computing summary statistics such as medians.

- `models/`  
  Directory containing trained model files and metadata scaler.

- `experimental_peptides.csv`  
  Input dataset containing experimentally validated peptides.

- `results/`  
  Output directory where generated sequences and scoring tables are written.

---

# Required Model Artifacts

After training the CVAE and predictor using the curated cellulase dataset, the following files must be available:

```
encoder_7k.h5
decoder_7k.h5
predictor_7k.h5
cvae_7k.h5
meta_scaler_7k.pkl
```

These files should be placed inside the `models/` directory.

### Important

`meta_scaler_7k.pkl` is critical because it contains the fitted metadata scaler used during training.  
The metadata columns and their order must exactly match those used during training.

If this file is lost, latent encoding of experimental peptides will become inconsistent and the generation procedure will not function correctly.

---

# Quick Start

## Step 1 — Train the Model

Run the training pipeline on the curated cellulase dataset.

```bash
python cellulase_generator.py \
--data Fungal_7k_curated.csv \
--save_dir models/
```

This script should produce:

- encoder model
- decoder model
- predictor model
- metadata scaler (`meta_scaler_7k.pkl`)

---

## Step 2 — Prepare Experimental Peptides

Create a CSV file:

```
experimental_peptides.csv
```

Required columns include:

```
id
seq
mol_wt
aromaticity
pi
charge
gravy
instability_index
```

The metadata columns must be identical to the columns used during training.

Column order must match the scaler.

---

## Step 3 — Run Guided Generation

Execute the Phase-2 pipeline:

```bash
python phase_2.py \
--encoder models/encoder_7k.h5 \
--decoder models/decoder_7k.h5 \
--predictor models/predictor_7k.h5 \
--scaler models/meta_scaler_7k.pkl \
--experimental_csv experimental_peptides.csv \
--output_dir results/ \
--n_per_anchor 1500 \
--sigma auto \
--stability_threshold 40 \
--final_k 10
```

---

# Output Files

The pipeline generates the following outputs:

### Top candidates

```
results/top_guided_candidates.csv
```

Contains the highest scoring sequences selected for experimental synthesis.

### FASTA output

```
results/top_guided_candidates.fasta
```

FASTA formatted sequences for peptide ordering.

### Full audit table

```
results/all_stable_candidates_scored.csv
```

Contains all filtered candidates with predicted stability scores and latent distances.

---

# Phase-2 Pipeline Overview

The guided generation process performs the following steps:

1. Load trained encoder, decoder, predictor, and metadata scaler.
2. Import preprocessing functions used during training.
3. Read experimental peptides.
4. Encode peptides to obtain latent anchor vectors.
5. Estimate sampling radius (`sigma`).
6. Sample latent points near anchor vectors.
7. Decode latent vectors to amino acid sequences.
8. Remove duplicate sequences.
9. Predict stability using the CNN predictor.
10. Filter sequences by instability threshold.
11. Compute latent distances from anchors.
12. Score candidates using stability and proximity.
13. Cluster sequences in latent space to enforce diversity.
14. Select the final candidate sequences.

---

# Sampling Radius Selection

Sampling radius controls how far the algorithm explores from the anchor peptides.

Typical values:

| Anchor spacing | Recommended sigma |
|----------------|------------------|
| very close | 0.10 – 0.20 |
| moderate | 0.20 – 0.30 |
| far apart | 0.30 – 0.40 |

Automatic estimation:

```
sigma = median_pairwise_anchor_distance * 0.2
```

Manual inspection is recommended before running large generation batches.

---

# Stability Filtering

Generated sequences are filtered using the predicted instability index.

Default filter:

```
instability_index < 40
```

Stricter filter:

```
instability_index < 30
```

Lower instability values indicate higher predicted stability.

---

# Diversity Enforcement

To avoid near-identical sequences, candidates are clustered in latent space.

Example method:

```
MiniBatchKMeans
```

The top scoring candidate from each cluster is selected to ensure sequence diversity.

---

# Recommended Hyperparameters

```
n_per_anchor = 1500
sigma = 0.15 – 0.25
stability_threshold = 40
final_k = 10
```

Composite scoring weights:

```
w_instability = 0.7
w_proximity = 0.3
```

Suggested cluster count:

```
clusters ≈ min(12, number_of_candidates / 10)
```

---

# Manual Quality Checks Before Synthesis

Inspect candidate sequences for:

- valid amino acid alphabet
- appropriate sequence length
- absence of long repetitive motifs
- acceptable predicted instability values
- sufficient diversity between candidates

Optional checks:

- BLAST similarity search
- signal peptide prediction
- transmembrane domain prediction

---

# Troubleshooting

## Decoder produces invalid amino acids

Ensure the amino acid mapping used in `decode_onehot_to_sequence()` matches the mapping used during training.

---

## Metadata scaler errors

Verify that:

- `meta_scaler_7k.pkl` was generated during training.
- metadata columns appear in the same order as training data.

---

## Too few candidates pass the stability filter

Possible solutions:

- increase `n_per_anchor`
- increase `sigma`
- relax the stability threshold slightly

---

## Too many similar sequences

Increase the number of clusters used for diversity filtering.

---

# Conda Environment Example

```
conda create -n cellulase_env python=3.9 -y
conda activate cellulase_env

pip install numpy
pip install pandas
pip install scikit-learn
pip install tensorflow==2.11.0
pip install joblib
pip install scipy
```

Adjust the TensorFlow version if necessary to match the version used when saving the trained models.

---

# Reproducibility

For reproducibility, retain the following files:

```
top_guided_candidates.csv
top_guided_candidates.fasta
all_stable_candidates_scored.csv
encoder_7k.h5
decoder_7k.h5
predictor_7k.h5
meta_scaler_7k.pkl
```

---

# Scientific Limitations

The current pipeline optimizes sequence stability rather than catalytic activity.

Future improvements may include:

- training an activity predictor using experimentally measured activity data
- retraining the predictor with new labeled results from synthesis experiments
- comparing guided generation with random latent sampling strategies

---

# License

MIT License © 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.
