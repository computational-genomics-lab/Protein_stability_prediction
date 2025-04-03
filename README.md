# Stable Cellulase Sequence Generation using Machine Learning

## Overview

This project uses machine learning models built with TensorFlow/Keras to generate novel bacterial cellulase protein sequences that are predicted to be stable. It leverages a dataset of known cellulase sequences and their associated physicochemical properties.

The process involves two main components:
1.  A **Stability Predictor**: A supervised model trained to predict the instability index of a cellulase based on its sequence and metadata.
2.  A **Conditional Variational Autoencoder (CVAE)**: A generative model trained to learn the distribution of cellulase sequences conditioned on their metadata. This model can generate new sequences based on desired target metadata properties.

Generated sequences are then filtered using the Stability Predictor to select candidates with a predicted instability index below 40 (a common threshold for stable proteins).

## Dataset

The script requires a CSV file named `cellulase_data.csv` located in the same directory.

* **Source:** This file should contain data curated from literature/databases (e.g., NCBI), ideally with a substantial number of sequences (the original context mentioned ~13k).
* **Required Columns:**
    * `seq`: The amino acid sequence of the protein.
    * `mol_wt`: Molecular weight.
    * `aromaticity`: Aromaticity value.
    * `pi`: Isoelectric point.
    * `ss`: Secondary structure fractional composition (e.g., alpha helix, beta-sheet, random coil). **Format Note:** The script expects this column to contain data convertible to three numerical values (e.g., a list/tuple `[0.3, 0.4, 0.3]` or a string that can be parsed into this format).
    * `chrg`: Net charge.
    * `molar extinction coefficient`: Molar extinction coefficient values. **Format Note:** The script expects this column to contain data convertible to two numerical values.
    * `flexibility`: Per-residue flexibility scores or a summary statistic. **Format Note:** The script expects this column to contain data where `numpy.mean()` can be applied (e.g., a list/array of numbers). If it's already a single mean value per protein, the `.apply(np.mean)` step in the script should be adjusted.
    * `gravy`: Grand average of hydropathicity.
    * `instability index`: The target variable for the predictor and the primary stability metric used for filtering generated sequences.

* **Data Preprocessing:** The script handles tokenization, padding (to `max_len=500` by default), and one-hot encoding for sequences. Metadata features are scaled using `StandardScaler`.

## Dependencies

* Python 3.x
* NumPy
* Pandas
* Scikit-learn
* TensorFlow (>=2.x)

## Setup

1.  Clone or download the repository/script.
2.  Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn tensorflow
    ```
3.  Ensure you have the `cellulase_data.csv` file prepared according to the specified format and placed in the same directory as the script.

## Usage

1.  **Configure Target Properties:**
    * Open the Python script (`your_script_name.py`).
    * Locate the `### Sequence Generation ###` section.
    * Modify the `target_metadata` list. This list defines the desired physicochemical properties for the *new* sequences you want to generate. The order must match the `meta_features` list defined in the preprocessing section:
        `['mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy', 'alpha', 'beta', 'random_coil', 'flexibility']`
    * **Important:** Choose realistic target values. Analyze your `cellulase_data.csv` (e.g., using `df[meta_features].describe()` or analyzing properties of known stable proteins in your dataset) to set achievable goals. Unrealistic targets may lead to poor quality generation.

2.  **Run the Script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

3.  **Output:** The script will train the predictor and CVAE models. After training, it will generate a number of candidate sequences (`n_samples=50` by default) based on the `target_metadata`. It then uses the predictor to estimate the stability of these generated sequences and prints the first 3 sequences (`stable_seqs[:3]`) that have a predicted instability index below 40.

## Code Structure

The script is organized into the following main parts:

1.  **Imports:** Imports necessary libraries.
2.  **Data Preprocessing:** Loads data, cleans/formats sequences and metadata, performs encoding and scaling.
3.  **Stability Predictor Model:** Defines, compiles, and trains the supervised model for predicting the instability index.
4.  **Conditional Variational Autoencoder (CVAE):** Defines the encoder, decoder, sampling function, custom VAE loss, and trains the CVAE model.
5.  **Sequence Generation:** Contains the `generate_sequences` function using the trained CVAE decoder and the target metadata.
6.  **Filtering:** Generates candidate sequences, uses the Stability Predictor to assess them, and prints the stable candidates.


