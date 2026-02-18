import numpy as np
import pandas as pd
import ast # For safely evaluating string representations of lists/tuples


### Data Loading and Preprocessing ###

print("1. Loading data...")
df = pd.read_csv('Fungal_analysis.csv')
print(f"Initial rows: {len(df)}")

# --- Initial Sequence Cleaning ---
print("2. Initial Sequence Cleaning...")
# Remove rows where 'seq' is not a string or is NaN *first*
initial_count = len(df)
df = df.dropna(subset=['seq'])
df = df[df['seq'].apply(lambda x: isinstance(x, str))]
print(f"Removed {initial_count - len(df)} rows with invalid 'seq' entries. Rows remaining: {len(df)}")

# --- Finding statistics of the sequences ---
# Create a new column for sequence lengths
#df['seq_len'] = df['seq'].apply(len)

# Summary statistics
#print(df['seq_len'].describe())

# --- Median values for target metadata  ---

print("Calculating median metadata for stable proteins...")

# Define the stability threshold
STABILITY_THRESHOLD = 40 # Or maybe even lower, like 20, for more confidence

# Define the columns for which you need median values (must match meta_features_cols later!)
# Make sure these columns exist in 'df' after cleaning!
cols_for_median = [
    'mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
    'molar_extinct1', 'molar_extinct2',
    'alpha', 'beta', 'random_coil',
    'flexibility_mean'
]

target_col_name = 'instability index' # Make sure this matches your column name

# Filter the DataFrame for stable proteins
stable_df = df[df[target_col_name] < STABILITY_THRESHOLD].copy()

if stable_df.empty:
    print(f"Warning: No proteins found with instability < {STABILITY_THRESHOLD}. Check data or threshold.")
    # Handle this case: maybe use the overall median, or a specific example known to be stable
    # For now, let's calculate overall median as a fallback
    median_stable_meta = df[cols_for_median].median()
    print("Using OVERALL median metadata as fallback.")
else:
    # Calculate the median for each column
    median_stable_meta = stable_df[cols_for_median].median()

    print(median_stable_meta)

# --- Use these median values for generation ---
target_metadata_example = median_stable_meta.tolist()

print(target_metadata_example)
