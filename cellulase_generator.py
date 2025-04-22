import numpy as np
import pandas as pd
import ast # For safely evaluating string representations of lists/tuples
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks, losses
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

### Configuration ###
MAX_LEN = 500       # Maximum sequence length for padding/truncating
LATENT_DIM = 128     # Dimensionality of the latent space
EPOCHS_PREDICTOR = 50 # Training epochs for the stability predictor
EPOCHS_CVAE = 200     # Maximum training epochs for the CVAE 
BATCH_SIZE = 64      # Batch size for training
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.15 # Used within CVAE training
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
KL_WEIGHT = 1.0 # Weight for KL divergence term in VAE loss (can be tuned)
LEARNING_RATE = 0.0005
CLIPNORM = 1.0      # Gradient clipping norm

### Helper Functions ###

def safe_split_pair(value, expected_len):
    """
    Safely parses a value expected to be a list/tuple (or its string representation)
    of a specific length. Returns a list of NaNs on failure or wrong length.
    """
    default_return = [np.nan] * expected_len

    if pd.isna(value):
        return default_return

    parsed_value = None
    if isinstance(value, (list, tuple)):
        parsed_value = value
    elif isinstance(value, str):
        try:
            # Use ast.literal_eval for safe parsing of '[...]' or '(...)'
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return default_return # String is not a valid literal

    # Check if parsing was successful and the result is a list/tuple
    if isinstance(parsed_value, (list, tuple)):
        if len(parsed_value) == expected_len:
            # Convert to float, handling potential non-numeric entries gracefully
            try:
                return [float(item) for item in parsed_value]
            except (ValueError, TypeError):
                 return default_return # Contains non-numeric items
        else:
            # Handle wrong length (returning NaNs is often safest)
            # print(f"Warning: Expected length {expected_len}, got {len(parsed_value)} for value {value}")
            return default_return
    else:
        # Input was not a list/tuple or a valid string representation, or parsed to something else
         return default_return

def fix_and_parse_list_string(x):
    """
    Attempts to fix common string formatting issues for list representations
    and then parses the string into a list using ast.literal_eval.
    Returns None if parsing fails. Calculates the mean if successful.
    """
    if isinstance(x, list): # Already a list
         try:
             # Ensure items are numeric and calculate mean
             return np.mean([float(item) for item in x])
         except (ValueError, TypeError):
             return np.nan # List contains non-numeric items
    elif isinstance(x, str):
        x = x.strip()
        # Attempt basic fixes (e.g., missing brackets)
        if not x.startswith('['): x = '[' + x
        if not x.endswith(']'): x = x + ']'
        try:
            parsed_list = ast.literal_eval(x)
            if isinstance(parsed_list, list):
                 # Ensure items are numeric and calculate mean
                 return np.mean([float(item) for item in parsed_list])
            else:
                 return np.nan # Parsed, but not into a list
        except (ValueError, SyntaxError, TypeError):
            # print(f"Could not parse list string: {x}") # Optional debug
            return np.nan # Parsing failed
    # Handle other types or NaN input
    return np.nan


def reconstruction_loss(y_true, y_pred):
    """Categorical cross-entropy reconstruction loss for the VAE."""
    # Clip predictions to prevent log(0) issues which result in NaN loss
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    # Calculate categorical crossentropy
    recon_loss = losses.categorical_crossentropy(y_true, y_pred)
    # Sum over the sequence length dimension and take the mean over the batch
    return K.mean(K.sum(recon_loss, axis=-1))

def sampling(args):
    """Reparameterization trick by sampling from learned normal distribution."""
    z_mean_sample, z_log_var_sample = args
    batch = K.shape(z_mean_sample)[0]
    dim = K.int_shape(z_mean_sample)[1]
    # By default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean_sample + K.exp(0.5 * z_log_var_sample) * epsilon

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


# --- Metadata Parsing and Feature Engineering ---
print("3. Parsing and Processing Metadata...")

# Process 'molar extinction coefficient' (expected: 2 values)
print("   - Processing 'molar extinction coefficient'...")
split_data_molar = df['molar extinction coefficient'].apply(lambda x: safe_split_pair(x, expected_len=2))
split_df_molar = pd.DataFrame(split_data_molar.tolist(), index=df.index, columns=['molar_extinct1', 'molar_extinct2'])
df = pd.concat([df, split_df_molar], axis=1)

# Process 'ss' (secondary structure, expected: 3 values)
print("   - Processing 'ss' (secondary structure)...")
split_data_ss = df['ss'].apply(lambda x: safe_split_pair(x, expected_len=3))
split_df_ss = pd.DataFrame(split_data_ss.tolist(), index=df.index, columns=['alpha', 'beta', 'random_coil'])
df = pd.concat([df, split_df_ss], axis=1)

# Process 'flexibility' (expected: list of floats -> calculate mean)
print("   - Processing 'flexibility'...")
df['flexibility_mean'] = df['flexibility'].apply(fix_and_parse_list_string)

# --- Final Filtering based on All Required Data ---
print("4. Final Filtering based on Sequences, Metadata, and Target...")
# Define all columns required for the models
meta_features_cols = ['mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
                      'molar_extinct1', 'molar_extinct2', # From 'molar extinction coefficient'
                      'alpha', 'beta', 'random_coil',    # From 'ss'
                      'flexibility_mean']                # From 'flexibility'
target_col = 'instability index'
all_required_cols = ['seq'] + meta_features_cols + [target_col]

# Drop rows where *any* of the required columns have NaN
initial_rows_meta = len(df)
df.dropna(subset=all_required_cols, inplace=True)
print(f"   - Dropped {initial_rows_meta - len(df)} rows due to NaNs in required features or target.")
print(f"Rows remaining after final filtering: {len(df)}")

# Reset index after final filtering to ensure it's contiguous (0 to N-1)
df.reset_index(drop=True, inplace=True)

# --- Data Extraction and Preparation for Models ---
print("5. Extracting and Preparing Data for Models...")

# Extract cleaned sequences, metadata, and targets
sequences_clean = df['seq'].tolist()
metadata = df[meta_features_cols].values
targets = df[target_col].values
n_meta_features = metadata.shape[1]

# --- Median values for target metadata  ---

print("Calculating median metadata for stable proteins...")

# Define the stability threshold
STABILITY_THRESHOLD = 40 

# Define the columns for which you need median values (must match meta_features_cols later!)

cols_for_median = [
    'mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
    'molar_extinct1', 'molar_extinct2',
    'alpha', 'beta', 'random_coil',
    'flexibility_mean'
]
target_col_name = 'instability index' 

# Filter the DataFrame for stable proteins
stable_df = df[df[target_col_name] < STABILITY_THRESHOLD].copy()

if stable_df.empty:
    print(f"Warning: No proteins found with instability < {STABILITY_THRESHOLD}. Check data or threshold.")

    median_stable_meta = df[cols_for_median].median()
    print("Using OVERALL median metadata as fallback.")
else:
    # Calculate the median for each column
    median_stable_meta = stable_df[cols_for_median].median()
    print(f"Median metadata for proteins with instability < {STABILITY_THRESHOLD}:")
    print(median_stable_meta)

target_metadata_example = median_stable_meta.tolist()

meta_features_cols = cols_for_median 

assert len(target_metadata_example) == len(meta_features_cols), \
    f"Length mismatch: target_metadata ({len(target_metadata_example)}) vs meta_features_cols ({len(meta_features_cols)})"


# --- Sequence Encoding (AFTER final filtering) ---
print("   - Encoding Sequences...")
# Define amino acid vocabulary and mapping using the *cleaned* sequences
aa_vocab = sorted(list(set(''.join(sequences_clean))))
aa_to_idx = {aa: i + 1 for i, aa in enumerate(aa_vocab)} # 0 reserved for padding
idx_to_aa = {v: k for k, v in aa_to_idx.items()}
vocab_size = len(aa_vocab) + 1 # Include padding

# Convert sequences to padded integer arrays
sequences_int = [[aa_to_idx.get(aa, 0) for aa in seq] for seq in sequences_clean] # Use get for safety
padded_seqs = utils.pad_sequences(sequences_int, maxlen=MAX_LEN, padding='post', truncating='post')

# One-hot encode sequences
onehot_seqs = utils.to_categorical(padded_seqs, num_classes=vocab_size)

# --- Metadata Scaling ---
print("   - Scaling Metadata...")
scaler = StandardScaler()
scaled_meta = scaler.fit_transform(metadata)

# --- Target Variable Processing (Check for Infs, NaNs handled by dropna earlier) ---
if np.isinf(targets).sum() > 0:
     print(f"   - Found {np.isinf(targets).sum()} Infs in target. Replacing with 0.")
     targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0 or median/mean

# --- Final Shape Verification ---
print(f"Final data shapes: Sequences={onehot_seqs.shape}, Metadata={scaled_meta.shape}, Targets={targets.shape}")
assert onehot_seqs.shape[0] == scaled_meta.shape[0] == targets.shape[0], "Data dimensions mismatch!"

# --- Train/Test Split ---
print("6. Splitting data into Train/Test sets...")
# Now split the fully aligned and cleaned data
X_seq_train, X_seq_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    onehot_seqs, scaled_meta, targets, test_size=TEST_SPLIT_RATIO, random_state=42)

print(f"Train set size: {X_seq_train.shape[0]}")
print(f"Test set size: {X_seq_test.shape[0]}")

# --- Data Validation Checks (Post-Split) ---
print("7. Final data validation checks (NaNs/Infs in splits)...")
datasets = {'X_seq_train': X_seq_train, 'X_meta_train': X_meta_train, 'y_train': y_train,
            'X_seq_test': X_seq_test, 'X_meta_test': X_meta_test, 'y_test': y_test}
for name, data in datasets.items():
    nans = np.isnan(data).sum()
    infs = np.isinf(data).sum()
    if nans > 0 or infs > 0:
        print(f"   WARNING: Found {nans} NaNs / {infs} Infs in {name} AFTER split. Applying nan_to_num.")
        datasets[name] = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

X_seq_train, X_meta_train, y_train = datasets['X_seq_train'], datasets['X_meta_train'], datasets['y_train']
X_seq_test, X_meta_test, y_test = datasets['X_seq_test'], datasets['X_meta_test'], datasets['y_test']


### Stability Predictor Model ###
print("\n8. Building and Training Stability Predictor...")

# --- Model Definition ---
seq_input_pred = layers.Input(shape=(MAX_LEN, vocab_size), name="pred_seq_input")
meta_input_pred = layers.Input(shape=(n_meta_features,), name="pred_meta_input")

# Sequence branch
x_pred = layers.Conv1D(128, 5, activation='relu')(seq_input_pred)
x_pred = layers.MaxPooling1D(2)(x_pred)
x_pred = layers.Conv1D(256, 3, activation='relu')(x_pred)
x_pred = layers.GlobalMaxPooling1D()(x_pred)

# Metadata branch
y_pred = layers.Dense(64, activation='relu')(meta_input_pred)

# Combined
combined_pred = layers.concatenate([x_pred, y_pred])
z_pred = layers.Dense(128, activation='relu')(combined_pred)
output_pred = layers.Dense(1, name="predictor_output")(z_pred) # Output is single value (instability index)

predictor = models.Model([seq_input_pred, meta_input_pred], output_pred, name="stability_predictor")
predictor.compile(optimizer=Adam(), loss='mse', metrics=['mae']) # Mean Squared Error for regression

# --- Training ---
print("   - Training predictor...")
history_predictor = predictor.fit(
    [X_seq_train, X_meta_train], y_train,
    validation_data=([X_seq_test, X_meta_test], y_test),
    epochs=EPOCHS_PREDICTOR,
    batch_size=BATCH_SIZE,
    verbose=1 # Shows progress
)

# --- Evaluation ---
loss, mae = predictor.evaluate([X_seq_test, X_meta_test], y_test, verbose=0)
print(f"   - Predictor Test MAE: {mae:.4f}")


### Conditional Variational Autoencoder (CVAE) ###
print("\n9. Building and Training CVAE...")

# --- Encoder Definition ---
seq_input_cvae = layers.Input(shape=(MAX_LEN, vocab_size), name="cvae_seq_input")
meta_input_cvae = layers.Input(shape=(n_meta_features,), name="cvae_meta_input")

# Combined input processing for encoder
x_enc = layers.Conv1D(128, 5, activation='relu', padding='same')(seq_input_cvae)
x_enc = layers.MaxPooling1D(2)(x_enc) # -> MAX_LEN / 2
x_enc = layers.Conv1D(256, 3, activation='relu', padding='same')(x_enc)
x_enc = layers.MaxPooling1D(2)(x_enc) # -> MAX_LEN / 4
x_enc = layers.Flatten()(x_enc)

# Concatenate flattened sequence features with metadata
concat_enc = layers.concatenate([x_enc, meta_input_cvae])
hidden_enc = layers.Dense(256, activation='relu')(concat_enc) # Intermediate dense layer

# Latent space parameters
z_mean = layers.Dense(LATENT_DIM, name='z_mean')(hidden_enc)
z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(hidden_enc)

# Reparameterization trick layer
z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

# Standalone Encoder Model (optional, useful for analysis)
encoder = models.Model([seq_input_cvae, meta_input_cvae], [z_mean, z_log_var, z], name="encoder")

# --- Decoder Definition ---
decoder_input_latent = layers.Input(shape=(LATENT_DIM,), name="decoder_latent_input")
decoder_input_meta = layers.Input(shape=(n_meta_features,), name="decoder_meta_input")

# Concatenate latent vector and metadata for conditioning
concat_dec = layers.concatenate([decoder_input_latent, decoder_input_meta])

# Calculate the shape needed before reshaping (depends on ConvTranspose layers)
decoder_dense_units = (MAX_LEN // 4) * 256 # Example filter size

x_dec = layers.Dense(decoder_dense_units, activation='relu')(concat_dec)
x_dec = layers.Reshape((MAX_LEN // 4, 256))(x_dec) # Reshape to (timesteps, features)

# Upsample using Transposed Convolutions
x_dec = layers.Conv1DTranspose(256, 5, strides=2, padding='same', activation='relu')(x_dec) # -> MAX_LEN / 2
x_dec = layers.Conv1DTranspose(128, 5, strides=2, padding='same', activation='relu')(x_dec)  # -> MAX_LEN

# Final layer to get back to vocab dimension with softmax activation
decoder_output = layers.Conv1D(vocab_size, 3, activation='softmax', padding='same', name='decoder_output')(x_dec)

# Standalone Decoder Model
decoder = models.Model([decoder_input_latent, decoder_input_meta], decoder_output, name="decoder")

# --- End-to-End CVAE Model ---
# Connect encoder output (z) and original metadata input to the decoder
cvae_outputs = decoder([z, meta_input_cvae])
cvae = models.Model([seq_input_cvae, meta_input_cvae], cvae_outputs, name="cvae")

# --- CVAE Loss Calculation ---
# 1. Reconstruction Loss (defined as a function)
# 2. KL Divergence Loss
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
kl_loss *= KL_WEIGHT # Apply weight
cvae.add_loss(K.mean(kl_loss)) # Add KL loss scaled by weight
cvae.add_metric(K.mean(kl_loss), name='kl_loss') # Track KL loss during training

# --- Compile CVAE ---
# Use only reconstruction loss here, as KL loss is added via add_loss
cvae.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
             loss=reconstruction_loss,
             metrics=[reconstruction_loss]) # Track recon loss explicitly too

# --- Callbacks ---
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=0.001, # Minimum change to qualify as an improvement
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5, # Reduce LR by half
    patience=REDUCE_LR_PATIENCE,
    min_lr=1e-6, # Don't reduce below this
    verbose=1
)

# --- Train CVAE ---
print("   - Training CVAE...")
history_cvae = cvae.fit(
    [X_seq_train, X_meta_train], X_seq_train, # Input seq/meta, target is seq
    epochs=EPOCHS_CVAE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT_RATIO, # Use part of training data for validation
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

### Sequence Generation ###
print("\n10. Generating Sequences...")

def generate_sequences(n_samples, target_meta_unscaled, decoder_model, scaler_obj, latent_dim, idx_to_aa_map, aa_to_idx_map, max_len, vocab_size_):
    """
    Generates sequences using the decoder based on target metadata.

    Args:
        n_samples (int): Number of sequences to generate.
        target_meta_unscaled (list or np.array): Single sample of target metadata features (unscaled).
        decoder_model (tf.keras.Model): The trained decoder model.
        scaler_obj (sklearn.preprocessing.StandardScaler): The fitted scaler for metadata.
        latent_dim (int): The dimension of the latent space.
        idx_to_aa_map (dict): Dictionary mapping indices back to amino acids.
        aa_to_idx_map (dict): Dictionary mapping amino acids to indices.
        max_len (int): Max sequence length for padding.
        vocab_size_ (int): Vocabulary size including padding.


    Returns:
        list: A list of generated amino acid sequences (strings).
        np.array: The one-hot encoded generated sequences.
        np.array: The scaled metadata used for generation (repeated).
    """
    # Sample from the prior distribution (standard normal) in the latent space
    latent_samples = np.random.normal(size=(n_samples, latent_dim))

    # Scale the target metadata
    target_meta_unscaled = np.array(target_meta_unscaled).reshape(1, -1)
    if np.isnan(target_meta_unscaled).any():
        raise ValueError("Target metadata contains NaN values before scaling")
    scaled_target_meta = scaler_obj.transform(target_meta_unscaled)

    # Repeat the scaled metadata for each sample
    scaled_meta_batch = np.repeat(scaled_target_meta, n_samples, axis=0)

    if np.isnan(scaled_meta_batch).any():
        print("Warning: Scaled metadata contains NaNs. Check scaler and input.")
        scaled_meta_batch = np.nan_to_num(scaled_meta_batch, nan=0.0) # Example fix

    # Generate sequences using the decoder
    generated_onehot = decoder_model.predict([latent_samples, scaled_meta_batch])

    # Convert one-hot encoded sequences back to amino acid strings
    generated_indices = np.argmax(generated_onehot, axis=-1)
    generated_sequences = []
    for seq_indices in generated_indices:
        # Map indices to amino acids, ignoring padding (index 0)
        sequence = ''.join([idx_to_aa_map.get(idx, '') for idx in seq_indices if idx != 0])
        generated_sequences.append(sequence)

    # Also return the generated one-hot sequences and scaled metadata for predictor input
    # Re-create one-hot from generated sequences to ensure consistency if needed,
    # or directly use the output from decoder if confident.
    # Let's re-create to be safe, handling potential length differences if any
    gen_sequences_int = [[aa_to_idx_map.get(aa, 0) for aa in seq] for seq in generated_sequences]
    gen_padded_seqs = utils.pad_sequences(gen_sequences_int, maxlen=max_len, padding='post')
    gen_onehot_seqs = utils.to_categorical(gen_padded_seqs, num_classes=vocab_size_)


    # return generated_sequences, generated_onehot, scaled_meta_batch # Option 1: Use direct decoder output
    return generated_sequences, gen_onehot_seqs, scaled_meta_batch # Option 2: Re-encode


# --- Example Generation ---

assert len(target_metadata_example) == n_meta_features, \
    f"Length mismatch: target_metadata has {len(target_metadata_example)}, expected {n_meta_features}"

num_sequences_to_generate = 50
generated_seqs, gen_onehot_for_pred, gen_meta_for_pred = generate_sequences(
    num_sequences_to_generate,
    target_metadata_example,
    decoder,
    scaler,
    LATENT_DIM,
    idx_to_aa,
    aa_to_idx,   # Pass the aa_to_idx map
    MAX_LEN,     # Pass MAX_LEN
    vocab_size   # Pass vocab_size
)

print(f"\nGenerated {len(generated_seqs)} initial sequences.")


### Filtering Generated Sequences ###
print("\n11. Filtering Generated Sequences by Predicted Stability...")

# Predict instability index for generated sequences using the prepared inputs
predicted_stabilities = predictor.predict([gen_onehot_for_pred, gen_meta_for_pred])

print(predicted_stabilities)
# Filter based on a stability threshold
stability_threshold = 40
stable_sequences_indices = [i for i, score in enumerate(predicted_stabilities) if score < stability_threshold]
stable_sequences = [generated_seqs[i] for i in stable_sequences_indices]
stable_scores = [predicted_stabilities[i][0] for i in stable_sequences_indices]

print(f"\nFound {len(stable_sequences)} sequences predicted to be stable (Instability Index < {stability_threshold}):")

top_stable_sequences = []

for i, seq in enumerate(stable_sequences[:5]): # Display top 5
    print(f"  Score: {stable_scores[i]:.2f} | Seq: {seq[:60]}...")
    top_stable_sequences.append(f"{stable_scores[i]:.2f} | Seq: {seq}")

    print(top_stable_sequences)


with open('fungal_sequences.txt', 'w') as file:
    for sequence in top_stable_sequences:
        file.write(f"{sequence}\n")
