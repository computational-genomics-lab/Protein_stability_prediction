# phase2_modified.py
# Phase 2: Guided generation and ranking using models saved from phase 1.
# Expects the following files in working directory:
# - fungal_sequences.csv            (training sequences; used to reconstruct vocab & medians)
# - experimental_peptides.csv       (rows guiding generation)
# - encoder_7k.h5
# - decoder_7k.h5
# - predictor_7k.h5
# - cvae_7k.h5       (optional)
# - meta_scaler_7k.pkl
#
# The script will:
# - attempt to import helper functions & globals from cellulase_generator.py if available
# - otherwise reconstruct helpers locally (robust to minor differences)
# - for each experimental peptide row: encode to latent anchor, sample latents nearby,
#   decode to sequences, predict instability, filter, re-encode, compute composite scores,
#   select diverse top candidates, and save CSV / FASTA outputs.
#
# Outputs (per experimental row i):
# - top_guided_candidates_anchor_i.csv
# - top_guided_candidates_anchor_i.fasta
# - all_stable_candidates_anchor_i_scored.csv

import os
import sys
import ast
import math
import logging
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.models import load_model
from tensorflow.keras import utils as kutils
from tensorflow.keras import backend as K

# ---------- Configuration ----------
TRAINING_CSV = "Fungal_sequences.csv"
EXP_CSV = "experimental_peptides.csv"

ENCODER_PTH = "encoder_7k.h5"
DECODER_PTH = "decoder_7k.h5"
PREDICTOR_PTH = "predictor_7k.h5"
CVAE_PTH = "cvae_7k.h5"       # optional; not strictly required for generation (decoder is enough)
SCALER_PTH = "meta_scaler_7k.pkl"

MAX_LEN_DEFAULT = 500
STABILITY_THRESHOLD = 40.0
N_PER_ANCHOR = 1500
INTERP_STEPS = 25
TOP_K = 10
CLUSTER_MIN = 3
CLUSTER_MAX = 12
RANDOM_STATE = 42
BATCH_DECODE = 128
BATCH_PRED = 128
LOGLEVEL = logging.INFO

# Metadata columns expected (must match Phase 1 order)
META_FEATURE_COLS = [
    'mol_wt', 'aromaticity', 'pi', 'chrg', 'gravy',
    'molar_extinct1', 'molar_extinct2',
    'alpha', 'beta', 'random_coil',
    'flexibility_mean'
]

# Columns to parse from experimental CSV if present as raw strings
RAW_MOLAR_COL = 'molar extinction coefficient'
RAW_SS_COL = 'ss'
RAW_FLEX_COL = 'flexibility'
TARGET_COL = 'instability index'  # only in training CSV

# ---------- Logging ----------
logging.basicConfig(level=LOGLEVEL, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("phase2")

# ---------- Try to import helpers from cellulase_generator.py ----------
safe_split_pair = None
fix_and_parse_list_string = None
generate_sequences_helper = None
aa_to_idx_global = None
idx_to_aa_global = None
vocab_size_global = None
MAX_LEN = None
meta_features_cols_from_training = None
target_metadata_example_from_training = None

try:
    import importlib
    mod = importlib.import_module("cellulase_generator")
    safe_split_pair = getattr(mod, "safe_split_pair", None)
    fix_and_parse_list_string = getattr(mod, "fix_and_parse_list_string", None)
    generate_sequences_helper = getattr(mod, "generate_sequences", None)
    aa_to_idx_global = getattr(mod, "aa_to_idx", None)
    idx_to_aa_global = getattr(mod, "idx_to_aa", None)
    vocab_size_global = getattr(mod, "vocab_size", None)
    MAX_LEN = getattr(mod, "MAX_LEN", None)
    meta_features_cols_from_training = getattr(mod, "meta_features_cols", None)
    target_metadata_example_from_training = getattr(mod, "target_metadata_example", None)
    logger.info("Imported helpers/globals from cellulase_generator.py")
except Exception as e:
    logger.info("Could not import cellulase_generator.py or some attributes; building local helpers. (%s)", str(e))

# ---------- Fallback helper implementations ----------
def _safe_split_pair_local(value, expected_len):
    default_return = [np.nan] * expected_len
    if pd.isna(value):
        return default_return
    parsed_value = None
    if isinstance(value, (list, tuple)):
        parsed_value = value
    elif isinstance(value, str):
        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            return default_return
    if isinstance(parsed_value, (list, tuple)):
        if len(parsed_value) == expected_len:
            try:
                return [float(item) for item in parsed_value]
            except Exception:
                return default_return
        else:
            return default_return
    return default_return

def _fix_and_parse_list_string_local(x):
    if isinstance(x, list):
        try:
            return np.mean([float(item) for item in x])
        except Exception:
            return np.nan
    elif isinstance(x, str):
        xs = x.strip()
        if not xs.startswith('['):
            xs = '[' + xs
        if not xs.endswith(']'):
            xs = xs + ']'
        try:
            parsed_list = ast.literal_eval(xs)
            if isinstance(parsed_list, list):
                try:
                    return np.mean([float(item) for item in parsed_list])
                except Exception:
                    return np.nan
            else:
                return np.nan
        except Exception:
            return np.nan
    return np.nan

# choose the functions (prefer imported ones)
safe_split_pair = safe_split_pair if safe_split_pair is not None else _safe_split_pair_local
fix_and_parse_list_string = fix_and_parse_list_string if fix_and_parse_list_string is not None else _fix_and_parse_list_string_local

# ---------- Utility functions ----------
def one_hot_encode_sequence(seq: str, aa_to_idx_map: dict, max_len: int, vocab_size_local: int):
    seq = "" if seq is None else str(seq)
    seq_int = [aa_to_idx_map.get(aa, 0) for aa in seq]
    padded = kutils.pad_sequences([seq_int], maxlen=max_len, padding='post', truncating='post')[0]
    onehot = kutils.to_categorical(padded, num_classes=vocab_size_local)
    return onehot

def decode_onehot_to_sequence(onehot_array: np.ndarray, idx_to_aa_map: dict):
    # Accept (timesteps, vocab) or (1, timesteps, vocab)
    if onehot_array.ndim == 3 and onehot_array.shape[0] == 1:
        onehot_array = onehot_array[0]
    indices = np.argmax(onehot_array, axis=-1)
    seq = ''.join([idx_to_aa_map.get(int(i), '') for i in indices if int(i) != 0])
    return seq

def unpack_encoder_output(enc_out):
    # encoder may return [z_mean, z_log_var, z] or [z_mean, z] or just z
    if isinstance(enc_out, (list, tuple)):
        if len(enc_out) >= 3:
            return enc_out[0], enc_out[1], enc_out[2]
        elif len(enc_out) == 2:
            return enc_out[0], np.zeros_like(enc_out[0]), enc_out[1]
        else:
            return enc_out[0], np.zeros_like(enc_out[0]), enc_out[0]
    else:
        return enc_out, np.zeros_like(enc_out), enc_out

def normalize_array(a):
    a = np.array(a, dtype=float)
    if a.size == 0:
        return a
    if np.nanmax(a) == np.nanmin(a):
        return np.ones_like(a) * 0.5
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

# ---------- Load training CSV to reconstruct vocabulary and medians ----------
if not os.path.exists(TRAINING_CSV):
    logger.warning("Training CSV '%s' not found. Vocabulary will fall back to standard 20 amino acids.", TRAINING_CSV)
    # fallback alphabet
    aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i + 1 for i, aa in enumerate(sorted(aa_alphabet))}
    idx_to_aa = {v: k for k, v in aa_to_idx.items()}
    vocab_size = len(aa_to_idx) + 1
    MAX_LEN = MAX_LEN_DEFAULT if MAX_LEN is None else MAX_LEN
    df_train = None
else:
    df_train = pd.read_csv(TRAINING_CSV)
    # basic clean consistent with training script
    df_train = df_train.dropna(subset=['seq'])
    df_train = df_train[df_train['seq'].apply(lambda x: isinstance(x, str))]
    sequences_clean = df_train['seq'].astype(str).tolist()
    aa_vocab = sorted(list(set(''.join(sequences_clean))))
    aa_to_idx = aa_to_idx_global if aa_to_idx_global is not None else {aa: i + 1 for i, aa in enumerate(aa_vocab)}
    idx_to_aa = idx_to_aa_global if idx_to_aa_global is not None else {v: k for k, v in aa_to_idx.items()}
    vocab_size = vocab_size_global if vocab_size_global is not None else len(aa_vocab) + 1
    MAX_LEN = MAX_LEN if MAX_LEN is not None else (MAX_LEN_DEFAULT)
    logger.info("Reconstructed vocabulary: %d amino acids (+padding). MAX_LEN=%d", vocab_size - 1, MAX_LEN)

# If meta_features_cols exported from training, prefer it
if meta_features_cols_from_training is not None:
    META_FEATURE_COLS = meta_features_cols_from_training

# ---------- Load models and scaler ----------
if not os.path.exists(ENCODER_PTH):
    raise FileNotFoundError(f"Encoder model not found at {ENCODER_PTH}")
if not os.path.exists(DECODER_PTH):
    raise FileNotFoundError(f"Decoder model not found at {DECODER_PTH}")
if not os.path.exists(PREDICTOR_PTH):
    raise FileNotFoundError(f"Predictor model not found at {PREDICTOR_PTH}")
if not os.path.exists(SCALER_PTH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PTH}")

logger.info("Loading models...")
encoder = load_model(ENCODER_PTH, compile=False)
decoder = load_model(DECODER_PTH, compile=False)
predictor = load_model(PREDICTOR_PTH, compile=False)
cvae = None
if os.path.exists(CVAE_PTH):
    try:
        cvae = load_model(CVAE_PTH, compile=False)
    except Exception:
        cvae = None
        logger.info("Could not load CVAE (this is optional). Proceeding with decoder only.")
scaler = joblib.load(SCALER_PTH)
logger.info("Models and scaler loaded.")

# ---------- Load experimental CSV and preprocess metadata fields ----------
if not os.path.exists(EXP_CSV):
    raise FileNotFoundError(f"Experimental CSV not found: {EXP_CSV}")
exp_df = pd.read_csv(EXP_CSV)
exp_df = exp_df.loc[:, ~exp_df.columns.str.contains('^Unnamed')]
exp_df = exp_df.dropna(subset=['seq'])

# If experimental CSV has raw columns like 'molar extinction coefficient', 'ss', 'flexibility', parse them
if RAW_MOLAR_COL in exp_df.columns:
    # parse molar extinction into two columns
    parsed = exp_df[RAW_MOLAR_COL].apply(lambda x: safe_split_pair(x, expected_len=2))
    p_df = pd.DataFrame(parsed.tolist(), index=exp_df.index, columns=['molar_extinct1', 'molar_extinct2'])
    exp_df = pd.concat([exp_df, p_df], axis=1)
if RAW_SS_COL in exp_df.columns:
    parsed_ss = exp_df[RAW_SS_COL].apply(lambda x: safe_split_pair(x, expected_len=3))
    pss = pd.DataFrame(parsed_ss.tolist(), index=exp_df.index, columns=['alpha', 'beta', 'random_coil'])
    exp_df = pd.concat([exp_df, pss], axis=1)
if RAW_FLEX_COL in exp_df.columns:
    exp_df['flexibility_mean'] = exp_df[RAW_FLEX_COL].apply(fix_and_parse_list_string)

# Ensure META_FEATURE_COLS exist in exp_df; if not, try to fill from training median
missing_meta_cols = [c for c in META_FEATURE_COLS if c not in exp_df.columns]
if missing_meta_cols:
    logger.warning("Experimental CSV missing metadata columns: %s", missing_meta_cols)
    # attempt to fill from training median
    if df_train is not None and set(META_FEATURE_COLS).issubset(df_train.columns):
        med = df_train[META_FEATURE_COLS].median()
        for c in missing_meta_cols:
            exp_df[c] = med[c]
        logger.info("Filled missing experimental metadata columns from training medians.")
    else:
        # fallback fill with zeros
        for c in missing_meta_cols:
            exp_df[c] = 0.0
        logger.info("Filled missing experimental metadata columns with zeros (no training medians available).")

# ---------- Compute global stable target metadata (unscaled) ----------
if df_train is not None and set(META_FEATURE_COLS).issubset(df_train.columns) and TARGET_COL in df_train.columns:
    stable_rows = df_train[df_train[TARGET_COL] < STABILITY_THRESHOLD]
    if len(stable_rows) > 0:
        target_meta_unscaled = stable_rows[META_FEATURE_COLS].median().values.reshape(1, -1)
        logger.info("Using median metadata of %d stable training proteins as target meta.", len(stable_rows))
    else:
        target_meta_unscaled = df_train[META_FEATURE_COLS].median().values.reshape(1, -1)
        logger.info("No stable proteins in training set; using overall median metadata.")
elif df_train is not None and set(META_FEATURE_COLS).issubset(df_train.columns):
    target_meta_unscaled = df_train[META_FEATURE_COLS].median().values.reshape(1, -1)
    logger.info("Training instability column missing; using overall median metadata as target.")
else:
    # fallback: median of experimental metadata if present
    target_meta_unscaled = exp_df[META_FEATURE_COLS].median().values.reshape(1, -1)
    logger.info("Using experimental median metadata as fallback target metadata.")

# Try to scale target metadata with loaded scaler
try:
    target_meta_scaled = scaler.transform(target_meta_unscaled)
except Exception as e:
    logger.warning("Scaler transform failed for target meta: %s. Attempting to refit a scaler from training metadata if possible.", str(e))
    if df_train is not None and set(META_FEATURE_COLS).issubset(df_train.columns):
        from sklearn.preprocessing import StandardScaler
        scaler_local = StandardScaler().fit(df_train[META_FEATURE_COLS].values)
        target_meta_scaled = scaler_local.transform(target_meta_unscaled)
        logger.info("Refit local scaler from training metadata as fallback.")
    else:
        target_meta_scaled = np.zeros_like(target_meta_unscaled)
        logger.info("Using zeros for scaled metadata (no training metadata available).")

# ---------- Helper to compute sigma for guided sampling ----------
def compute_sigma_for_anchor(anchor_vec: np.ndarray, sample_size: int = 200, min_sigma=0.08, max_sigma=0.5) -> float:
    # If training data available, use encoder on sample to estimate median pairwise distance
    try:
        if df_train is None or not set(META_FEATURE_COLS).issubset(df_train.columns):
            return 0.25  # conservative default
        sample_seqs = df_train['seq'].dropna().astype(str).sample(n=min(sample_size, len(df_train)), random_state=RANDOM_STATE)
        sample_meta = df_train.loc[sample_seqs.index, META_FEATURE_COLS].fillna(method='ffill').values
        sample_oh = np.stack([one_hot_encode_sequence(s, aa_to_idx, MAX_LEN, vocab_size) for s in sample_seqs], axis=0)
        sample_meta_scaled = scaler.transform(sample_meta)
        enc_out = encoder.predict([sample_oh, sample_meta_scaled], batch_size=64)
        _, _, z_sample = unpack_encoder_output(enc_out)
        if z_sample.shape[0] > 1:
            pairwise = cdist(z_sample, z_sample, metric='euclidean')
            nonzero = pairwise[np.triu_indices_from(pairwise, k=1)]
            if len(nonzero) == 0:
                return 0.25
            median_anchor_dist = float(np.median(nonzero))
            sigma = float(min(max(median_anchor_dist * 0.2, min_sigma), max_sigma))
            return sigma
    except Exception as e:
        logger.debug("Failed to compute sigma from training sample: %s", str(e))
    return 0.25

# ---------- Main generation loop for each experimental row ----------
for idx, row in exp_df.reset_index(drop=True).iterrows():
    try:
        logger.info("Processing experimental row %d", idx)
        seq = str(row['seq'])
        # Build metadata unscaled vector for this row (fallback to target if missing)
        meta_unscaled_row = np.array([row.get(c, np.nan) for c in META_FEATURE_COLS]).reshape(1, -1)
        if np.isnan(meta_unscaled_row).any():
            logger.debug("Row %d has NaNs in metadata; using target_meta_unscaled as fallback.", idx)
            meta_unscaled_row = target_meta_unscaled.copy()
        # Scale row metadata for encoding anchor (prefer row metadata; if fails, use target scaled)
        try:
            meta_scaled_row = scaler.transform(meta_unscaled_row)
        except Exception as e:
            logger.debug("Scaling row metadata failed (%s); using target_meta_scaled.", str(e))
            meta_scaled_row = target_meta_scaled.copy()

        # One-hot encode experimental sequence
        oh_seq = one_hot_encode_sequence(seq, aa_to_idx, MAX_LEN, vocab_size)
        oh_seq_batch = np.expand_dims(oh_seq, axis=0)

        # Encode to latent anchor
        enc_out = encoder.predict([oh_seq_batch, meta_scaled_row], batch_size=1)
        z_mean_row, z_logvar_row, z_row = unpack_encoder_output(enc_out)
        # anchor = z_row.reshape(-1)
        anchor = np.asarray(z_mean_row).reshape(-1)
        latent_dim = anchor.shape[0]
        logger.info("Anchor latent dim: %d", latent_dim)

        # Compute sigma guided by training latent spread if possible
        sigma = compute_sigma_for_anchor(anchor)
        logger.info("Using sigma=%0.4f for sampling around anchor %d", sigma, idx)

        # Sample latents around anchor
        n_samples = N_PER_ANCHOR
        noise = np.random.normal(loc=0.0, scale=sigma, size=(n_samples, latent_dim))
        guided_latents = anchor.reshape(1, -1) + noise  # (n_samples, latent_dim)

        # Prepare metadata batch for decoder: use target_meta_scaled to bias towards stable chemotype
        scaled_meta_batch = np.repeat(target_meta_scaled, guided_latents.shape[0], axis=0)

        # Decode latents -> probability distributions (one-hot like)
        logger.info("Decoding %d latents via decoder...", guided_latents.shape[0])
        decoded_onehot = decoder.predict([guided_latents, scaled_meta_batch], batch_size=BATCH_DECODE, verbose=0)

        # Convert decoded outputs to sequences (use argmax per position)
        generated_seqs = [decode_onehot_to_sequence(onehot, idx_to_aa) for onehot in decoded_onehot]
        # Deduplicate while preserving order
        unique_ordered = list(OrderedDict.fromkeys(generated_seqs))
        logger.info("Decoded %d sequences -> %d unique sequences after dedup.", len(generated_seqs), len(unique_ordered))

        if len(unique_ordered) == 0:
            logger.warning("No sequences decoded for experimental row %d. Skipping.", idx)
            continue

        # Re-encode for predictor input
        gen_onehot_for_pred = np.stack([one_hot_encode_sequence(s, aa_to_idx, MAX_LEN, vocab_size) for s in unique_ordered], axis=0)
        gen_meta_for_pred = np.repeat(target_meta_scaled, len(unique_ordered), axis=0)

        # Predict instability
        logger.info("Predicting instability for %d candidates...", len(unique_ordered))
        predicted_inst = predictor.predict([gen_onehot_for_pred, gen_meta_for_pred], batch_size=BATCH_PRED).reshape(-1)
        # Keep those below threshold
        stable_mask = predicted_inst < STABILITY_THRESHOLD
        stable_indices = np.where(stable_mask)[0]

        # If none found, relax slightly
        if len(stable_indices) == 0:
            relaxed = STABILITY_THRESHOLD * 1.1
            stable_indices = np.where(predicted_inst < relaxed)[0]
            logger.info("No candidates under threshold %0.2f; relaxed to %0.2f, found %d.", STABILITY_THRESHOLD, relaxed, len(stable_indices))

        if len(stable_indices) == 0:
            logger.warning("No stable candidates found for row %d after relaxation. Continuing.", idx)
            continue

        stable_seqs = [unique_ordered[i] for i in stable_indices]
        stable_scores = predicted_inst[stable_indices]

        # Re-encode stable candidates into latent space for ranking/diversity
        gen_onehot_for_stable = np.stack([one_hot_encode_sequence(s, aa_to_idx, MAX_LEN, vocab_size) for s in stable_seqs], axis=0)
        gen_meta_for_stable = np.repeat(target_meta_scaled, len(stable_seqs), axis=0)
        enc_out_stable = encoder.predict([gen_onehot_for_stable, gen_meta_for_stable], batch_size=64)
        _, _, z_stable = unpack_encoder_output(enc_out_stable)

        # Compute distances to anchor and normalized scores
        dist_to_anchor = cdist(z_stable, anchor.reshape(1, -1), metric='euclidean').reshape(-1)
        norm_inst = 1.0 - normalize_array(stable_scores)   # higher is better
        norm_prox = 1.0 - normalize_array(dist_to_anchor)  # higher is better
        w_inst, w_prox = 0.7, 0.3
        composite_scores = w_inst * norm_inst + w_prox * norm_prox

        # Clustering for diversity
        n_clusters = min(CLUSTER_MAX, max(CLUSTER_MIN, len(stable_seqs) // 10))
        if len(z_stable) < n_clusters:
            n_clusters = max(1, len(z_stable))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, batch_size=256)
        clusters = kmeans.fit_predict(z_stable)
        cluster_indices = {c: [] for c in range(n_clusters)}
        for ii, cl in enumerate(clusters):
            cluster_indices[cl].append(ii)

        # Select top from each cluster and then global
        selected = []
        for cl, idxs in cluster_indices.items():
            if not idxs:
                continue
            best_local = sorted(idxs, key=lambda j: -composite_scores[j])[0]
            selected.append(best_local)
        selected_global = sorted(range(len(stable_seqs)), key=lambda j: -composite_scores[j])
        final_indices = []
        for s in selected:
            if s not in final_indices:
                final_indices.append(s)
        for g in selected_global:
            if len(final_indices) >= TOP_K:
                break
            if g not in final_indices:
                final_indices.append(g)
        final_indices = final_indices[:TOP_K]

        final_seqs = [stable_seqs[i] for i in final_indices]
        final_scores = [float(stable_scores[i]) for i in final_indices]
        final_prox = [float(dist_to_anchor[i]) for i in final_indices]
        final_comp = [float(composite_scores[i]) for i in final_indices]

        # Save results for this experimental row
        base_prefix = f"anchor_{idx}"
        out_csv = f"top_guided_candidates_{base_prefix}.csv"
        fasta_file = f"top_guided_candidates_{base_prefix}.fasta"
        all_scored_csv = f"all_stable_candidates_{base_prefix}_scored.csv"

        df_top = pd.DataFrame({
            'sequence': final_seqs,
            'predicted_instability': final_scores,
            'distance_to_anchor': final_prox,
            'composite_score': final_comp
        })
        df_top.to_csv(out_csv, index=False)

        df_all = pd.DataFrame({
            'sequence': stable_seqs,
            'predicted_instability': stable_scores,
            'distance_to_anchor': dist_to_anchor,
            'composite_score': composite_scores
        }).sort_values('composite_score', ascending=False)
        df_all.to_csv(all_scored_csv, index=False)

        with open(fasta_file, "w") as fh:
            for i_seq, seq_out in enumerate(final_seqs):
                fh.write(f">{base_prefix}|candidate_{i_seq+1}|instability:{final_scores[i_seq]:.4f}|prox:{final_prox[i_seq]:.4f}|comp:{final_comp[i_seq]:.4f}\n")
                fh.write(seq_out + "\n")

        logger.info("Saved: %s, %s, %s", out_csv, all_scored_csv, fasta_file)

    except Exception as e:
        logger.exception("Failed processing experimental row %d: %s", idx, str(e))

logger.info("Phase 2 generation completed for all experimental rows.")



