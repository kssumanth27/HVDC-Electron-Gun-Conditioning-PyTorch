##LSTM or CNN gemini LEREC v5 with BatchNorm + MaxPool + max voltage limit + dynamic step size
## The plots shown on the test file voltage simulation are normalized to max and min limits of each file.

## 
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
os.environ['KMP_WARNINGS'] = '0'
os.environ['MKL_CBWR'] = 'AUTO'
os.environ['MKL_WARN'] = '0'
from contextlib import redirect_stdout
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # <--- ADDED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import copy
import warnings
# Missing from both files - should be added at the start:
import random
random.seed(33)
np.random.seed(33)
torch.manual_seed(545)
torch.cuda.manual_seed_all(545)
torch.backends.cudnn.deterministic = True

# Hide most noisy warnings so prints stay visible
warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

# Pandas specific common noise
pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning

try:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass


# ==========================================
# 0. Missing Data Detection Helper
# ==========================================

def get_missing_mask(X_data, missing_value=-100.0, sensor_channels=(0, 1, 2)):
    """
    Detects which samples already have missing (naturally absent) sensor data.
    
    Args:
        X_data: Tensor of shape [N, seq_len, n_channels]
        missing_value: The sentinel value used for missing data (default -100.0)
        sensor_channels: Tuple of sensor channel indices to check (default (0, 1, 2))
    
    Returns:
        mask: Boolean tensor of shape [N, len(sensor_channels)]
              mask[i, ch_idx] = True if sample i has ANY timestep in sensor_channels[ch_idx]
              equal to missing_value.
    """
    mask = torch.zeros(X_data.shape[0], len(sensor_channels), dtype=torch.bool)
    for idx, ch in enumerate(sensor_channels):
        mask[:, idx] = (X_data[:, :, ch] == missing_value).any(dim=1)
    return mask


# ==========================================
# 0.1 Sensor Dropout Function for Training Augmentation
# ==========================================

def apply_sensor_dropout(batch_x, prob_full_missing=0.2, prob_block_missing=0.3,
                         min_block_pct=0.2, max_block_pct=0.8, missing_value=-100.0):
    """
    Applies random sensor dropout to a batch of training data.
    
    Dropout modes (applied per sample):
    - 50% chance: No dropout (clean sample)
    - 20% chance: Full window missing (entire sequence for one sensor channel)
    - 30% chance: Block missing (contiguous block for one sensor = missing_value)
    
    Only sensor channels (0,1,2) are dropped, not Prev_A1 (channel 3).
    
    Args:
        batch_x: Tensor of shape (batch, seq_len, 4)
        prob_full_missing: Probability of full window dropout
        prob_block_missing: Probability of block dropout
        min_block_pct: Minimum block size as fraction of seq_len
        max_block_pct: Maximum block size as fraction of seq_len
        missing_value: Value to use for missing data
    
    Returns:
        Modified batch_x tensor with dropout applied
    """
    batch_x = batch_x.clone()
    batch_size, seq_len, n_channels = batch_x.shape
    
    min_block = max(1, int(min_block_pct * seq_len))
    max_block = max(1, int(max_block_pct * seq_len))
    
    for i in range(batch_size):
        r = random.random()
        ch = random.randint(0, 2)  # Only sensor channels (0,1,2), not Prev_A1
        
        if r < prob_full_missing:
            # Full window missing - entire sequence for this channel
            batch_x[i, :, ch] = missing_value
        elif r < prob_full_missing + prob_block_missing:
            # Block missing - random contiguous block
            max_start = max(0, seq_len - min_block)
            start = random.randint(0, max_start)
            max_len = min(max_block, seq_len - start)
            length = random.randint(min_block, max_len)
            batch_x[i, start:start+length, ch] = missing_value
        # else: no dropout (50% chance)
    
    return batch_x


# ==========================================
# 0.1 Output Folder and Figure Saving Utilities
# ==========================================

def create_output_folder(prefix="polgun_cnn"):
    """
    Creates a timestamped output folder with a figures subfolder.
    
    Args:
        prefix: Custom prefix for the folder name (e.g., 'polgun_cnn', 'lerec_lstm')
    
    Returns:
        folder_name: Path to the created output folder
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = f"{prefix}_{timestamp}"
    figures_path = os.path.join(folder_name, "figures")
    os.makedirs(figures_path, exist_ok=True)
    print(f">>> Created output folder: {folder_name}")
    return folder_name


def save_figure(fig, output_folder, filename):
    """
    Saves a matplotlib figure to the figures subfolder as PNG.
    
    Args:
        fig: matplotlib figure object
        output_folder: path to the output folder
        filename: name for the file (without extension)
    """
    filepath = os.path.join(output_folder, "figures", f"{filename}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")


# ==========================================
# 1. Physics Constraints (Helper Functions)
# ==========================================
# GOAL: Scan training data BEFORE training to learn physical limits of the machine.
# ==========================================

def find_global_max_voltage(folder_path, voltage_col='glassmanDataXfer:hvPsVoltageMeasM'):
    """
    WHAT: Determines the absolute Maximum Voltage of the Gun from the training data.
    HOW:  Scans every CSV in the training folder, reads ONLY the voltage column,
          and finds the global maximum. This is passed to the Simulation as a hard Interlock.

    Inputs: Training folder path, voltage column name.
    Outputs: Maximum voltage (float).
    """
    print(f"--- Scanning {folder_path} for Max Voltage Limit ---")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    global_max = -float('inf')
    
    for filename in all_files:
        try:
            # OPTIMIZATION: usecols reads only 1 column to save RAM/Time
            df = pd.read_csv(filename, usecols=[voltage_col])
            current_max = df[voltage_col].max()
            if pd.notna(current_max) and current_max > global_max:
                global_max = current_max
        except:
            pass
            
    if global_max == -float('inf'):
        print("Warning: No voltage data found. Defaulting limit to Infinity.")
        return float('inf')
        
    print(f">>> Global Max Voltage Limit Found: {global_max}")
    return global_max

def calculate_average_step_size(folder_path, voltage_col='glassmanDataXfer:hvPsVoltageMeasM', target_col='VoltageChange'):
    """
    WHAT: Calculates the average step size of the training files for increasing the voltage.
    HOW:  It aligns row t (Command=1) with row t+1 (Voltage).
          It calculates V(t+1) - V(t) for every successful ramp and averages them.

    Inputs: Training folder path, voltage column name, VoltageChange column name.
    Outputs: Average step size (float).
    """
    print(f"--- Calculating Average Step Size from {folder_path} ---")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    all_increases = []
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename, usecols=[voltage_col, target_col])
            df[voltage_col] = pd.to_numeric(df[voltage_col], errors='coerce')
            
            # LOGIC: Look Forward (t+1)
            # We want to know the EFFECT of the command at time 't'.
            df['next_voltage'] = df[voltage_col].shift(-1)
            df['forward_step'] = df['next_voltage'] - df[voltage_col]
            
            # FILTER:
            # 1. Target=1: A command to increase was given.
            # 2. Step > 0.1: The voltage ACTUALLY increased (filters out failures/noise).
            valid_increases = df[ (df[target_col] == 1) & (df['forward_step'] > 0.1) ]
            
            if not valid_increases.empty:
                all_increases.extend(valid_increases['forward_step'].values)
        except:
            pass
            
    if not all_increases:
        print("Warning: No positive steps found. Defaulting to 10.0")
        return 10.0
        
    avg_step = np.mean(all_increases)
    print(f">>> Calculated Average Step Size: {avg_step:.4f}")
    return avg_step




"""
The three functions below implement the "Quiet Negative" filtering logic i.e. if the system is stable wrt the three sensors
(GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal), and the voltage change is 0, we skip that sample during training.

This is expected to increase the model's accuracy and focus on learning the ramp-up events. 

STILL NEED TO FIND A PERFECT NOISE FUNCTION TO RECTIFY. AS OF NOW, NO DATA IS BEING FILTERED OUT DURING TRAINING.
"""
def robust_noise_sigma_mad_diff(series: pd.Series):
    """
    Robust noise sigma estimate using Medium Absolute Deviation of first differences.
    """
    x = pd.to_numeric(series, errors='coerce').dropna()
    if len(x) < 5:
        return np.nan

    dx = x.diff().dropna()
    mad = np.median(np.abs(dx - np.median(dx)))
    sigma = mad / (0.6745 * np.sqrt(2))
    return sigma


def estimate_folder_noise_thresholds(folder_path, cols, k=1.0):
    """
    Computes per-file MAD-diff sigma for each sensor,
    then returns median sigma across files as the threshold.
    threshold = k * median_sigma
    """
    print(f"--- Estimating Noise Thresholds (MAD baseline) from {folder_path} ---")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    sigma_map = {c: [] for c in cols}

    for f in all_files:
        try:
            df = pd.read_csv(f)
            for c in cols:
                if c in df.columns:
                    s = robust_noise_sigma_mad_diff(df[c])
                    if pd.notna(s):
                        sigma_map[c].append(s)
        except:
            pass

    thresholds = {}
    for c in cols:
        if sigma_map[c]:
            base = float(np.median(sigma_map[c]))
            thresholds[c] = k * base
        else:
            thresholds[c] = 0.0

    print(">>> Noise thresholds (MAD diff):")
    for k_, v_ in thresholds.items():
        print(f"  {k_}: {v_:.6f}")

    return thresholds


def should_skip_quiet_negative(df, idx, noise_thresholds, voltage_limit=None):
    """
    Quiet negative definition (your updated rule):
    - VoltageChange == 0 (checked outside)
    - AND all three sensors are quiet based on delta noise:
        |x[t] - x[t-1]| < sigma_x
    - AND voltage[t] < voltage_limit (per-file max)

    Returns True if sample should be skipped.
    """
    required = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]
    for c in required:
        if c not in df.columns:
            return False

    if idx <= 0:
        return False

    # Current timestep values
    cur_val = pd.to_numeric(df['GunCurrent.Avg'].iloc[idx], errors='coerce')
    pre_val = pd.to_numeric(df['peg-BL-cc:pressureM'].iloc[idx], errors='coerce')
    rad_val = pd.to_numeric(df['RadiationTotal'].iloc[idx], errors='coerce')

    # Previous timestep values
    cur_prev = pd.to_numeric(df['GunCurrent.Avg'].iloc[idx-1], errors='coerce')
    pre_prev = pd.to_numeric(df['peg-BL-cc:pressureM'].iloc[idx-1], errors='coerce')
    rad_prev = pd.to_numeric(df['RadiationTotal'].iloc[idx-1], errors='coerce')

    if not all(pd.notna(v) for v in [cur_val, pre_val, rad_val, cur_prev, pre_prev, rad_prev]):
        return False

    low_activity = (
        abs(cur_val - cur_prev) < noise_thresholds.get('GunCurrent.Avg', 0) and
        abs(pre_val - pre_prev) < noise_thresholds.get('peg-BL-cc:pressureM', 0) and
        abs(rad_val - rad_prev) < noise_thresholds.get('RadiationTotal', 0)
    )

    if not low_activity:
        return False

    # Per-file voltage condition
    if voltage_limit is not None and 'glassmanDataXfer:hvPsVoltageMeasM' in df.columns:
        volt_val = pd.to_numeric(df['glassmanDataXfer:hvPsVoltageMeasM'].iloc[idx], errors='coerce')
        if pd.notna(volt_val):
            if not (volt_val < voltage_limit):
                return False

    return True

# ==========================================
# 2. Data Loading & Processing pipeline
# ==========================================

def read_folder_csvs(folder_path):

    ## Reads the csv files from a folder into a list of dataframes.
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Store filename for later debug prints
            df.attrs['source_file'] = os.path.basename(filename)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return dfs


def process_single_df_to_sequences(df, scaler, sequence_length,
                                   noise_thresholds=None,
                                   filter_quiet_negatives=True):
    # 0. Pre-processing: Find Time Column and Resample for Gaps
    # We look for case-insensitive 'time' column



    """
    Reads the time column and removes any duplicates.
    Checks for any NaNs in VoltageChange before and after time handling.
    If VoltageChange column has NaNs introduced after time handling, those rows will be skipped later.
    If VoltageChange column has '-1', those will be mapped to '0' and the next 5-steps will also be read as '0'.
    Uses normalized values for inputs of current, pressure, radiation.
    Inputs: DataFrame, fitted scaler, sequence length, noise thresholds, filter flag.
    Outputs: X_sequences (NumPy array), y_sequences (NumPy array) (training data inputs and voltagechange)

    Used  by: prepare_all_data(), test_single_file_simulation()
    
    """
    df = df.copy()

# -------- DEBUG: BEFORE time handling ----------
    
    # if 'VoltageChange' in df.columns:
    #     vc_before = pd.to_numeric(df['VoltageChange'], errors='coerce').values
    #     print(f"[DEBUG BEFORE] rows={len(df)} | VoltageChange NaNs={np.isnan(vc_before).sum()}")
    # else:
    #     print(f"[DEBUG BEFORE] rows={len(df)} | VoltageChange column missing")
# ----------------------------------------------

    time_col = next((col for col in df.columns if col.lower() == 'time'), None)

    # --- RESTORED YOUR ORIGINAL LOGIC HERE ---
    if time_col:
        # --- FIX: Ensure Time is Numeric ---
        # The error "str - str" happens if time is read as strings.
        # We assume time is either numeric strings ("1", "2") or datetime strings.
        
        is_numeric = False
        try:
            # Try converting to numbers first (e.g. "1.0", "2.0")
            df[time_col] = pd.to_numeric(df[time_col])
            is_numeric = True
        except ValueError:
            pass # Was not simple numbers, might be datetime
            
        if not is_numeric:
            try:
                # Try converting to Datetime (e.g. "2024-01-01 12:00:00")
                df[time_col] = pd.to_datetime(df[time_col])
                # Convert Datetime to "Seconds since start" so we can do math
                start_time = df[time_col].min()
                df[time_col] = (df[time_col] - start_time).dt.total_seconds()
            except Exception as e:
                print(f"Warning: Could not convert '{time_col}' to numbers. Skipping resampling. Error: {e}")
                time_col = None # Disable resampling for this file

    if time_col:
        # Drop duplicates to prevent indexing errors
        df = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col)
        
        # Calculate dominant time step (frequency)
        if len(df) > 1:
            time_diffs = df[time_col].diff()
            # mode() returns a series, take first element. Rounding handles float jitter.
            freq = time_diffs.mode()[0]
            
            # Ensure freq is valid and positive
            if pd.notna(freq) and freq > 0:
                # Create a complete time grid from start to end
                min_t, max_t = df[time_col].min(), df[time_col].max()
                
                # Use numpy arange for float safety
                full_time_index = np.arange(min_t, max_t + freq/1000.0, freq)
                
                # Rounding to avoid float precision mismatch (e.g. 1.0000001 vs 1.0)
                df[time_col] = df[time_col].round(6)
                full_time_index = np.round(full_time_index, 6)
                
                # Reindex the dataframe. 
                df = df.set_index(time_col).reindex(full_time_index).reset_index()
                df = df.rename(columns={'index': time_col, time_col: time_col})

    # -------- DEBUG: check if time handling introduced NaNs ----------
#     if 'VoltageChange' in df.columns:
#         vc_tmp = pd.to_numeric(df['VoltageChange'], errors='coerce').values
#         print(f"[DEBUG] After time handling: rows={len(df)} | VoltageChange NaNs={np.isnan(vc_tmp).sum()}")
#     else:
#         print(f"[DEBUG] After time handling: rows={len(df)} | VoltageChange column missing")
# # ---------------------------------------------------------------


    # # 1. Map Target: -1 -> 0
    # if 'VoltageChange' in df.columns:
    #     df['VoltageChange'] = df['VoltageChange'].apply(lambda x: 0 if x == -1 else x)


        # 1. Map Target: -1 -> 0, and enforce a 5-step refractory window after each -1
    if 'VoltageChange' in df.columns:
        # Work on a numeric copy
        vc_raw = pd.to_numeric(df['VoltageChange'], errors='coerce').values.copy()
        n = len(vc_raw)

        # mask of indices that should be forced to 0
        force_zero = np.zeros(n, dtype=bool)

        # locations where the *original* label is -1
        neg_indices = np.where(vc_raw == -1)[0]

        # for each -1, force current index + next 5 to 0
        for idx in neg_indices:
            end = min(n, idx + 6)  # idx, idx+1, ..., idx+5
            force_zero[idx:end] = True

        vc_clean = vc_raw.copy()
        # first map -1 → 0
        vc_clean[vc_clean == -1] = 0
        # then enforce refractory window → 0, overriding any 1s
        vc_clean[force_zero] = 0

        df['VoltageChange'] = vc_clean



    # --- NEW: Per-file max voltage for TRAIN filtering ---
    file_max_voltage = None
    if 'glassmanDataXfer:hvPsVoltageMeasM' in df.columns:
        v = pd.to_numeric(df['glassmanDataXfer:hvPsVoltageMeasM'], errors='coerce')
        if v.notna().any():
            file_max_voltage = float(v.max())


        # --- For debug printing ---
    file_label = df.attrs.get('source_file', 'unknown_file')

    # --- Skip counters ---
    total_zero_targets = 0
    skipped_quiet_zero = 0


    
    # 2. Handle Inputs (NO MASKS, NO VOLTAGE INCLUDED IN FEATURES)
    # Channels: [Current, Pressure, Radiation, Prev_A1] = 4 channels
    feature_cols = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]
    input_data = []
    
    for col in feature_cols:
        # Keep NaN as NaN - will be replaced with -100 after normalization
        val = df[col].astype(float)
        input_data.append(val.values)
        
    # 3. Add Previous Output (A1) to Input Features
    a1_history = df['VoltageChange'].fillna(0).values
    input_data.append(a1_history)

    # Stack features: [Current, Pressure, Radiation, Prev_A1] = 4 channels
    X_raw = np.stack(input_data, axis=1)
    
    # 4. Apply Scaler (Only to sensor values - indices 0, 1, 2)
    # Store NaN mask before transform
    nan_mask = np.isnan(X_raw[:, [0, 1, 2]])
    
    # Replace NaN temporarily with 0 for transform
    X_sensor_temp = np.nan_to_num(X_raw[:, [0, 1, 2]], nan=0.0)
    X_raw[:, [0, 1, 2]] = scaler.transform(X_sensor_temp)
    
    # Restore NaN positions and replace with -100 (missing indicator)
    for i in range(3):
        X_raw[nan_mask[:, i], i] = -100.0
    
    # 5. Create Sequences
    if len(X_raw) <= sequence_length:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1))

    X_seq, y_seq = [], []
    
    y_raw_target = df['VoltageChange'].values 

    #print(f"[DEBUG] Max possible windows={max(0, len(X_raw)-sequence_length)} | "
     # f"NaN targets={np.isnan(y_raw_target).sum()}")

    
    for i in range(len(X_raw) - sequence_length):
        idx = i + sequence_length
        target_val = y_raw_target[idx]

        # Only train if we have a VALID target (not a filled gap)
        if np.isnan(target_val):
            continue

        # --- NEW: Skip quiet negatives (TRAIN ONLY) ---
        if filter_quiet_negatives and noise_thresholds and target_val == 0:
            total_zero_targets += 1

            if should_skip_quiet_negative(
                df, idx,
                noise_thresholds=noise_thresholds,
                voltage_limit=file_max_voltage  # per-file max
            ):
                skipped_quiet_zero += 1
                continue

        X_seq.append(X_raw[i : i + sequence_length])
        y_seq.append(target_val)


        
    if len(X_seq) == 0:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1))
    
        # --- Debug summary per file ---
    # if filter_quiet_negatives and noise_thresholds:
    #     if total_zero_targets > 0:
    #         pct = 100.0 * skipped_quiet_zero / total_zero_targets
    #       #  p#rint(f"[Quiet-Neg Filter] {file_label}: skipped {skipped_quiet_zero}/{total_zero_targets} "
    #           #    f"quiet 0-targets ({pct:.1f}%)")
    #     #else:
    #      #   print(f"[Quiet-Neg Filter] {file_label}: no 0-targets found to evaluate.")

        
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

def prepare_all_data(train_folder, test_folder, sequence_length,
                     noise_thresholds=None,
                     filter_quiet_negatives=True):

    """
    WHAT: Orchestrates data loading, scaling, and splitting.

    Uses shuffled train_test_split for 85/15 train/val split.
    NaN-aware normalization (ignores NaN when computing mean/std).
    
    Reads csv files from train and test folders,
    Fits scaler on ALL training data (NaN-aware),
    Processes them into sequences using process_single_df_to_sequences(),
    Splits sequences into train/val using shuffled train_test_split,
    Calculates class weights for imbalanced data.

    Inputs: Train folder path, Test folder path, sequence length,
    Outputs: data dict (X_train, y_train, X_val, y_val, X_test, y_test),
             pos_weight, scaler

    X_* are torch FloatTensors of shape (Samples, Seq_Len, 4) (inputs for the model)
    y_* are torch FloatTensors of shape (Samples, 1) (Our final target: VoltageChange)
    """
    print(f"Loading Data...")
    
    # --- READ ALL TRAINING FILES ---
    train_dfs = read_folder_csvs(train_folder)
    test_dfs = read_folder_csvs(test_folder)
    
    print(f"Loaded {len(train_dfs)} training files")
    
    # --- NaN-AWARE SCALER ---
    # Fit scaler on ALL training data, ignoring NaN values
    print("Fitting NaN-aware Scaler on training data...")
    feature_cols = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]
    train_concat = pd.concat(train_dfs, ignore_index=True)
    train_data = train_concat[feature_cols].values
    
    # Compute mean and std ignoring NaN
    scaler = StandardScaler()
    scaler.mean_ = np.nanmean(train_data, axis=0)
    scaler.scale_ = np.nanstd(train_data, axis=0, ddof=0)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 3
    scaler.n_samples_seen_ = np.sum(~np.isnan(train_data), axis=0)
    
    print(f"  Scaler mean (ignoring NaN): {scaler.mean_}")
    print(f"  Scaler std (ignoring NaN): {scaler.scale_}")
    
    # --- PROCESS ALL TRAIN SEQUENCES ---
    print("Processing Train Sequences...")
    X_train_list, y_train_list = [], []
    for df in train_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler, sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=filter_quiet_negatives
        )
        if len(x_s) > 0:
            X_train_list.append(x_s)
            y_train_list.append(y_s)
    
    if X_train_list:
        X_train_all = np.concatenate(X_train_list)
        y_train_all = np.concatenate(y_train_list)
    else:
        X_train_all = np.empty((0, sequence_length, 4))
        y_train_all = np.empty((0, 1))
    
    # --- SHUFFLED TRAIN/VAL SPLIT ---
    print("Splitting into train/val with shuffled train_test_split (85/15)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, 
        test_size=0.15, 
        shuffle=True, 
        random_state=97
    )
    
    # --- PROCESS TEST SEQUENCES ---
    print("Processing Test Sequences...")
    X_test_list, y_test_list = [], []
    for df in test_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler, sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=False  # Don't filter test data
        )
        if len(x_s) > 0:
            X_test_list.append(x_s)
            y_test_list.append(y_s)
            
    if X_test_list:
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0, sequence_length, 4))
        y_test = np.empty((0, 1))

    # --- CLASS WEIGHTS ---
    # Logic: Dataset is imbalanced (mostly stable, few ramps).
    # We calculate Weight = Negatives / Positives to force model to learn the ramps.
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float)
    
    data = {
        'X_train': torch.FloatTensor(X_train), 'y_train': torch.FloatTensor(y_train),
        'X_val': torch.FloatTensor(X_val), 'y_val': torch.FloatTensor(y_val),
        'X_test': torch.FloatTensor(X_test), 'y_test': torch.FloatTensor(y_test)
    }

    # Validation class counts
    n_pos_val = np.sum(y_val == 1)
    n_neg_val = np.sum(y_val == 0)
    
    # Test class counts
    n_pos_test = np.sum(y_test == 1)
    n_neg_test = np.sum(y_test == 0)

    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Training Set:   {data['X_train'].shape}  (Samples, Seq_Len, Channels)")
    print(f"Validation Set: {data['X_val'].shape}")
    print(f"Test Set:       {data['X_test'].shape}")
    print("-" * 50)
    print(f"Shuffled Split: 85% train / 15% val (random_state=97)")
    print("-" * 50)
    print(f"CLASS DISTRIBUTION:")
    print(f"  TRAIN:      Stable(0)={int(n_neg):>6}  |  Increase(1)={int(n_pos):>6}  |  1-ratio={100*n_pos/(n_pos+n_neg):.2f}%")
    print(f"  VALIDATION: Stable(0)={int(n_neg_val):>6}  |  Increase(1)={int(n_pos_val):>6}  |  1-ratio={100*n_pos_val/(n_pos_val+n_neg_val+1e-8):.2f}%")
    print(f"  TEST:       Stable(0)={int(n_neg_test):>6}  |  Increase(1)={int(n_pos_test):>6}  |  1-ratio={100*n_pos_test/(n_pos_test+n_neg_test+1e-8):.2f}%")
    print("-" * 50)
    print(f"Calculated Pos_Weight (for Train): {pos_weight.item():.4f}")
    print("="*50 + "\n")
    
    return data, pos_weight, scaler

# ==========================================
# 3. Model Architecture (CNN + BN + Pool)
# ==========================================
""""
Defines CNN and LSTM models using PyTorch.

Outputs: model layers in dicts, forward functions."""

# --- LSTM MODEL ---
def create_lstm_model(input_dim, hidden_dim):
    lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
    linear = nn.Linear(hidden_dim, 1)
    return {'lstm': lstm, 'linear': linear}

def forward_lstm(models, x):
    # LSTM expects (Batch, Seq, Features) -> No permutation needed
    lstm_out, _ = models['lstm'](x)
    # Take the last time step output
    return models['linear'](lstm_out[:, -1, :])


def create_cnn_model(
    input_dim,
    seq_len,
    n_filters1: int = 128,
    n_filters2: int = 256,
    kernel_size: int = 3,
    pool_size: int = 2,
):
    """
    WHAT: Defines the Neural Network structure.
    ARCH: Conv1d -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear

    Output: All the layers in a dict for easy access.

    The default values for n_filters1, n_filters2, kernel_size, and pool_size
    are chosen to match the original architecture. They can be overridden
    by passing explicit values (e.g. from a hyperparameter dict).
    """
    
    # Block 1: Feature Extraction (Low level)
    # Use padding so that length is preserved for odd kernel sizes.
    conv1 = nn.Conv1d(input_dim, n_filters1, kernel_size=kernel_size, padding=kernel_size // 2)
    bn1 = nn.BatchNorm1d(n_filters1) # Normalizes activations for stability
    
    # Block 2: Feature Extraction (High level)
    conv2 = nn.Conv1d(n_filters1, n_filters2, kernel_size=kernel_size, padding=kernel_size // 2)
    bn2 = nn.BatchNorm1d(n_filters2)
    
    # Pooling: Shrinks time dimension by half (Downsampling)
    pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
    
    # --- LINEAR LAYER SIZING MATH ---
    # Logic: We must calculate exactly how many neurons are left after flattening.
    # Pool 1 reduces length: 25 -> 12 (for pool_size=2)
    len_after_pool1 = seq_len // pool_size
    # Pool 2 reduces length: 12 -> 6 (for pool_size=2)
    len_after_pool2 = len_after_pool1 // pool_size
    
    # Total inputs = Filters * Remaining Time Steps
    linear_input_size = n_filters2 * len_after_pool2
    #linear_input_size = n_filters2 * seq_len  # No pooling used
    
   # print(f"Model Init: SeqLen {seq_len} -> Reduced to {len_after_pool2} steps -> Linear Input {linear_input_size}")
    
    linear = nn.Linear(linear_input_size, 1)

    return {'conv1': conv1, 'bn1': bn1, 'conv2': conv2, 'bn2': bn2, 'pool':pool, 'linear': linear}   


def init_kaiming_normal_for_conv(m):
    # Only touches Conv1d/Conv2d/Conv3d
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # For ReLU-like nonlinearities (your model uses ReLU)
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def forward_cnn(models, x):
    # Permute: PyTorch Conv1d expects [Batch, Channels, Time]
    x = x.permute(0, 2, 1) 
    
    # Block 1
    x = models['conv1'](x)
    x = models['bn1'](x)
    x = F.relu(x)
    x = models['pool'](x)
    
    # Block 2
    x = models['conv2'](x)
    x = models['bn2'](x)
    x = F.relu(x)
    x = models['pool'](x)
    
    # Flatten: [Batch, Channels, Time] -> [Batch, Features]
    x = x.reshape(x.size(0), -1)
    return models['linear'](x)


def summarize_model_parameters(models, model_type="cnn"):
    """
    Print all trainable parameters for your dict-based model.
    
    models: dict of layers, e.g.
        {'conv1': conv1, 'bn1': bn1, 'conv2': conv2, 'bn2': bn2, 'linear': linear}
        or {'lstm': lstm, 'linear': linear}
    """
    print("\n" + "="*60)
    print(f"MODEL PARAMETER SUMMARY [{model_type.upper()}]")
    print("="*60)

    total_params = 0

    for layer_name, layer in models.items():
        layer_trainable = 0
        print(f"\nLayer: {layer_name} ({layer.__class__.__name__})")

        for pname, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            num = p.numel()
            layer_trainable += num
            total_params += num
            print(f"  - {pname:20s} shape={tuple(p.shape)}  params={num}")

        if layer_trainable == 0:
            print("  (no trainable parameters)")
        else:
            print(f"  Trainable params in {layer_name}: {layer_trainable}")

    print("\n" + "-"*60)
    print(f"TOTAL TRAINABLE PARAMETERS: {total_params}")
    print("="*60 + "\n")


# ==========================================
# 4. Training Loop
# ==========================================

def train_model(
    model_type,
    data,
    pos_weight,
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
    n_filters1: int = 128,
    n_filters2: int = 256,
    kernel_size: int = 3,
    pool_size: int = 2,
    hidden_dim: int = 64,
    prob_full_missing: float = 0.2,
    prob_block_missing: float = 0.3,
    min_block_pct: float = 0.2,
    max_block_pct: float = 0.8,
    missing_value: float = -100.0,
):

    """
    Training loop for the model.
    Creates the models LSTM and CNN based on model_type.
    Uses Adam optimizer and BCEWithLogitsLoss with pos_weight for imbalance.
    
    NEW: Applies sensor dropout augmentation during training using apply_sensor_dropout().
    
    Inputs: model_type ('cnn' or 'lstm'), data dict, pos_weight tensor, epochs, batch_size
            prob_full_missing: probability of full window dropout (20%)
            prob_block_missing: probability of block dropout (30%)
            Remaining 50%: no dropout
    Outputs: trained models dict, forward function, training history dict

    Prints: training and validation loss/accuracy per epoch.
    """ 
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # Standard TensorDataset - dropout applied per-batch in training loop
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f">>> Training with sensor dropout augmentation:")
    print(f"    - Full missing prob: {prob_full_missing}")
    print(f"    - Block missing prob: {prob_block_missing}")
    print(f"    - Block range: {min_block_pct*100:.0f}%-{max_block_pct*100:.0f}% of sequence")
    
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    
   # --- MODEL SWITCHING LOGIC RESTORED ---
    if model_type == 'lstm':
        print("Initializing LSTM Model...")
        models = create_lstm_model(input_dim, hidden_dim=hidden_dim)
        forward_fn = forward_lstm
    else:
        print("Initializing CNN Model...")
        models = create_cnn_model(
            input_dim,
            seq_len,
            n_filters1=n_filters1,
            n_filters2=n_filters2,
            kernel_size=kernel_size,
            pool_size=pool_size,
        )
        forward_fn = forward_cnn
     

     # >>> PASTE POINT-2 HERE (right after CNN creation) <<<
    # for k in ["conv1", "conv2"]:
    #     nn.init.kaiming_normal_(models[k].weight, mode="fan_out", nonlinearity="relu")
    #     if models[k].bias is not None:
    #         nn.init.zeros_(models[k].bias)

    # --- CAPTURE INITIAL WEIGHTS ---
    # Use deepcopy to ensure the initial weights are not changed by training
    initial_weights = copy.deepcopy({name: layer.state_dict() for name, layer in models.items()})
    print(">>> Initial model weights have been captured.")


    params = [p for layer in models.values() for p in layer.parameters()]

    optimizer = optim.Adam(params, lr=learning_rate)
    
        # --- DEBUG: print model parameter summary ---
    summarize_model_parameters(models, model_type=model_type)

    # LOSS FUNCTION:
    # BCEWithLogitsLoss combines Sigmoid + CrossEntropy. 
    # pos_weight scales the loss for the minority class (Ramps).
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    # --- PER-EPOCH WEIGHT TRACKING ---
    weight_history = []  # List to store weight snapshots each epoch

    for epoch in range(epochs):
        
        # --- TRAIN LOOP (Mini-Batches) ---
        if model_type == 'lstm':
             models['lstm'].train() 
        else:
             models['conv1'].train()
             models['bn1'].train()
            
             models['conv2'].train()
             models['bn2'].train()
             #models['conv3'].train()
   
        
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Apply sensor dropout augmentation to this batch
            batch_X = apply_sensor_dropout(
                batch_X, 
                prob_full_missing=prob_full_missing,
                prob_block_missing=prob_block_missing,
                min_block_pct=min_block_pct,
                max_block_pct=max_block_pct,
                missing_value=missing_value
            )
            
            optimizer.zero_grad()
            preds = forward_fn(models, batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

         # --- VAL LOOP (Full Batch usually fine for Val) ---
        if model_type == 'lstm':
             models['lstm'].eval() 
        else:
             models['conv1'].eval()
             models['bn1'].eval()
             models['conv2'].eval()
             models['bn2'].eval()
             #models['conv3'].eval()
        
      
        with torch.no_grad():
            val_preds = forward_fn(models, X_val)
            v_loss = criterion(val_preds, y_val)
            pred_cls = (val_preds > 0).float()
            acc = (pred_cls == y_val).float().mean()
            
            history['val_loss'].append(v_loss.item())
            history['val_acc'].append(acc.item())
        
        #if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f} | Val Loss {v_loss.item():.4f} | Val Acc {acc.item():.4f}")

        # --- CAPTURE WEIGHTS THIS EPOCH ---
        epoch_snapshot = {
            'epoch': epoch,
            'conv1': copy.deepcopy(models['conv1'].state_dict()) if 'conv1' in models else None
        }
        weight_history.append(epoch_snapshot)

    # --- CAPTURE FINAL WEIGHTS ---
    final_weights = {name: layer.state_dict() for name, layer in models.items()}
    print(">>> Final trained model weights have been captured.")
    print(f">>> Weight history captured for {len(weight_history)} epochs.")
            
    return models, forward_fn, history, initial_weights, final_weights, weight_history


# ==========================================
# 4.5 Model Checkpointing (Save & Load)
# ==========================================

def save_checkpoint(output_folder, models, scaler, hyperparams, history, 
                    weight_history, initial_weights, final_weights, 
                    pos_weight, train_dir, test_dir, max_voltage, avg_step,
                    channel_names, model_type='cnn'):
    """
    Saves both lightweight and full checkpoints.
    
    Lightweight (checkpoint.pth): Model weights only - fast loading for inference
    Full (full_checkpoint.pth): Everything needed for analysis and reproducibility
    """
    
    # --- Lightweight checkpoint: just model weights ---
    lightweight_path = os.path.join(output_folder, "checkpoint.pth")
    torch.save(
        {name: layer.state_dict() for name, layer in models.items()},
        lightweight_path
    )
    print(f"Saved lightweight checkpoint: {lightweight_path}")
    
    # --- Full checkpoint: everything ---
    full_checkpoint = {
        # Model
        'model_state_dict': {name: layer.state_dict() for name, layer in models.items()},
        'model_type': model_type,
        
        # Preprocessing (scaler parameters for reconstructing StandardScaler)
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'scaler_var': scaler.var_.tolist(),
        'scaler_n_features': scaler.n_features_in_,
        
        # Hyperparameters
        'hyperparameters': hyperparams,
        
        # Training results
        'training_history': history,
        'weight_history': weight_history,
        'initial_weights': initial_weights,
        'final_weights': final_weights,
        
        # Data info
        'pos_weight': pos_weight.item() if torch.is_tensor(pos_weight) else pos_weight,
        'train_dir': train_dir,
        'test_dir': test_dir,
        'channel_names': channel_names,
        
        # Constraints learned from data
        'max_voltage_limit': max_voltage,
        'avg_step_size': avg_step,
    }
    
    full_path = os.path.join(output_folder, "full_checkpoint.pth")
    torch.save(full_checkpoint, full_path)
    print(f"Saved full checkpoint: {full_path}")
    
    return lightweight_path, full_path


def load_checkpoint(checkpoint_path, model_type='cnn'):
    """
    Loads a saved checkpoint and reconstructs model + scaler.
    
    Args:
        checkpoint_path: Path to full_checkpoint.pth
        model_type: 'cnn' or 'lstm'
    
    Returns:
        models: Reconstructed model dict with loaded weights
        forward_fn: Forward function for the model
        scaler: Reconstructed StandardScaler
        checkpoint: Full checkpoint dict with all saved data
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])
    scaler.var_ = np.array(checkpoint['scaler_var'])
    scaler.n_features_in_ = checkpoint['scaler_n_features']
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparameters']
    seq_len = hyperparams['sequence_length']
    input_dim = 4  # Fixed for this architecture: 3 sensors + prev_A1 (no masks)
    
    # Reconstruct model
    model_type = checkpoint.get('model_type', model_type)
    
    if model_type == 'lstm':
        models = create_lstm_model(
            input_dim,
            hidden_dim=hyperparams.get('hidden_dim', 64),
        )
        forward_fn = forward_lstm
    else:
        models = create_cnn_model(
            input_dim,
            seq_len,
            n_filters1=hyperparams.get('n_filters1', 128),
            n_filters2=hyperparams.get('n_filters2', 256),
            kernel_size=hyperparams.get('kernel_size', 3),
            pool_size=hyperparams.get('pool_size', 2),
        )
        forward_fn = forward_cnn
    
    # Load weights
    for name, state_dict in checkpoint['model_state_dict'].items():
        if name in models:
            models[name].load_state_dict(state_dict)
    
    # Set to eval mode
    for layer in models.values():
        layer.eval()
    
    print(">>> Checkpoint loaded successfully!")
    print(f"    Model type: {model_type}")
    print(f"    Epochs trained: {len(checkpoint['training_history']['train_loss'])}")
    print(f"    Sequence length: {seq_len}")
    
    return models, forward_fn, scaler, checkpoint


# ==========================================
# 5. Visualizations (RESTORED)
# ==========================================
def flatten_layer(state_dict, include_bias=True):
    """Flatten weights (and optionally bias) from a layer state_dict into 1D numpy array."""
    vecs = []
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        if k == "num_batches_tracked":  # BN bookkeeping
            continue
        if k.startswith("weight") or k == "weight":
            vecs.append(v.detach().cpu().numpy().ravel())
        elif include_bias and (k.startswith("bias") or k == "bias"):
            vecs.append(v.detach().cpu().numpy().ravel())
    return np.concatenate(vecs) if vecs else np.array([])

def viz_layer(initial_weights, final_weights, layer_name, include_bias=True, max_points=30000, bins=120, seed=0):
    """Polar radiating plot (μ at center, σ rings) + histogram overlay for one layer."""
    if layer_name not in initial_weights or layer_name not in final_weights:
        print(f"[WARN] '{layer_name}' not found. Keys: {list(initial_weights.keys())}")
        return

    xi = flatten_layer(initial_weights[layer_name], include_bias=include_bias)
    xf = flatten_layer(final_weights[layer_name],   include_bias=include_bias)

    xi = xi[np.isfinite(xi)]
    xf = xf[np.isfinite(xf)]
    if xi.size == 0 or xf.size == 0:
        print(f"[WARN] Empty weights for {layer_name}.")
        return

    # subsample for speed
    rng = np.random.default_rng(seed)
    if xi.size > max_points:
        xi = xi[rng.choice(xi.size, max_points, replace=False)]
    if xf.size > max_points:
        xf = xf[rng.choice(xf.size, max_points, replace=False)]

    def polar(ax, x, title):
        mu = float(x.mean())
        sig = float(x.std(ddof=0)) if x.size > 1 else 0.0
        z = x - mu
        r = np.abs(z)

        # Encode sign into angle halves: + on [0,π), - on [π,2π)
        theta = np.empty_like(r)
        pos = z >= 0
        theta[pos]  = rng.uniform(0, np.pi,  size=pos.sum())
        theta[~pos] = rng.uniform(np.pi, 2*np.pi, size=(~pos).sum())

        ax.scatter(theta, r, s=6, alpha=0.2)
        ax.set_title(f"{title}\nμ={mu:.4g}, σ={sig:.4g}")
        ax.set_rticks([])
        ax.grid(True, alpha=0.3)

        if sig > 0:
            t = np.linspace(0, 2*np.pi, 600)
            for k in (1, 2, 3):
                ax.plot(t, np.full_like(t, k*sig), linestyle="--", linewidth=1)
            ax.set_rmax(max(r.max(), 3*sig) * 1.05)

    # --- Polar plots ---
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    polar(ax1, xi, f"{layer_name}: Initial")
    polar(ax2, xf, f"{layer_name}: Final")
    plt.tight_layout()
    plt.show()

    # --- Distribution overlay ---
    plt.figure()
    plt.hist(xi, bins=bins, density=True, histtype="step", linewidth=2, label="Initial")
    plt.hist(xf, bins=bins, density=True, histtype="step", linewidth=2, label="Final")

    for x in (xi, xf):
        mu = float(x.mean())
        sig = float(x.std(ddof=0)) if x.size > 1 else 0.0
        plt.axvline(mu, linestyle="--", linewidth=1)
        if sig > 0:
            plt.axvline(mu - sig, linestyle=":", linewidth=1)
            plt.axvline(mu + sig, linestyle=":", linewidth=1)

    plt.title(f"{layer_name}: Initial vs Final (distribution)")
    plt.xlabel("Weight value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




CHANNEL_NAMES = [
    "Current",
    "Pressure",
    "Radiation",
    "Prev_A1",
]

def get_conv1_weight_tensor(models_or_state):
    """
    Accepts:
      - your dict of live layers: models['conv1'] is nn.Conv1d
      - OR a captured state_dict like final_weights['conv1']
    Returns conv1.weight as a torch.Tensor with shape [out_ch, in_ch, k]
    """
    if isinstance(models_or_state, dict) and "conv1" in models_or_state:
        conv1 = models_or_state["conv1"]
        if hasattr(conv1, "weight"):
            return conv1.weight.detach().cpu()
        # if it's a state_dict
        if isinstance(conv1, dict) and "weight" in conv1:
            return conv1["weight"].detach().cpu()

    raise ValueError("Could not find conv1 weights. Pass `models` or a state_dict like final_weights['conv1'].")

def plot_conv1_input_channel_hists(models, initial_weights=None, final_weights=None, bins=120):
    """
    If initial_weights/final_weights provided, overlays BEFORE vs AFTER in each plot.
    Otherwise plots current model weights only.
    """
    w_after = get_conv1_weight_tensor(models)  # [out, in, k]

    w_before = None
    if initial_weights is not None:
        w_before = get_conv1_weight_tensor({"conv1": initial_weights["conv1"]})

    # summary stats per channel
    print("Conv1 per-input-channel summary:")
    for i, name in enumerate(CHANNEL_NAMES):
        v_after = w_after[:, i, :].reshape(-1).numpy()
        msg = (
            f"{i}: {name:14s} | "
            f"mean={v_after.mean(): .4e}  std={v_after.std(): .4e}  "
            f"mean|w|={np.mean(np.abs(v_after)): .4e}  "
            f"L2={np.linalg.norm(v_after): .4e}"
        )
        print(msg)

    # plots
    for i, name in enumerate(CHANNEL_NAMES):
        plt.figure(figsize=(7, 4))
        v_after = w_after[:, i, :].reshape(-1).numpy()
        plt.hist(v_after, bins=bins, alpha=0.6, label="after")

        if w_before is not None:
            v_before = w_before[:, i, :].reshape(-1).numpy()
            plt.hist(v_before, bins=bins, alpha=0.6, label="before")

        plt.title(f"Conv1 weight distribution — input channel {i}: {name}")
        plt.xlabel("weight value")
        plt.ylabel("count")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()

# ---- usage ----
# If you still have the live model:

# Or, if you only have final_weights and not live models, you can create a tiny wrapper:
# plot_conv1_input_channel_hists({"conv1": final_weights["conv1"]}, initial_weights=initial_weights)


# ==========================================
# 5.1 Channel Importance Analysis (NEW)
# ==========================================

def plot_channel_importance_analysis(initial_weights, final_weights, output_folder=None, 
                                     channel_names=None):
    """
    Generates 4 static visualizations for Conv1 channel importance:
    1. Bar chart - Channel importance (L2 norm, mean|w|) - Initial vs Final
    2. Heatmap - Filter x Channel weights (Initial, Final, Delta)
    3. Delta bar chart - Weight change per channel
    4. Box plots - Initial vs Final distribution per channel
    
    Args:
        initial_weights: Dict of initial model weights
        final_weights: Dict of final model weights
        output_folder: Optional folder to save figures
        channel_names: List of channel names (defaults to CHANNEL_NAMES)
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    # Extract conv1 weights: [out_channels=128, in_channels=4, kernel_size=3]
    w_init = initial_weights['conv1']['weight'].detach().cpu().numpy()
    w_final = final_weights['conv1']['weight'].detach().cpu().numpy()
    
    n_channels = len(channel_names)
    
    # --- Compute per-channel metrics ---
    l2_init = np.zeros(n_channels)
    l2_final = np.zeros(n_channels)
    mean_abs_init = np.zeros(n_channels)
    mean_abs_final = np.zeros(n_channels)
    
    for i in range(n_channels):
        ch_init = w_init[:, i, :].ravel()
        ch_final = w_final[:, i, :].ravel()
        
        l2_init[i] = np.linalg.norm(ch_init)
        l2_final[i] = np.linalg.norm(ch_final)
        mean_abs_init[i] = np.mean(np.abs(ch_init))
        mean_abs_final[i] = np.mean(np.abs(ch_final))
    
    # ============================================
    # PLOT 1: Bar Chart - Channel Importance
    # ============================================
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(n_channels)
    width = 0.35
    
    # L2 Norm subplot
    axes[0].bar(x - width/2, l2_init, width, label='Initial', alpha=0.8)
    axes[0].bar(x + width/2, l2_final, width, label='Final', alpha=0.8)
    axes[0].set_xlabel('Input Channel')
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_title('Channel Importance: L2 Norm (Initial vs Final)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channel_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mean |w| subplot
    axes[1].bar(x - width/2, mean_abs_init, width, label='Initial', alpha=0.8)
    axes[1].bar(x + width/2, mean_abs_final, width, label='Final', alpha=0.8)
    axes[1].set_xlabel('Input Channel')
    axes[1].set_ylabel('Mean |Weight|')
    axes[1].set_title('Channel Importance: Mean |Weight| (Initial vs Final)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig1, output_folder, "channel_importance_bar_chart")
    plt.show()
    
    # ============================================
    # PLOT 2: Heatmaps - Filter x Channel
    # ============================================
    # Aggregate weights per filter-channel pair (sum absolute values across kernel)
    heatmap_init = np.sum(np.abs(w_init), axis=2)  # [128, 7]
    heatmap_final = np.sum(np.abs(w_final), axis=2)  # [128, 7]
    heatmap_delta = heatmap_final - heatmap_init
    
    # Use same color scale for init and final
    vmax_abs = max(heatmap_init.max(), heatmap_final.max())
    
    fig2, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    im0 = axes[0].imshow(heatmap_init, aspect='auto', cmap='viridis', vmin=0, vmax=vmax_abs)
    axes[0].set_title('Initial Weights\n(Sum |w| per filter-channel)')
    axes[0].set_xlabel('Input Channel')
    axes[0].set_ylabel('Filter Index')
    axes[0].set_xticks(range(n_channels))
    axes[0].set_xticklabels(channel_names, rotation=45, ha='right')
    plt.colorbar(im0, ax=axes[0], shrink=0.6)
    
    im1 = axes[1].imshow(heatmap_final, aspect='auto', cmap='viridis', vmin=0, vmax=vmax_abs)
    axes[1].set_title('Final Weights\n(Sum |w| per filter-channel)')
    axes[1].set_xlabel('Input Channel')
    axes[1].set_ylabel('Filter Index')
    axes[1].set_xticks(range(n_channels))
    axes[1].set_xticklabels(channel_names, rotation=45, ha='right')
    plt.colorbar(im1, ax=axes[1], shrink=0.6)
    
    # Delta heatmap with diverging colormap
    vmax_delta = max(abs(heatmap_delta.min()), abs(heatmap_delta.max()))
    im2 = axes[2].imshow(heatmap_delta, aspect='auto', cmap='RdBu_r', 
                         vmin=-vmax_delta, vmax=vmax_delta)
    axes[2].set_title('Weight Change (Final - Initial)\n(Sum |w| per filter-channel)')
    axes[2].set_xlabel('Input Channel')
    axes[2].set_ylabel('Filter Index')
    axes[2].set_xticks(range(n_channels))
    axes[2].set_xticklabels(channel_names, rotation=45, ha='right')
    plt.colorbar(im2, ax=axes[2], shrink=0.6)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig2, output_folder, "channel_importance_heatmaps")
    plt.show()
    
    # ============================================
    # PLOT 3: Delta Analysis - Weight Change per Channel
    # ============================================
    delta_l2 = np.zeros(n_channels)
    delta_mean_abs = np.zeros(n_channels)
    
    for i in range(n_channels):
        ch_init = w_init[:, i, :].ravel()
        ch_final = w_final[:, i, :].ravel()
        delta = ch_final - ch_init
        
        delta_l2[i] = np.linalg.norm(delta)
        delta_mean_abs[i] = np.mean(np.abs(delta))
    
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    axes[0].bar(x, delta_l2, color=colors, alpha=0.8)
    axes[0].set_xlabel('Input Channel')
    axes[0].set_ylabel('L2 Norm of (Final - Initial)')
    axes[0].set_title('Weight Change Magnitude per Channel: L2 Norm')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channel_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x, delta_mean_abs, color=colors, alpha=0.8)
    axes[1].set_xlabel('Input Channel')
    axes[1].set_ylabel('Mean |Final - Initial|')
    axes[1].set_title('Weight Change Magnitude per Channel: Mean Absolute')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig3, output_folder, "channel_weight_delta_analysis")
    plt.show()
    
    # ============================================
    # PLOT 4: Box Plots - Initial vs Final per Channel
    # ============================================
    # Create 2x2 grid for 4 channels
    fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_channels):
        ch_init = w_init[:, i, :].ravel()
        ch_final = w_final[:, i, :].ravel()
        
        bp = axes[i].boxplot([ch_init, ch_final], labels=['Initial', 'Final'], 
                             patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[i].set_title(f'{channel_names[i]}')
        axes[i].set_ylabel('Weight Value')
        axes[i].grid(True, alpha=0.3)
    
    # Hide extra subplots if we have fewer channels than grid cells
    for j in range(n_channels, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Conv1 Weight Distributions: Initial vs Final (per Channel)', fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig4, output_folder, "channel_boxplots_initial_vs_final")
    plt.show()
    
    # --- Print summary statistics ---
    print("\n" + "="*70)
    print("CHANNEL IMPORTANCE SUMMARY (Conv1)")
    print("="*70)
    print(f"{'Channel':<16} | {'L2 Init':>10} | {'L2 Final':>10} | {'Delta L2':>10} | {'Change %':>10}")
    print("-"*70)
    for i, name in enumerate(channel_names):
        pct_change = 100 * (l2_final[i] - l2_init[i]) / (l2_init[i] + 1e-8)
        print(f"{name:<16} | {l2_init[i]:>10.4f} | {l2_final[i]:>10.4f} | {delta_l2[i]:>10.4f} | {pct_change:>+10.1f}%")
    print("="*70 + "\n")


def plot_channel_importance_over_epochs(weight_history, output_folder=None, channel_names=None):
    """
    Generates epoch-based visualizations showing how channel importance evolves during training.
    
    1. Line plot - L2 norm per channel over epochs
    2. Line plot - Mean |w| per channel over epochs
    
    Args:
        weight_history: List of dicts with 'epoch' and 'conv1' state_dict per epoch
        output_folder: Optional folder to save figures
        channel_names: List of channel names (defaults to CHANNEL_NAMES)
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    if not weight_history:
        print("Warning: weight_history is empty. Skipping epoch visualization.")
        return
    
    n_epochs = len(weight_history)
    n_channels = len(channel_names)
    
    # Initialize arrays to store metrics per epoch per channel
    l2_over_epochs = np.zeros((n_epochs, n_channels))
    mean_abs_over_epochs = np.zeros((n_epochs, n_channels))
    
    # Extract metrics from each epoch snapshot
    for epoch_idx, snapshot in enumerate(weight_history):
        if snapshot['conv1'] is None:
            continue
            
        w = snapshot['conv1']['weight'].detach().cpu().numpy()  # [128, 4, 3]
        
        for ch_idx in range(n_channels):
            ch_weights = w[:, ch_idx, :].ravel()
            l2_over_epochs[epoch_idx, ch_idx] = np.linalg.norm(ch_weights)
            mean_abs_over_epochs[epoch_idx, ch_idx] = np.mean(np.abs(ch_weights))
    
    epochs = np.arange(1, n_epochs + 1)  # 1-indexed for display
    
    # Color palette for channels
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    # ============================================
    # PLOT 1: L2 Norm over Epochs
    # ============================================
    fig1, ax = plt.subplots(figsize=(12, 6))
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs, l2_over_epochs[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.set_title('Channel Importance Over Training: L2 Norm per Channel', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig1, output_folder, "channel_importance_over_epochs_L2")
    plt.show()
    
    # ============================================
    # PLOT 2: Mean |w| over Epochs
    # ============================================
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs, mean_abs_over_epochs[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean |Weight|', fontsize=12)
    ax.set_title('Channel Importance Over Training: Mean |Weight| per Channel', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig2, output_folder, "channel_importance_over_epochs_mean_abs")
    plt.show()
    
    # ============================================
    # PLOT 3: Normalized view (relative change from epoch 1)
    # ============================================
    fig3, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize by initial value (epoch 0)
    l2_normalized = l2_over_epochs / (l2_over_epochs[0, :] + 1e-8)
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs, l2_normalized[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Initial (1.0)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative L2 Norm (vs Epoch 1)', fontsize=12)
    ax.set_title('Channel Importance Change Over Training (Normalized)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig3, output_folder, "channel_importance_over_epochs_normalized")
    plt.show()
    
    # ============================================
    # PLOT 4: RELATIVE RATE OF CHANGE - L2 Norm
    # Formula: (L2[t] - L2[t-1]) / L2[t-1] * 100 = percentage change
    # ============================================
    fig4, ax = plt.subplots(figsize=(12, 6))
    
    # Compute RELATIVE rate of change: (new - old) / old * 100
    l2_prev = l2_over_epochs[:-1, :]  # All but last
    l2_curr = l2_over_epochs[1:, :]   # All but first
    l2_relative_roc = 100 * (l2_curr - l2_prev) / (l2_prev + 1e-8)  # Percentage change
    
    epochs_roc = np.arange(2, n_epochs + 1)  # Epochs 2 to n
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs_roc, l2_relative_roc[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Rate of Change (%)', fontsize=12)
    ax.set_title('Relative Rate of Change: L2 Norm per Channel (% change per epoch)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig4, output_folder, "channel_relative_rate_of_change_L2")
    plt.show()
    
    # ============================================
    # PLOT 5: RELATIVE RATE OF CHANGE - Mean |w|
    # ============================================
    fig5, ax = plt.subplots(figsize=(12, 6))
    
    # Compute RELATIVE rate of change for mean absolute weights
    mean_prev = mean_abs_over_epochs[:-1, :]
    mean_curr = mean_abs_over_epochs[1:, :]
    mean_abs_relative_roc = 100 * (mean_curr - mean_prev) / (mean_prev + 1e-8)
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs_roc, mean_abs_relative_roc[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Rate of Change (%)', fontsize=12)
    ax.set_title('Relative Rate of Change: Mean |Weight| per Channel (% change per epoch)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig5, output_folder, "channel_relative_rate_of_change_mean_abs")
    plt.show()
    
    # ============================================
    # PLOT 6: CUMULATIVE RELATIVE CHANGE (total % movement)
    # ============================================
    fig6, ax = plt.subplots(figsize=(12, 6))
    
    # Cumulative sum of absolute relative changes
    cumulative_relative_change = np.cumsum(np.abs(l2_relative_roc), axis=0)
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs_roc, cumulative_relative_change[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cumulative |% Change|', fontsize=12)
    ax.set_title('Cumulative Learning Effort: Total Relative Weight Movement per Channel', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig6, output_folder, "channel_cumulative_relative_change")
    plt.show()
    
    # --- Print summary: which channels changed most ---
    print("\n" + "="*60)
    print("CHANNEL LEARNING DYNAMICS SUMMARY")
    print("="*60)
    
    final_l2 = l2_over_epochs[-1, :]
    initial_l2 = l2_over_epochs[0, :]
    pct_changes = 100 * (final_l2 - initial_l2) / (initial_l2 + 1e-8)
    
    # Sort by absolute change
    sorted_idx = np.argsort(np.abs(pct_changes))[::-1]
    
    print("Channels ranked by learning magnitude (largest change first):")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {channel_names[idx]:<16}: {pct_changes[idx]:+.1f}% change")
    print("="*60 + "\n")


# ==========================================
# 5.1.1 Excitation/Inhibition Analysis (Sign-Aware)
# ==========================================

def plot_excitation_inhibition_analysis(initial_weights, final_weights, 
                                         output_folder=None, channel_names=None):
    """
    Sign-aware weight analysis for excitation vs inhibition.
    
    Generates 4 visualizations:
    1. Stacked bar chart - Excitation (positive) vs Inhibition (negative) per channel
    2. E/I Ratio bar chart - Balance for each channel
    3. Scatter plot - Delta Excitation vs Delta Inhibition
    4. Summary table with all metrics
    
    Args:
        initial_weights: Dict of initial model weights
        final_weights: Dict of final model weights
        output_folder: Optional folder to save figures
        channel_names: List of channel names (defaults to CHANNEL_NAMES)
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    # Extract conv1 weights: [out_channels=128, in_channels=4, kernel_size=3]
    w_init = initial_weights['conv1']['weight'].detach().cpu().numpy()
    w_final = final_weights['conv1']['weight'].detach().cpu().numpy()
    
    n_channels = len(channel_names)
    
    # --- Compute per-channel excitation/inhibition metrics ---
    sum_pos_init = np.zeros(n_channels)
    sum_pos_final = np.zeros(n_channels)
    sum_neg_init = np.zeros(n_channels)
    sum_neg_final = np.zeros(n_channels)
    mean_pos_init = np.zeros(n_channels)
    mean_pos_final = np.zeros(n_channels)
    mean_neg_init = np.zeros(n_channels)
    mean_neg_final = np.zeros(n_channels)
    count_pos_init = np.zeros(n_channels)
    count_pos_final = np.zeros(n_channels)
    count_neg_init = np.zeros(n_channels)
    count_neg_final = np.zeros(n_channels)
    
    for i in range(n_channels):
        ch_init = w_init[:, i, :].ravel()
        ch_final = w_final[:, i, :].ravel()
        
        # Initial weights
        pos_mask_init = ch_init > 0
        neg_mask_init = ch_init < 0
        sum_pos_init[i] = ch_init[pos_mask_init].sum() if pos_mask_init.any() else 0
        sum_neg_init[i] = ch_init[neg_mask_init].sum() if neg_mask_init.any() else 0
        mean_pos_init[i] = ch_init[pos_mask_init].mean() if pos_mask_init.any() else 0
        mean_neg_init[i] = ch_init[neg_mask_init].mean() if neg_mask_init.any() else 0
        count_pos_init[i] = pos_mask_init.sum()
        count_neg_init[i] = neg_mask_init.sum()
        
        # Final weights
        pos_mask_final = ch_final > 0
        neg_mask_final = ch_final < 0
        sum_pos_final[i] = ch_final[pos_mask_final].sum() if pos_mask_final.any() else 0
        sum_neg_final[i] = ch_final[neg_mask_final].sum() if neg_mask_final.any() else 0
        mean_pos_final[i] = ch_final[pos_mask_final].mean() if pos_mask_final.any() else 0
        mean_neg_final[i] = ch_final[neg_mask_final].mean() if neg_mask_final.any() else 0
        count_pos_final[i] = pos_mask_final.sum()
        count_neg_final[i] = neg_mask_final.sum()
    
    # E/I Ratio = Sum(w+) / |Sum(w-)|
    ei_ratio_init = sum_pos_init / (np.abs(sum_neg_init) + 1e-8)
    ei_ratio_final = sum_pos_final / (np.abs(sum_neg_final) + 1e-8)
    
    x = np.arange(n_channels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    # ============================================
    # PLOT 1: Stacked Bar Chart - Excitation vs Inhibition
    # ============================================
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initial weights
    axes[0].bar(x, sum_pos_init, label='Excitation (w+)', color='green', alpha=0.7)
    axes[0].bar(x, sum_neg_init, label='Inhibition (w-)', color='red', alpha=0.7)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Input Channel')
    axes[0].set_ylabel('Sum of Weights')
    axes[0].set_title('Initial Weights: Excitation vs Inhibition')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channel_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final weights
    axes[1].bar(x, sum_pos_final, label='Excitation (w+)', color='green', alpha=0.7)
    axes[1].bar(x, sum_neg_final, label='Inhibition (w-)', color='red', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Input Channel')
    axes[1].set_ylabel('Sum of Weights')
    axes[1].set_title('Final Weights: Excitation vs Inhibition')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig1, output_folder, "excitation_inhibition_bar_chart")
    plt.show()
    
    # ============================================
    # PLOT 2: E/I Ratio Bar Chart
    # ============================================
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    width = 0.35
    bars_init = ax.bar(x - width/2, ei_ratio_init, width, label='Initial', alpha=0.8)
    bars_final = ax.bar(x + width/2, ei_ratio_final, width, label='Final', alpha=0.8)
    
    # Color bars based on E/I ratio (green for excitatory, red for inhibitory)
    for bar, ratio in zip(bars_final, ei_ratio_final):
        if ratio > 1:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Neutral (E/I=1)')
    ax.set_xlabel('Input Channel', fontsize=12)
    ax.set_ylabel('E/I Ratio (Sum w+ / |Sum w-|)', fontsize=12)
    ax.set_title('Excitation/Inhibition Ratio per Channel\n(>1 = Excitatory, <1 = Inhibitory)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig2, output_folder, "excitation_inhibition_ratio")
    plt.show()
    
    # ============================================
    # PLOT 3: Scatter Plot - Delta Excitation vs Delta Inhibition
    # ============================================
    delta_exc = sum_pos_final - sum_pos_init
    delta_inh = sum_neg_final - sum_neg_init  # Note: inhibition values are negative
    
    fig3, ax = plt.subplots(figsize=(10, 8))
    
    for i, (de, di, name, color) in enumerate(zip(delta_exc, delta_inh, channel_names, colors)):
        ax.scatter(de, di, s=200, c=[color], label=name, edgecolors='black', linewidth=1)
        ax.annotate(name, (de, di), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Δ Excitation (Final - Initial Sum w+)', fontsize=12)
    ax.set_ylabel('Δ Inhibition (Final - Initial Sum w-)', fontsize=12)
    ax.set_title('Channel Weight Changes: Excitation vs Inhibition\n(Quadrant shows direction of change)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[1]*0.8, 'More Excitatory\nLess Inhibitory', fontsize=9, alpha=0.5, ha='center')
    ax.text(xlim[0]*0.7, ylim[0]*0.8, 'Less Excitatory\nMore Inhibitory', fontsize=9, alpha=0.5, ha='center')
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig3, output_folder, "excitation_inhibition_delta_scatter")
    plt.show()
    
    # ============================================
    # PRINT: Summary Table
    # ============================================
    print("\n" + "="*90)
    print("EXCITATION/INHIBITION ANALYSIS SUMMARY (Conv1)")
    print("="*90)
    print(f"{'Channel':<16} | {'E/I Init':>10} | {'E/I Final':>10} | {'Δ E/I':>10} | {'Interpretation':<20}")
    print("-"*90)
    
    for i, name in enumerate(channel_names):
        delta_ei = ei_ratio_final[i] - ei_ratio_init[i]
        if ei_ratio_final[i] > 1.1:
            interp = "Excitatory"
        elif ei_ratio_final[i] < 0.9:
            interp = "Inhibitory"
        else:
            interp = "Neutral"
        
        print(f"{name:<16} | {ei_ratio_init[i]:>10.3f} | {ei_ratio_final[i]:>10.3f} | {delta_ei:>+10.3f} | {interp:<20}")
    
    print("="*90)
    print("\nDetailed Metrics (Final Weights):")
    print("-"*90)
    print(f"{'Channel':<16} | {'Sum(w+)':>10} | {'Sum(w-)':>10} | {'#Pos':>6} | {'#Neg':>6} | {'Mean(w+)':>10} | {'Mean(w-)':>10}")
    print("-"*90)
    for i, name in enumerate(channel_names):
        print(f"{name:<16} | {sum_pos_final[i]:>10.4f} | {sum_neg_final[i]:>10.4f} | {int(count_pos_final[i]):>6} | {int(count_neg_final[i]):>6} | {mean_pos_final[i]:>10.6f} | {mean_neg_final[i]:>10.6f}")
    print("="*90 + "\n")
    
    return ei_ratio_final  # Return for use in consistency check


def analyze_activation_statistics(
    models,
    X_data,
    channel_names=None,
    output_folder=None,
    sample_limit: int = 2000,
):
    """
    Analyze activation statistics for the first convolutional layer (Conv1) of the CNN.

    Computes:
      - Per-filter stats after BatchNorm + ReLU:
          * mean, std, max activation
          * sparsity (% of activations exactly zero)
          * counts of mostly-dead filters (sparsity >= 90%)
      - Per-input-channel contribution to Conv1 pre-BN output:
          * L2 "energy" contribution per channel
          * mean and std of |activation| per channel
          * fraction of total activation energy per channel

    Also generates:
      - Plot 1: Channel activation energy bar chart
      - Plot 2: Per-channel activation histograms (2x2 grid)
      - Plot 3: Per-filter activation heatmap (filters x channels)
      - Plot 4: Filter sparsity bar chart
      - Plot 5: Per-filter summary stats (mean activation vs sparsity, sorted)
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    if "conv1" not in models:
        print("analyze_activation_statistics currently supports CNN models (conv1 not found). Skipping.")
        return

    # Ensure we have a torch tensor [N, T, C]
    if isinstance(X_data, np.ndarray):
        X = torch.from_numpy(X_data)
    else:
        X = X_data

    if X.ndim != 3:
        raise ValueError(f"Expected X_data with shape [N, T, C], got {tuple(X.shape)}")

    # Optionally subsample for speed
    if sample_limit is not None and X.shape[0] > sample_limit:
        X = X[:sample_limit]

    conv1 = models["conv1"]
    bn1 = models["bn1"]

    device = next(conv1.parameters()).device

    # Be extra safe: snapshot original training/eval modes and
    # force eval mode during analysis so BatchNorm running stats
    # are not updated and dropout (if any) is disabled.
    conv1_was_training = conv1.training
    bn1_was_training = bn1.training
    conv1.eval()
    bn1.eval()

    with torch.no_grad():
        X_dev = X.to(device)
        # Conv1 expects [N, C, T]
        x_perm = X_dev.permute(0, 2, 1)

        # --- Per-filter stats after BN + ReLU ---
        z = conv1(x_perm)
        z_bn = bn1(z)
        a = F.relu(z_bn)  # [N, F, L]

        a_cpu = a.detach().cpu()
        n_samples, n_filters, seq_len = a_cpu.shape

        a_flat = a_cpu.reshape(n_samples * seq_len, n_filters)  # [N*T, F]

        mean_per_filter = a_flat.mean(dim=0).numpy()
        std_per_filter = a_flat.std(dim=0, unbiased=False).numpy()
        max_per_filter = a_flat.max(dim=0).values.numpy()
        sparsity_per_filter = (a_flat == 0).float().mean(dim=0).numpy()

        dead_mask = sparsity_per_filter >= 0.90
        weak_mask = sparsity_per_filter >= 0.70

        n_dead = int(dead_mask.sum())
        n_weak = int(weak_mask.sum())

        # --- Per-input-channel contributions pre-BN ---
        weight = conv1.weight.detach()  # [F, C, K]
        stride = conv1.stride[0]
        padding = conv1.padding[0]
        dilation = conv1.dilation[0]

        n_channels = weight.shape[1]

        energy_per_channel = np.zeros(n_channels, dtype=np.float64)
        mean_abs_per_channel = np.zeros(n_channels, dtype=np.float64)
        std_abs_per_channel = np.zeros(n_channels, dtype=np.float64)
        # For heatmap: mean |activation| per (filter, channel)
        heatmap_per_filter_channel = np.zeros((n_filters, n_channels), dtype=np.float64)

        # Compute contributions channel by channel
        for ch_idx in range(n_channels):
            x_c = x_perm[:, ch_idx : ch_idx + 1, :]  # [N, 1, T]
            w_c = weight[:, ch_idx : ch_idx + 1, :]  # [F, 1, K]
            contrib = F.conv1d(
                x_c,
                w_c,
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )  # [N, F, L]

            contrib_abs = contrib.abs()

            # Scalar summaries per channel
            energy_per_channel[ch_idx] = contrib.pow(2).sum().item()
            mean_abs_per_channel[ch_idx] = contrib_abs.mean().item()
            std_abs_per_channel[ch_idx] = contrib_abs.std(unbiased=False).item()

            # Per-filter, per-channel mean |activation|
            contrib_abs_cpu = contrib_abs.detach().cpu()
            heatmap_per_filter_channel[:, ch_idx] = (
                contrib_abs_cpu.mean(dim=(0, 2)).numpy()
            )  # [F]

        total_energy = float(energy_per_channel.sum() + 1e-8)
        frac_energy_per_channel = 100.0 * energy_per_channel / total_energy

        # --- Per-channel contributions AFTER BN + ReLU (one-channel-at-a-time) ---
        post_heatmap_per_filter_channel = np.zeros((n_filters, n_channels), dtype=np.float64)

        for ch_idx in range(n_channels):
            # Build an input where only channel ch_idx is present
            X_single = torch.zeros_like(X_dev)
            X_single[:, :, ch_idx] = X_dev[:, :, ch_idx]

            x_perm_single = X_single.permute(0, 2, 1)  # [N, C, T]
            z_single = conv1(x_perm_single)
            z_bn_single = bn1(z_single)
            a_single = F.relu(z_bn_single)  # [N, F, L]

            # Mean activation per filter for this channel
            a_single_cpu = a_single.detach().cpu()
            post_heatmap_per_filter_channel[:, ch_idx] = a_single_cpu.mean(dim=(0, 2)).numpy()

    # Restore original training/eval modes
    conv1.train(conv1_was_training)
    bn1.train(bn1_was_training)

    # ============================
    # Plot 1: Channel energy bar chart
    # ============================
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(n_channels)
    ax1.barh(y_pos, frac_energy_per_channel, color="steelblue", alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(channel_names)
    ax1.set_xlabel("Fraction of Conv1 Activation Energy (%)")
    ax1.set_title("Conv1 Channel Contribution (Pre-BN)")
    for i, v in enumerate(frac_energy_per_channel):
        ax1.text(v + 0.5, i, f"{v:5.1f}%", va="center")
    ax1.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if output_folder:
        save_figure(fig1, output_folder, "activation_channel_energy_bar")
    plt.show()

    # ============================
    # Plot 2: Per-channel activation histograms (2x2)
    # ============================
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    axes2 = axes2.flatten()

    for ch_idx in range(n_channels):
        x_c = x_perm[:, ch_idx : ch_idx + 1, :]
        w_c = weight[:, ch_idx : ch_idx + 1, :]
        contrib = F.conv1d(
            x_c,
            w_c,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        vals = contrib.abs().detach().cpu().reshape(-1).numpy()
        # Optional subsampling for very large arrays
        if vals.size > 200000:
            idx = np.random.choice(vals.size, size=200000, replace=False)
            vals = vals[idx]

        ax = axes2[ch_idx]
        ax.hist(vals, bins=80, alpha=0.8, color="tab:blue")
        ax.set_title(f"Channel {ch_idx}: {channel_names[ch_idx]}")
        ax.set_xlabel("|activation|")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (in case n_channels < 4)
    for j in range(n_channels, len(axes2)):
        axes2[j].axis("off")

    plt.suptitle("Conv1 Per-Channel Activation Distributions", fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig2, output_folder, "activation_per_channel_histograms")
    plt.show()

    # ============================
    # Plot 3: Per-filter activation heatmap (pre-BN)
    # ============================
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im = ax3.imshow(
        heatmap_per_filter_channel,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax3.set_xlabel("Input Channel")
    ax3.set_ylabel("Filter Index")
    ax3.set_xticks(np.arange(n_channels))
    ax3.set_xticklabels(channel_names, rotation=45, ha="right")
    ax3.set_title("Conv1 Mean |Activation| per Filter-Channel")
    plt.colorbar(im, ax=ax3, shrink=0.7)
    plt.tight_layout()
    if output_folder:
        save_figure(fig3, output_folder, "activation_filter_channel_heatmap")
    plt.show()

    # ============================
    # EXTRA Plot: Per-filter post-BN+ReLU activation heatmap (one-channel-at-a-time)
    # ============================
    fig3b, ax3b = plt.subplots(figsize=(8, 6))
    im_post = ax3b.imshow(
        post_heatmap_per_filter_channel,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax3b.set_xlabel("Input Channel")
    ax3b.set_ylabel("Filter Index")
    ax3b.set_xticks(np.arange(n_channels))
    ax3b.set_xticklabels(channel_names, rotation=45, ha="right")
    ax3b.set_title("Conv1 Mean Activation per Filter-Channel\n(Post BN+ReLU, One Channel at a Time)")
    plt.colorbar(im_post, ax=ax3b, shrink=0.7)
    plt.tight_layout()
    if output_folder:
        save_figure(fig3b, output_folder, "activation_post_bn_relu_filter_channel_heatmap")
    plt.show()

    # ============================
    # Plot 4: Filter sparsity bar chart
    # ============================
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    x_idx = np.arange(n_filters)
    ax4.bar(x_idx, sparsity_per_filter, color="tab:gray", alpha=0.8)
    # Highlight mostly-dead filters
    if n_dead > 0:
        ax4.bar(
            x_idx[dead_mask],
            sparsity_per_filter[dead_mask],
            color="tab:red",
            alpha=0.9,
            label="Dead (>= 90% zero)",
        )
    ax4.axhline(0.9, linestyle="--", color="red", alpha=0.6, label="90% sparsity")
    ax4.set_xlabel("Filter Index")
    ax4.set_ylabel("Sparsity (fraction of zeros)")
    ax4.set_title("Conv1 Filter Sparsity")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend()
    plt.tight_layout()
    if output_folder:
        save_figure(fig4, output_folder, "activation_filter_sparsity")
    plt.show()

    # ============================
    # Plot 5: Per-filter summary stats (sorted by mean activation)
    # ============================
    order = np.argsort(-mean_per_filter)  # descending
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(
        mean_per_filter[order],
        label="Mean activation",
        color="tab:blue",
        linewidth=2,
    )
    ax6 = ax5.twinx()
    ax6.plot(
        sparsity_per_filter[order],
        label="Sparsity",
        color="tab:orange",
        linewidth=2,
    )
    ax5.set_xlabel("Filters (sorted by mean activation)")
    ax5.set_ylabel("Mean activation")
    ax6.set_ylabel("Sparsity (fraction of zeros)")
    ax5.set_title("Conv1 Filter Activity vs Sparsity")
    ax5.grid(True, axis="y", alpha=0.3)
    # Build a combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax6.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    if output_folder:
        save_figure(fig5, output_folder, "activation_filter_summary")
    plt.show()

    # ============================
    # Printed summary
    # ============================
    print("\n" + "=" * 80)
    print("CONV1 ACTIVATION STATISTICS")
    print("=" * 80)
    print("Per-channel contributions (pre-BN):")
    for ch_idx, name in enumerate(channel_names):
        print(
            f"  {ch_idx}: {name:12s} | "
            f"energy_frac={frac_energy_per_channel[ch_idx]:6.2f}%  "
            f"mean|a|={mean_abs_per_channel[ch_idx]:.4e}  "
            f"std|a|={std_abs_per_channel[ch_idx]:.4e}"
        )

    print("\nPer-filter summary after BN + ReLU:")
    print(f"  Total filters: {n_filters}")
    print(f"  Dead filters (sparsity >= 90%): {n_dead}")
    print(f"  Weak filters (sparsity >= 70%): {n_weak}")
    print("=" * 80 + "\n")


def plot_excitation_inhibition_over_epochs(weight_history, output_folder=None, channel_names=None):
    """
    Track excitation and inhibition strength per channel over training epochs.
    
    Generates 2 visualizations:
    1. Line plot - Sum(w+) and Sum(w-) per channel over epochs
    2. Line plot - E/I Ratio per channel over epochs
    
    Args:
        weight_history: List of dicts with 'epoch' and 'conv1' state_dict per epoch
        output_folder: Optional folder to save figures
        channel_names: List of channel names (defaults to CHANNEL_NAMES)
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    if not weight_history:
        print("Warning: weight_history is empty. Skipping E/I epoch visualization.")
        return
    
    n_epochs = len(weight_history)
    n_channels = len(channel_names)
    
    # Initialize arrays to store metrics per epoch per channel
    sum_pos_over_epochs = np.zeros((n_epochs, n_channels))
    sum_neg_over_epochs = np.zeros((n_epochs, n_channels))
    ei_ratio_over_epochs = np.zeros((n_epochs, n_channels))
    
    # Extract metrics from each epoch snapshot
    for epoch_idx, snapshot in enumerate(weight_history):
        if snapshot['conv1'] is None:
            continue
            
        w = snapshot['conv1']['weight'].detach().cpu().numpy()  # [128, 4, 3]
        
        for ch_idx in range(n_channels):
            ch_weights = w[:, ch_idx, :].ravel()
            
            pos_mask = ch_weights > 0
            neg_mask = ch_weights < 0
            
            sum_pos = ch_weights[pos_mask].sum() if pos_mask.any() else 0
            sum_neg = ch_weights[neg_mask].sum() if neg_mask.any() else 0
            
            sum_pos_over_epochs[epoch_idx, ch_idx] = sum_pos
            sum_neg_over_epochs[epoch_idx, ch_idx] = sum_neg
            ei_ratio_over_epochs[epoch_idx, ch_idx] = sum_pos / (np.abs(sum_neg) + 1e-8)
    
    epochs = np.arange(1, n_epochs + 1)  # 1-indexed for display
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    # ============================================
    # PLOT 1: Excitation and Inhibition Over Epochs
    # ============================================
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Excitation (Sum w+)
    for ch_idx, ch_name in enumerate(channel_names):
        axes[0].plot(epochs, sum_pos_over_epochs[:, ch_idx], 
                     label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Sum of Positive Weights', fontsize=12)
    axes[0].set_title('Excitation Strength Over Training', fontsize=14)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([1, n_epochs])
    
    # Inhibition (Sum w-)
    for ch_idx, ch_name in enumerate(channel_names):
        axes[1].plot(epochs, sum_neg_over_epochs[:, ch_idx], 
                     label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Sum of Negative Weights', fontsize=12)
    axes[1].set_title('Inhibition Strength Over Training', fontsize=14)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([1, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig1, output_folder, "excitation_inhibition_over_epochs")
    plt.show()
    
    # ============================================
    # PLOT 2: E/I Ratio Over Epochs
    # ============================================
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    for ch_idx, ch_name in enumerate(channel_names):
        ax.plot(epochs, ei_ratio_over_epochs[:, ch_idx], 
                label=ch_name, color=colors[ch_idx], linewidth=2, marker='o', markersize=3)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Neutral (E/I=1)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('E/I Ratio', fontsize=12)
    ax.set_title('Excitation/Inhibition Ratio Over Training\n(>1 = Excitatory, <1 = Inhibitory)', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, n_epochs])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig2, output_folder, "ei_ratio_over_epochs")
    plt.show()
    
    # --- Print summary: E/I changes ---
    print("\n" + "="*70)
    print("E/I RATIO DYNAMICS SUMMARY")
    print("="*70)
    
    initial_ei = ei_ratio_over_epochs[0, :]
    final_ei = ei_ratio_over_epochs[-1, :]
    delta_ei = final_ei - initial_ei
    
    print(f"{'Channel':<16} | {'E/I Epoch 1':>12} | {'E/I Final':>12} | {'Δ E/I':>10} | {'Trend':<15}")
    print("-"*70)
    for i, name in enumerate(channel_names):
        if delta_ei[i] > 0.05:
            trend = "→ More Excitatory"
        elif delta_ei[i] < -0.05:
            trend = "→ More Inhibitory"
        else:
            trend = "→ Stable"
        print(f"{name:<16} | {initial_ei[i]:>12.3f} | {final_ei[i]:>12.3f} | {delta_ei[i]:>+10.3f} | {trend:<15}")
    print("="*70 + "\n")


# ==========================================
# 5.1.2 Validation Test Cases
# ==========================================

def channel_ablation_study(models, forward_fn, X_test, y_test, channel_names=None, output_folder=None):
    """
    Ablate each channel using appropriate strategy and measure accuracy drop.
    
    Uses ACTUAL TEST DATA (X_test, y_test from your test CSV files).
    Runs on ALL samples (no filtering) since test data has natural missingness that
    varies per file. Reports missingness statistics so you know how many samples
    had the target channel already missing (redundant ablation) or other channels missing.
    
    For clean ablation metrics, use the robustness suite on validation data instead.
    
    4-channel architecture [Current, Pressure, Radiation, Prev_A1]
    Strategy by channel type:
    - Sensors (indices 0, 1, 2): Set to missing_value (-100) to simulate missing sensor
    - Prev_A1 (index 3): Shuffle across samples (preserves distribution)
    
    Args:
        models: Dict of model layers
        forward_fn: Forward function
        X_test: Test data tensor [N, seq_len, 4] - from actual test files
        y_test: Test labels tensor [N, 1] - from actual test files
        channel_names: List of channel names
        output_folder: Optional folder to save figure
    
    Returns:
        results: Dict with ablation results per channel (includes missingness stats for sensors)
    """
    missing_value = -100.0  # Default missing value
    
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    # Set eval mode
    for layer in models.values():
        layer.eval()
    
    n_total = X_test.shape[0]
    
    # Compute baseline accuracy over all samples
    with torch.no_grad():
        baseline_logits = forward_fn(models, X_test)
        baseline_preds = (baseline_logits > 0).float()
        baseline_acc = (baseline_preds == y_test).float().mean().item()
    
    # Channel index mapping (4 channels):
    # 0: Current, 1: Pressure, 2: Radiation, 3: Prev_A1
    sensor_channels = [0, 1, 2]
    
    # Detect pre-existing missing data per sample per sensor channel
    missing_mask = get_missing_mask(X_test, missing_value=missing_value)  # [N, 3] bool
    
    results = {}
    accuracy_drops = []
    
    print("\n" + "="*120)
    print("CHANNEL ABLATION STUDY (test data -- all samples, with missingness reporting)")
    print("="*120)
    print(f"Total test samples: {n_total} | Overall Baseline Accuracy: {baseline_acc:.4f}")
    print("\nPre-existing missingness in test data:")
    for s_idx, s_name in enumerate(['Current', 'Pressure', 'Radiation']):
        n_missing = missing_mask[:, s_idx].sum().item()
        pct = 100.0 * n_missing / n_total if n_total > 0 else 0
        print(f"  {s_name}: {int(n_missing)}/{n_total} samples already missing ({pct:.1f}%)")
    
    any_missing = missing_mask.any(dim=1).sum().item()
    all_clean = n_total - any_missing
    print(f"  Fully clean (no sensor missing): {all_clean}/{n_total} ({100.0*all_clean/n_total:.1f}%)")
    
    print("-"*120)
    print(f"{'Channel':<16} | {'Method':<12} | {'Ablated Acc':>11} | {'Drop':>8} | {'Importance':<10} | {'N_total':>7} | {'N_tgt_miss':>10} | {'N_oth_miss':>10} | {'N_clean':>7}")
    print("-"*120)
    
    for ch_idx, ch_name in enumerate(channel_names):
        X_ablated = X_test.clone()
        
        if ch_idx in sensor_channels:
            X_ablated[:, :, ch_idx] = missing_value
            method = f'set_{missing_value}'
            
            # Missingness stats (informational, not used for filtering)
            target_already_missing = missing_mask[:, ch_idx]
            other_chs = [j for j in range(3) if j != ch_idx]
            other_missing = missing_mask[:, other_chs].any(dim=1)
            
            n_target_miss = int(target_already_missing.sum().item())
            n_other_miss = int(other_missing.sum().item())
            n_clean = int((~target_already_missing & ~other_missing).sum().item())
            
            with torch.no_grad():
                logits = forward_fn(models, X_ablated)
                preds = (logits > 0).float()
                ablated_acc = (preds == y_test).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            accuracy_drops.append(drop)
            
            if drop > 0.05:
                importance = "HIGH"
            elif drop > 0.01:
                importance = "MEDIUM"
            elif drop > 0:
                importance = "LOW"
            else:
                importance = "NONE/NEG"
            
            print(f"{ch_name:<16} | {method:<12} | {ablated_acc:>11.4f} | {drop:>+8.4f} | {importance:<10} | {n_total:>7} | {n_target_miss:>10} | {n_other_miss:>10} | {n_clean:>7}")
            
            results[ch_name] = {
                'baseline': baseline_acc,
                'ablated': ablated_acc,
                'drop': drop,
                'method': method,
                'importance': importance,
                'n_total': n_total,
                'n_target_already_missing': n_target_miss,
                'n_other_missing': n_other_miss,
                'n_fully_clean': n_clean,
            }
        
        elif ch_idx == 3:
            # Prev_A1: Shuffle across samples (preserves distribution)
            n_samples = X_ablated.shape[0]
            perm = torch.randperm(n_samples)
            X_ablated[:, :, ch_idx] = X_ablated[perm, :, ch_idx]
            method = 'shuffle'
            
            with torch.no_grad():
                logits = forward_fn(models, X_ablated)
                preds = (logits > 0).float()
                ablated_acc = (preds == y_test).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            accuracy_drops.append(drop)
            
            if drop > 0.05:
                importance = "HIGH"
            elif drop > 0.01:
                importance = "MEDIUM"
            elif drop > 0:
                importance = "LOW"
            else:
                importance = "NONE/NEG"
            
            print(f"{ch_name:<16} | {method:<12} | {ablated_acc:>11.4f} | {drop:>+8.4f} | {importance:<10} | {n_total:>7} | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")
            
            results[ch_name] = {
                'baseline': baseline_acc,
                'ablated': ablated_acc,
                'drop': drop,
                'method': method,
                'importance': importance,
                'n_total': n_total,
                'n_target_already_missing': 0,
                'n_other_missing': 0,
                'n_fully_clean': n_total,
            }
        
        else:
            # Fallback
            X_ablated[:, :, ch_idx] = missing_value
            method = f'set_{missing_value}'
            
            with torch.no_grad():
                logits = forward_fn(models, X_ablated)
                preds = (logits > 0).float()
                ablated_acc = (preds == y_test).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            accuracy_drops.append(drop)
            importance = "HIGH" if drop > 0.05 else "MEDIUM" if drop > 0.01 else "LOW" if drop > 0 else "NONE/NEG"
            
            print(f"{ch_name:<16} | {method:<12} | {ablated_acc:>11.4f} | {drop:>+8.4f} | {importance:<10} | {n_total:>7} | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")
            
            results[ch_name] = {
                'baseline': baseline_acc,
                'ablated': ablated_acc,
                'drop': drop,
                'method': method,
                'importance': importance,
                'n_total': n_total,
                'n_target_already_missing': 0,
                'n_other_missing': 0,
                'n_fully_clean': n_total,
            }
    
    print("="*120)
    print("Note: Test data ablation uses ALL samples. Missingness columns are informational:")
    print("      N_tgt_miss = target channel was already -100 (ablation redundant for those samples)")
    print("      N_oth_miss = another sensor channel was already -100")
    print("      N_clean = samples where neither target nor other sensors were pre-missing")
    print("      For clean ablation metrics, see the VALIDATION robustness report.")
    print("="*120)
    
    # --- Plot ablation results ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(channel_names))
    colors = ['green' if d > 0.05 else 'orange' if d > 0.01 else 'gray' for d in accuracy_drops]
    
    bars = ax.bar(x, accuracy_drops, color=colors, alpha=0.8, edgecolor='black')
    
    # Annotate bars with missingness info for sensor channels
    for i, (bar, ch_name) in enumerate(zip(bars, channel_names)):
        r = results[ch_name]
        n_tgt = r.get('n_target_already_missing', 0)
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'n={r["n_total"]}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        if n_tgt > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.012,
                    f'({n_tgt} already missing)', ha='center', va='bottom', fontsize=7, color='red')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High importance threshold (5%)')
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium importance threshold (1%)')
    
    ax.set_xlabel('Input Channel', fontsize=12)
    ax.set_ylabel('Accuracy Drop (Baseline - Ablated)', fontsize=12)
    ax.set_title('Channel Ablation Study: Accuracy Drop per Channel (Test Data, All Samples)\n(Higher = More Important | Red annotation = samples where target was already missing)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig, output_folder, "ablation_study_results")
    plt.show()
    
    return results


def print_ei_consistency_check(ei_ratios, channel_names=None):
    """
    Compare E/I ratios against expected physics and flag anomalies.
    
    Args:
        ei_ratios: Array of E/I ratios per channel (from plot_excitation_inhibition_analysis)
        channel_names: List of channel names
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    # Expected behavior based on physics (4 channels, no masks)
    expected = {
        'Current': ('variable', 'Context-dependent'),
        'Pressure': ('inhibitory', 'High pressure → vacuum breakdown risk'),
        'Radiation': ('inhibitory', 'High radiation → field emission warning'),
        'Prev_A1': ('variable', 'Momentum vs caution behavior'),
    }
    
    print("\n" + "="*80)
    print("E/I CONSISTENCY CHECK (vs Expected Physics)")
    print("="*80)
    print(f"{'Channel':<16} | {'E/I Ratio':>10} | {'Observed':>12} | {'Expected':>12} | {'Status':<10}")
    print("-"*80)
    
    anomalies = []
    
    for i, name in enumerate(channel_names):
        ratio = ei_ratios[i]
        
        # Classify observed behavior
        if ratio > 1.1:
            observed = "Excitatory"
        elif ratio < 0.9:
            observed = "Inhibitory"
        else:
            observed = "Neutral"
        
        exp_type, exp_reason = expected.get(name, ('unknown', ''))
        
        # Check consistency
        if exp_type == 'inhibitory' and observed != 'Inhibitory':
            status = "ANOMALY"
            anomalies.append((name, observed, exp_type, exp_reason))
        elif exp_type == 'neutral' and observed != 'Neutral':
            status = "CHECK"
            anomalies.append((name, observed, exp_type, exp_reason))
        elif exp_type == 'variable':
            status = "OK"
        else:
            status = "OK"
        
        print(f"{name:<16} | {ratio:>10.3f} | {observed:>12} | {exp_type:>12} | {status:<10}")
    
    print("="*80)
    
    if anomalies:
        print("\n>>> ANOMALIES DETECTED:")
        for name, observed, expected_type, reason in anomalies:
            print(f"  - {name}: Observed {observed}, Expected {expected_type}")
            print(f"    Reason: {reason}")
    else:
        print("\n>>> All channels consistent with expected physics!")
    
    print("")
    return anomalies


def create_synthetic_test_cases(scaler, seq_len):
    """
    Generate synthetic test cases with known expected outputs.
    
    Args:
        scaler: Fitted StandardScaler for sensor values
        seq_len: Sequence length (e.g., 20)
    
    Returns:
        List of (name, sequence, expected_output) tuples
    """
    test_cases = []
    
    # Helper to create a constant sequence
    def make_seq(current, pressure, radiation, prev_a1=0):
        """Create a constant sequence with given values."""
        scaled = scaler.transform([[current, pressure, radiation]])[0]
        step = np.array([
            scaled[0],    # Current
            scaled[1],    # Pressure  
            scaled[2],    # Radiation
            prev_a1       # Prev_A1
        ], dtype=np.float32)
        return np.tile(step, (seq_len, 1))
    
    # Case 1: All quiet (low sensor values) → expect INCREASE (1)
    quiet_seq = make_seq(current=0.1, pressure=1e-9, radiation=0.01)
    test_cases.append(('all_quiet', quiet_seq, 1, "Low sensors → safe to increase"))
    
    # Case 2: Pressure spike at end → expect HOLD (0)
    pressure_spike_seq = quiet_seq.copy()
    high_pressure_scaled = scaler.transform([[0, 1e-6, 0]])[0, 1]
    pressure_spike_seq[-1, 1] = high_pressure_scaled  # Pressure channel (index 1 in 4-channel format)
    test_cases.append(('pressure_spike_end', pressure_spike_seq, 0, "High pressure → hold"))
    
    # Case 3: Radiation spike at end → expect HOLD (0)
    radiation_spike_seq = quiet_seq.copy()
    high_radiation_scaled = scaler.transform([[0, 0, 100]])[0, 2]
    radiation_spike_seq[-1, 2] = high_radiation_scaled  # Radiation channel (index 2 in 4-channel format)
    test_cases.append(('radiation_spike_end', radiation_spike_seq, 0, "High radiation → hold"))
    
    # Case 4: Current spike at end → expect HOLD (0)
    current_spike_seq = quiet_seq.copy()
    high_current_scaled = scaler.transform([[100, 0, 0]])[0, 0]
    current_spike_seq[-1, 0] = high_current_scaled  # Current channel
    test_cases.append(('current_spike_end', current_spike_seq, 0, "High current → hold"))
    
    # Case 5: Previous was increase, all quiet → could be 0 or 1 (momentum test)
    momentum_seq = make_seq(current=0.1, pressure=1e-9, radiation=0.01, prev_a1=1)
    test_cases.append(('momentum_test', momentum_seq, None, "Tests if model has momentum behavior"))
    
    return test_cases


def run_synthetic_tests(models, forward_fn, test_cases):
    """
    Run synthetic tests and report pass/fail.
    
    Args:
        models: Dict of model layers
        forward_fn: Forward function
        test_cases: List of (name, sequence, expected, description) tuples
    
    Returns:
        results: Dict with test results
    """
    # Set eval mode
    for layer in models.values():
        layer.eval()
    
    print("\n" + "="*80)
    print("SYNTHETIC TEST CASES")
    print("="*80)
    print(f"{'Test Name':<25} | {'Expected':>10} | {'Predicted':>10} | {'Logit':>10} | {'Status':<8}")
    print("-"*80)
    
    results = {}
    passed = 0
    total_graded = 0
    
    for name, seq, expected, description in test_cases:
        X = torch.FloatTensor(seq).unsqueeze(0)
        
        with torch.no_grad():
            logit = forward_fn(models, X).item()
            pred = 1 if logit > 0 else 0
        
        if expected is None:
            # Informational test (no expected value)
            status = "INFO"
            expected_str = "N/A"
        else:
            total_graded += 1
            if pred == expected:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
            expected_str = str(expected)
        
        results[name] = {
            'expected': expected,
            'predicted': pred,
            'logit': logit,
            'status': status,
            'description': description
        }
        
        print(f"{name:<25} | {expected_str:>10} | {pred:>10} | {logit:>10.4f} | {status:<8}")
    
    print("-"*80)
    if total_graded > 0:
        print(f"Passed: {passed}/{total_graded} ({100*passed/total_graded:.1f}%)")
    print("="*80 + "\n")
    
    return results


def temporal_sensitivity_test(models, forward_fn, scaler, seq_len, output_folder=None):
    """
    Test if spikes at different positions have different effects.
    Recent spikes (end of sequence) should matter more.
    
    Args:
        models: Dict of model layers
        forward_fn: Forward function
        scaler: Fitted StandardScaler
        seq_len: Sequence length
        output_folder: Optional folder to save figure
    
    Returns:
        results: Dict with logits for each spike position
    """
    # Set eval mode
    for layer in models.values():
        layer.eval()
    
    # Create base quiet sequence
    def make_base_seq():
        scaled = scaler.transform([[0.1, 1e-9, 0.01]])[0]
        step = np.array([
            scaled[0],    # Current
            scaled[1],    # Pressure  
            scaled[2],    # Radiation
            0             # Prev_A1
        ], dtype=np.float32)
        return np.tile(step, (seq_len, 1))
    
    # Get high values for spikes
    high_pressure_scaled = scaler.transform([[0, 1e-6, 0]])[0, 1]
    
    # Test spike at different positions
    spike_positions = list(range(seq_len))  # All positions
    results = {'position': [], 'logit': []}
    
    for spike_pos in spike_positions:
        test_seq = make_base_seq()
        test_seq[spike_pos, 2] = high_pressure_scaled  # Pressure spike
        
        X = torch.FloatTensor(test_seq).unsqueeze(0)
        with torch.no_grad():
            logit = forward_fn(models, X).item()
        
        results['position'].append(spike_pos)
        results['logit'].append(logit)
    
    # Print summary
    print("\n" + "="*60)
    print("TEMPORAL SENSITIVITY TEST (Pressure Spike)")
    print("="*60)
    print("Lower logit = more inhibition from spike at that position")
    print("-"*60)
    
    # Show key positions
    key_positions = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
    print(f"{'Position':<12} | {'Logit':>12} | {'Interpretation':<25}")
    print("-"*60)
    for pos in key_positions:
        logit = results['logit'][pos]
        if pos == 0:
            interp = "Oldest timestep"
        elif pos == seq_len - 1:
            interp = "Most recent timestep"
        else:
            interp = f"Middle ({pos}/{seq_len-1})"
        print(f"{pos:<12} | {logit:>12.4f} | {interp:<25}")
    
    # Check if recent spikes matter more
    early_avg = np.mean(results['logit'][:seq_len//3])
    late_avg = np.mean(results['logit'][2*seq_len//3:])
    
    print("-"*60)
    print(f"Average logit (early positions 0-{seq_len//3-1}): {early_avg:.4f}")
    print(f"Average logit (late positions {2*seq_len//3}-{seq_len-1}): {late_avg:.4f}")
    
    if late_avg < early_avg:
        print(">>> Model is MORE sensitive to RECENT spikes (expected behavior)")
    else:
        print(">>> Model is MORE sensitive to OLD spikes (unexpected)")
    print("="*60 + "\n")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(results['position'], results['logit'], 'b-o', markersize=4, linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Decision boundary (logit=0)')
    ax.fill_between(results['position'], results['logit'], 0, 
                    where=[l < 0 for l in results['logit']], 
                    alpha=0.3, color='red', label='Predicts HOLD')
    ax.fill_between(results['position'], results['logit'], 0,
                    where=[l >= 0 for l in results['logit']],
                    alpha=0.3, color='green', label='Predicts INCREASE')
    
    ax.set_xlabel('Spike Position in Sequence (0=oldest, 19=most recent)', fontsize=12)
    ax.set_ylabel('Model Logit', fontsize=12)
    ax.set_title('Temporal Sensitivity: Effect of Pressure Spike Position\n(Lower logit = stronger inhibition)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, seq_len-1])
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig, output_folder, "temporal_sensitivity_test")
    plt.show()
    
    return results


# ==========================================
# 5.2 Evaluation Metrics 
# ==========================================


## Evaluation Metrics are only for the test set after training is complete.
## All the functions below implement evaluation metrics and helps with visualizations.
## They can be commented out in the __main__ script if not needed.
## SOME FUNCTIONS STILL NEED TO BE ADDED OR FIXED FOR BETTER VISUALIZATIONS.

"""

    Evaluates the trained model on the test set.
    Calculates loss, accuracy, precision, recall, confusion matrix.
    Inputs: trained models dict, forward function, data dict, pos_weight tensor
    Outputs: prints evaluation metrics.

   """

def evaluate_test_set(models, forward_fn, data, pos_weight):

   


    print("\n" + "="*30)
    print("EVALUATING ON TEST SET")
    print("="*30)
    
    X_test, y_test = data['X_test'], data['y_test']
    
    # Safety check if test set is empty
    if len(X_test) == 0:
        print("Test set is empty. Cannot evaluate.")
        return

    # Set Eval Mode
    models['conv1'].eval(); models['bn1'].eval()
    models['conv2'].eval(); models['bn2'].eval()
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        logits = forward_fn(models, X_test)
        loss = criterion(logits, y_test)
        test_loss = loss.item()
        
        # Convert logits to 0 or 1
        preds = (logits > 0).float().cpu().numpy()
        y_true = y_test.cpu().numpy()
        
    # Calculate Metrics
    accuracy = (preds == y_true).mean()
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    
    # Counts
    real_vals, real_counts = np.unique(y_true, return_counts=True)
    pred_vals, pred_counts = np.unique(preds, return_counts=True)
    real_dict = dict(zip(real_vals, real_counts))
    pred_dict = dict(zip(pred_vals, pred_counts))
    
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print("-" * 30)
    print(f"Real Distribution:      {real_dict}")
    print(f"Predicted Distribution: {pred_dict}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(cm)
    print("="*30 + "\n")


    

def extract_last_step_three_features(X, scaler):
    """
    Extracts (current, pressure, radiation) from the LAST timestep of each sequence.
    X shape: (N, Seq, 4)
    Returns raw (inverse-scaled) physical values.

    Used by: plot_3d_real_labels(), plot_3d_predicted_labels()
    The plot_3d_* functions still need to be fixed or understood properly for better visualization.
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = X

    last = X_np[:, -1, :]  # last timestep
    vals_scaled = last[:, [0, 1, 2]]  # scaled current, pressure, radiation (4-channel format)

    # Inverse transform back to physical units
    vals_real = scaler.inverse_transform(vals_scaled)

    current = vals_real[:, 0]
    pressure = vals_real[:, 1]
    radiation = vals_real[:, 2]
    return current, pressure, radiation

from mpl_toolkits.mplot3d import Axes3D

##############################################################################
#### 3d PLOTTING for vizualtion. NEED TO BE FIXED OR UNDERSTOOD PROPERLY.
#### Ignore from Line 886 until Line 1120.
###############################################################################
def _set_3d_origin_and_view(ax, x, y, z, start_from_zero=True, pad=0.05, elev=25, azim=35):
    """
    Helper to control axis limits and view orientation.
    - start_from_zero=True: forces axes to include 0 and usually start at 0 if data is positive.
    - elev/azim: camera angle.
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # padding
    def pad_range(vmin, vmax):
        rng = vmax - vmin
        if rng == 0:
            rng = 1.0
        return vmin - pad * rng, vmax + pad * rng

    x_lo, x_hi = pad_range(x_min, x_max)
    y_lo, y_hi = pad_range(y_min, y_max)
    z_lo, z_hi = pad_range(z_min, z_max)

    if start_from_zero:
        # If data is mostly positive, anchor lower bound at 0.
        # If negatives exist, still include 0 without hiding data.
        x_lo = min(0, x_lo)
        y_lo = min(0, y_lo)
        z_lo = min(0, z_lo)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)

    # Draw origin reference lines (visual cue)
    ax.plot([0, x_hi], [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, y_hi], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, 0], [0, z_hi], linewidth=1)

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)


def plot_3d_real_labels(X, y, scaler, title="Real Labels in Feature Space",
                        start_from_zero=True, elev=25, azim=35):
    current, pressure, radiation = extract_last_step_three_features(X, scaler)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy().flatten()
    else:
        y_np = y.flatten()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    idx0 = (y_np == 0)
    idx1 = (y_np == 1)

    ax.scatter(current[idx0], radiation[idx0], pressure[idx0],  alpha=0.35, label="0 Stable")
    ax.scatter(current[idx1], radiation[idx1], pressure[idx1],  alpha=0.55, label="1 Increase")

    ax.set_xlabel("Current (real units)")
    ax.set_ylabel("Radiation (real units)")
    ax.set_zlabel("Pressure (real units)")
    ax.set_title(title)
    ax.legend()

    _set_3d_origin_and_view(ax, current, radiation, pressure,
                            start_from_zero=start_from_zero, elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


def plot_3d_predicted_labels(models, forward_fn, X, y, scaler,
                             title="Predicted Labels in Feature Space",
                             start_from_zero=True, elev=25, azim=35):
    for layer in models.values():
        layer.eval()

    with torch.no_grad():
        logits = forward_fn(models, X)
        preds = (logits > 0).float()

    current, pressure, radiation = extract_last_step_three_features(X, scaler)

    preds_np = preds.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    idx0 = (preds_np == 0)
    idx1 = (preds_np == 1)

    ax.scatter(current[idx0], radiation[idx0], pressure[idx0],  alpha=0.35, label="Pred 0 Stable")
    ax.scatter(current[idx1], radiation[idx1], pressure[idx1],  alpha=0.55, label="Pred 1 Increase")

    ax.set_xlabel("Current (real units)")
    ax.set_ylabel("Radiation (real units)")
    ax.set_zlabel("Pressure (real units)")
    ax.set_title(title + f"\n(Accuracy on this set: {(preds_np == y_np).mean():.3f})")
    ax.legend()

    _set_3d_origin_and_view(ax, current, radiation, pressure,
                            start_from_zero=start_from_zero, elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


def make_constant_sequence(curr, pres, rad, scaler, seq_len):
    """
    Build a synthetic constant sequence compatible with 4-channel input.
    Channels: [Current, Pressure, Radiation, Prev_A1]
    """
    scaled = scaler.transform([[curr, pres, rad]])[0]
    step = np.array([scaled[0], scaled[1], scaled[2], 0.0], dtype=np.float32)
    seq = np.tile(step, (seq_len, 1))  # (Seq, 4)
    return seq

def plot_decision_boundary_slices(models, forward_fn, data, scaler, seq_len,
                                  n_grid=60, band_frac=0.10, overlay="real"):
    """
    Approx decision boundary in Current vs Pressure plane,
    with scatter overlay for REAL or PREDICTED labels.

    overlay:
      - "real"  -> color scatter by true labels
      - "pred"  -> color scatter by model predictions
      - "both"  -> plot two figures per slice
    band_frac:
      fraction of radiation range used as +/- band around each slice level
      e.g. 0.10 means +-10% of rad_range
    """

    # Combine reference set to get realistic scatter distribution
    X_ref = torch.cat([data['X_train'], data['X_val']], dim=0)
    y_ref = torch.cat([data['y_train'], data['y_val']], dim=0)

    # Extract last-step physical values
    cur_all, pre_all, rad_all = extract_last_step_three_features(X_ref, scaler)

    y_all = y_ref.detach().cpu().numpy().flatten()

    # Model predictions for overlay="pred" or "both"
    pred_all = None
    if overlay in ("pred", "both"):
        for layer in models.values():
            layer.eval()
        with torch.no_grad():
            logits_all = forward_fn(models, X_ref)
            pred_all = (logits_all > 0).float().detach().cpu().numpy().flatten()

    # Choose slice radiation levels
    rad_levels = np.quantile(rad_all, [0.1, 0.5, 0.9])

    # Grid bounds based on real data
    cur_min, cur_max = np.percentile(cur_all, [1, 99])
    pre_min, pre_max = np.percentile(pre_all, [1, 99])

    cur_grid = np.linspace(cur_min, cur_max, n_grid)
    pre_grid = np.linspace(pre_min, pre_max, n_grid)

    rad_range = np.max(rad_all) - np.min(rad_all)
    band = band_frac * rad_range

    def plot_one_overlay(rad_fixed, label_mode):
        # Build synthetic constant sequences
        seqs = []
        for c in cur_grid:
            for p in pre_grid:
                seqs.append(make_constant_sequence(c, p, rad_fixed, scaler, seq_len))

        X_syn = torch.FloatTensor(np.array(seqs))

        for layer in models.values():
            layer.eval()

        with torch.no_grad():
            logits = forward_fn(models, X_syn).cpu().numpy().reshape(n_grid, n_grid)

        # Pick points near this radiation slice
        slice_mask = (rad_all >= (rad_fixed - band)) & (rad_all <= (rad_fixed + band))

        cur_slice = cur_all[slice_mask]
        pre_slice = pre_all[slice_mask]

        if label_mode == "real":
            lab_slice = y_all[slice_mask]
            subtitle = "Scatter = REAL labels"
        else:
            lab_slice = pred_all[slice_mask]
            subtitle = "Scatter = PREDICTED labels"

        # Plot
        plt.figure(figsize=(8, 6))

        # Decision boundary: logit=0
        CS = plt.contour(cur_grid, pre_grid, logits.T, levels=[0.0])
        plt.clabel(CS, inline=True, fontsize=8)

        # Scatter overlay (0 vs 1)
        if len(cur_slice) > 0:
            idx0 = (lab_slice == 0)
            idx1 = (lab_slice == 1)

            plt.scatter(cur_slice[idx0], pre_slice[idx0], alpha=0.25, label="0")
            plt.scatter(cur_slice[idx1], pre_slice[idx1], alpha=0.45, label="1")
        else:
            plt.text(0.5, 0.5, "No points in radiation band",
                     transform=plt.gca().transAxes, ha='center', va='center')

        plt.title(
            f"Approx Decision Boundary Slice\n"
            f"Radiation ≈ {rad_fixed:.3f} (band ± {band:.3f})\n{subtitle}"
        )
        plt.xlabel("Current (real units)")
        plt.ylabel("Pressure (real units)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Render according to overlay option
    for rad_fixed in rad_levels:
        if overlay == "real":
            plot_one_overlay(rad_fixed, "real")
        elif overlay == "pred":
            plot_one_overlay(rad_fixed, "pred")
        else:  # both
            plot_one_overlay(rad_fixed, "real")
            plot_one_overlay(rad_fixed, "pred")


#####################################################################################
def plot_training_curves(history, output_folder=None):
    """
    Plots the Loss and Accuracy curves after training.
    Optionally saves to output_folder if provided.
    """
    epochs = range(1, len(history['val_loss']) + 1)
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val', linestyle='--')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], color='green', label='Val Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig, output_folder, "training_curves")
    plt.show()


###########################################################################

#### Ignore from Line 1150 until Line 1330.
def build_sequences_with_indices_for_plot(df, scaler, sequence_length, missing_value=-100.0):
    """
    Lightweight sequence builder for plotting.
    Matches your feature construction (4 channels, no masks):
    [Current, Pressure, Radiation, Prev_A1]
    Returns:
      X_seq, y_seq, target_indices, df_clean
    """
    df = df.copy()

    # Map target -1 -> 0
    if 'VoltageChange' in df.columns:
        df['VoltageChange'] = df['VoltageChange'].apply(lambda x: 0 if x == -1 else x)
    else:
        df['VoltageChange'] = np.nan

    feature_cols = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]

    # Ensure cols exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Build inputs (no masks)
    input_data = []
    for col in feature_cols:
        val = pd.to_numeric(df[col], errors='coerce').astype(float)
        input_data.append(val.values)

    # Prev A1 history
    a1_history = pd.to_numeric(df['VoltageChange'], errors='coerce').fillna(0).values
    input_data.append(a1_history)

    X_raw = np.stack(input_data, axis=1)  # (T, 4)

    # Scale ONLY sensor columns and handle NaN
    nan_mask = np.isnan(X_raw[:, [0, 1, 2]])
    X_sensor_temp = np.nan_to_num(X_raw[:, [0, 1, 2]], nan=0.0)
    X_raw[:, [0, 1, 2]] = scaler.transform(X_sensor_temp)
    
    # Replace NaN positions with missing_value
    for i in range(3):
        X_raw[nan_mask[:, i], i] = missing_value

    y_raw = pd.to_numeric(df['VoltageChange'], errors='coerce').values

    if len(X_raw) <= sequence_length:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1)), np.array([], dtype=int), df

    X_seq, y_seq, idxs = [], [], []

    for i in range(len(X_raw) - sequence_length):
        idx = i + sequence_length
        target_val = y_raw[idx]

        # Only include valid targets
        if not np.isnan(target_val):
            X_seq.append(X_raw[i:i + sequence_length])
            y_seq.append(target_val)
            idxs.append(idx)

    if len(X_seq) == 0:
        return np.empty((0, sequence_length, 4)), np.empty((0, 1)), np.array([], dtype=int), df

    return np.array(X_seq), np.array(y_seq).reshape(-1, 1), np.array(idxs, dtype=int), df

def plot_decision_boundary_slices_shaded(models, forward_fn, data, scaler, seq_len,
                                         n_grid=60, band_frac=0.10, overlay="real",
                                         use_set="trainval"):
    """
    Decision boundary slice with shaded class regions and scatter overlay.

    Shading meaning:
      - logit < 0  -> class 0 region
      - logit > 0  -> class 1 region

    overlay:
      "real" | "pred"
    use_set:
      "train" | "val" | "test" | "trainval"
    """

    # -------- pick dataset for scatter ----------
    if use_set == "train":
        X_ref, y_ref = data['X_train'], data['y_train']
    elif use_set == "val":
        X_ref, y_ref = data['X_val'], data['y_val']
    elif use_set == "test":
        X_ref, y_ref = data['X_test'], data['y_test']
    else:
        X_ref = torch.cat([data['X_train'], data['X_val']], dim=0)
        y_ref = torch.cat([data['y_train'], data['y_val']], dim=0)

    if len(X_ref) == 0:
        print("Selected set is empty for boundary scatter.")
        return

    # Extract last-step physical values
    cur_all, pre_all, rad_all = extract_last_step_three_features(X_ref, scaler)
    y_all = y_ref.detach().cpu().numpy().flatten()

    # Predictions if overlay="pred"
    pred_all = None
    if overlay == "pred":
        for layer in models.values():
            layer.eval()
        with torch.no_grad():
            pred_all = (forward_fn(models, X_ref) > 0).float().cpu().numpy().flatten()

    # Slice radiation levels
    rad_levels = np.quantile(rad_all, [0.1, 0.5, 0.9])

    # Grid bounds
    cur_min, cur_max = np.percentile(cur_all, [1, 99])
    pre_min, pre_max = np.percentile(pre_all, [1, 99])

    cur_grid = np.linspace(cur_min, cur_max, n_grid)
    pre_grid = np.linspace(pre_min, pre_max, n_grid)

    rad_range = np.max(rad_all) - np.min(rad_all)
    band = band_frac * rad_range

    def get_labels(mask):
        if overlay == "real":
            return y_all[mask]
        else:
            return pred_all[mask]

    # Build shaded plots
    for rad_fixed in rad_levels:
        # --- synthetic grid for boundary ---
        seqs = []
        for c in cur_grid:
            for p in pre_grid:
                seqs.append(make_constant_sequence(c, p, rad_fixed, scaler, seq_len))
        X_syn = torch.FloatTensor(np.array(seqs))

        for layer in models.values():
            layer.eval()

        with torch.no_grad():
            logits = forward_fn(models, X_syn).cpu().numpy().reshape(n_grid, n_grid)

        # --- mask real points near this rad slice ---
        slice_mask = (rad_all >= (rad_fixed - band)) & (rad_all <= (rad_fixed + band))
        cur_slice = cur_all[slice_mask]
        pre_slice = pre_all[slice_mask]
        lab_slice = get_labels(slice_mask)

        plt.figure(figsize=(8, 6))

        # ---- SHADING ----
        # Two regions: logits < 0 (class 0), logits > 0 (class 1)
        plt.contourf(cur_grid, pre_grid, logits.T,
                     levels=[-1e9, 0, 1e9], 
                     colors=['green', 'grey'], alpha=0.18)

        # ---- BOUNDARY LINE ----
        CS = plt.contour(cur_grid, pre_grid, logits.T, levels=[0.0])
        plt.clabel(CS, inline=True, fontsize=8)

        # ---- SCATTER ----
        if len(cur_slice) > 0:
            idx0 = (lab_slice == 0)
            idx1 = (lab_slice == 1)
            plt.scatter(cur_slice[idx0], pre_slice[idx0], alpha=0.25, label=f"{overlay.upper()} 0")
            plt.scatter(cur_slice[idx1], pre_slice[idx1], alpha=0.45, label=f"{overlay.upper()} 1")
        else:
            plt.text(0.5, 0.5, "No points in radiation band",
                     transform=plt.gca().transAxes, ha='center', va='center')

        plt.title(
            f"Shaded Decision Regions + Scatter ({use_set.upper()})\n"
            f"Radiation ≈ {rad_fixed:.3f} (band ± {band:.3f})"
        )
        plt.xlabel("Current (real units)")
        plt.ylabel("Pressure (real units)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

##################################################################################################

def plot_test_file_trends(filepath, models, forward_fn, scaler, sequence_length, output_folder=None):
    """
    6-row subplot:
    1) Real VoltageChange (0/1)
    2) Pred VoltageChange (0/1)
    3) Real Voltage (continuous)
    4) Current vs time
    5) Pressure vs time
    6) Radiation vs time
    
    Optionally saves to output_folder if provided.
    """

    print(f"\n--- Trend Plot: {os.path.basename(filepath)} ---")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print("Could not read file:", e)
        return

    # Detect time column
    time_col = next((col for col in df.columns if col.lower() == 'time'), None)

    # Build time axis
    if time_col:
        t_raw = df[time_col].copy()

        # Try numeric first, else datetime
        t_num = pd.to_numeric(t_raw, errors='coerce')
        if t_num.notna().sum() > 0:
            t = t_num.fillna(method='ffill').fillna(method='bfill')
        else:
            t_dt = pd.to_datetime(t_raw, errors='coerce')
            if t_dt.notna().sum() > 0:
                t = t_dt.fillna(method='ffill').fillna(method='bfill')
            else:
                t = pd.Series(np.arange(len(df)))
    else:
        t = pd.Series(np.arange(len(df)))

    # Build sequences + aligned indices
    X_seq, y_seq, idxs, df_clean = build_sequences_with_indices_for_plot(df, scaler, sequence_length)

    if len(X_seq) == 0:
        print("File too short or no valid targets for plotting.")
        return

    X_tensor = torch.FloatTensor(X_seq)

    for layer in models.values():
        layer.eval()

    with torch.no_grad():
        logits = forward_fn(models, X_tensor)
        preds = (logits > 0).float().cpu().numpy().flatten()

    y_true = y_seq.flatten()

    # Extract raw sensor series for plotting
    cur_series = pd.to_numeric(
        df_clean.get('GunCurrent.Avg', pd.Series(np.nan, index=df_clean.index)),
        errors='coerce'
    )
    pre_series = pd.to_numeric(
        df_clean.get('peg-BL-cc:pressureM', pd.Series(np.nan, index=df_clean.index)),
        errors='coerce'
    )
    rad_series = pd.to_numeric(
        df_clean.get('RadiationTotal', pd.Series(np.nan, index=df_clean.index)),
        errors='coerce'
    )

     

    # Voltage series (try Glassman voltage, fall back to hvps.lerec if needed)
    if 'glassmanDataXfer:hvPsVoltageMeasM' in df_clean.columns:
        hv_series = pd.to_numeric(
            df_clean['glassmanDataXfer:hvPsVoltageMeasM'], errors='coerce'
        )
    elif 'hvps.lerec:voltageM' in df_clean.columns:
        hv_series = pd.to_numeric(
            df_clean['hvps.lerec:voltageM'], errors='coerce'
        )
    else:
        hv_series = pd.Series(np.nan, index=df_clean.index)


    # --- Plot ---
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(
        f"Trends + Real & Pred VoltageChange (Separate Panels)\n{os.path.basename(filepath)}",
        fontsize=14
    )

       # 1) REAL VoltageChange
    axes[0].step(t.iloc[idxs], y_true, label="Real VoltageChange", where='post')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_ylabel("Real 0/1")
    axes[0].grid(True, alpha=0.3)

    # 2) PRED VoltageChange
    axes[1].step(t.iloc[idxs], preds, label="Pred VoltageChange", where='post', color='orange')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("Pred 0/1")
    axes[1].grid(True, alpha=0.3)

    # 3) Real Voltage (continuous)
    axes[2].plot(t, hv_series, linewidth=1.2)
    axes[2].set_ylabel("Voltage")
    axes[2].grid(True, alpha=0.3)

    # 4) Current
    axes[3].plot(t, cur_series, linewidth=1.0)
    axes[3].set_ylabel("Current")
    axes[3].grid(True, alpha=0.3)

    # 5) Pressure
    axes[4].plot(t, pre_series, linewidth=1.0)
    axes[4].set_ylabel("Pressure")
    axes[4].grid(True, alpha=0.3)

    # 6) Radiation
    axes[5].plot(t, rad_series, linewidth=1.0)
    axes[5].set_ylabel("Radiation")
    axes[5].set_xlabel("Time")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_folder:
        # Create filename from the test file name
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_figure(fig, output_folder, f"trend_{base_name}")
    plt.show()



# ==========================================
# 6. Simulation (UPDATED: Real vs Sim Plot)
# ==========================================

def test_single_file_simulation(filepath, models, forward_fn, scaler, sequence_length, step_size, initial_A=0, max_limit_A=None, output_folder=None):
    """
    Runs simulation on a single test file and plots Real vs Simulated voltage.
    Optionally saves the figure to output_folder if provided.
    """
    print(f"\n--- Processing Simulation: {filepath} ---")
    
    try:
        df = pd.read_csv(filepath)
        
    except FileNotFoundError:
        print("File not found.")
        return

    # Process Input
   # print('raw rows',df.shape)
    X_seq, y_seq = process_single_df_to_sequences(df, scaler, sequence_length)
   # print('processed maybe',X_seq.shape)
   # print('processed',y_seq.shape)
    if len(X_seq) == 0:
        print("File too short or empty after processing.")
        return

    X_tensor = torch.FloatTensor(X_seq)
    
    # # Eval Mode
    # models['conv1'].eval(); models['bn1'].eval()
    # models['conv2'].eval(); models['bn2'].eval()
    
    with torch.no_grad():
        logits = forward_fn(models, X_tensor)
        predictions = (logits > 0).float().cpu().numpy().flatten()

    simulated_A = []
    current_A = initial_A 
    decisions = []
    
    print(f"Sim Config -> Step: {step_size:.2f} | Limit: {max_limit_A}")

    for raw_pred in predictions:
        decision = int(raw_pred)
        
        # # Interlock
        # if max_limit_A is not None:
        #     if current_A >= max_limit_A and decision == 1:
        #         decision = 0 
        
        decisions.append(decision)
        
        if decision == 1:
            current_A += step_size 
            
        simulated_A.append(current_A)

    # # --- UPDATED PLOTTING LOGIC: REAL vs SIMULATED ---
    # plt.figure(figsize=(12, 6))
    
    # # 1. Plot Real Voltage (if column exists)
    # # We must slice it to match predictions (starting from sequence_length)
    # if 'glassmanDataXfer:hvPsVoltageMeasM' in df.columns:
    #     real_A = df['glassmanDataXfer:hvPsVoltageMeasM'].values
    #     # We need the segment corresponding to the predictions
    #     # The predictions correspond to indices [sequence_length : end]
    #     real_A_segment = real_A[sequence_length : sequence_length + len(simulated_A)]
        
    #     plt.plot(real_A_segment, label='Real Voltage (Data)', color='green', linestyle='--', linewidth=2, alpha=0.6)
    
    # # 2. Plot Simulated Voltage
    # plt.plot(simulated_A, label='Simulated Voltage (Model)', color='blue', linewidth=2)
    
    # if max_limit_A is not None:
    #     plt.axhline(y=max_limit_A, color='red', linestyle=':', label=f'Max Limit ({max_limit_A:.1f})')
        
    # plt.title(f"Simulation vs Real: {os.path.basename(filepath)}\nStep: {step_size:.2f}")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Voltage")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()



    # --- UPDATED PLOTTING LOGIC: REAL vs SIMULATED (NORMALIZED 0–1) ---
    sim_arr = np.array(simulated_A, dtype=float)

    fig = plt.figure(figsize=(12, 6))

    real_norm = None

    # 1. Real Voltage (if column exists) → normalize 0–1
    if 'glassmanDataXfer:hvPsVoltageMeasM' in df.columns:
        real_A = pd.to_numeric(
            df['glassmanDataXfer:hvPsVoltageMeasM'], errors='coerce'
        ).values

        real_segment = real_A[sequence_length: sequence_length + len(sim_arr)]

        if len(real_segment) == len(sim_arr):
            r_min = np.nanmin(real_segment)
            r_max = np.nanmax(real_segment)
            if np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
                real_norm = (real_segment - r_min) / (r_max - r_min)
            else:
                real_norm = np.zeros_like(real_segment)

            plt.plot(
                real_norm,
                label='Real Voltage (normalized)',
                linestyle='--',
                linewidth=2,
                alpha=0.7
            )

    # 2. Simulated Voltage → normalize 0–1 (own min/max)
    s_min = np.min(sim_arr)
    s_max = np.max(sim_arr)
    if s_max > s_min:
        sim_norm = (sim_arr - s_min) / (s_max - s_min)
    else:
        sim_norm = np.zeros_like(sim_arr)

    plt.plot(
        sim_norm,
        label='Simulated Voltage (normalized)',
        linewidth=2
    )

    plt.title(f"Simulation vs Real (Normalized): {os.path.basename(filepath)}\nStep: {step_size:.2f}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Voltage (0–1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_folder:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_figure(fig, output_folder, f"simulation_{base_name}")
    plt.show()


# ==========================================
# 6.5 Robustness Validation Suite
# ==========================================

def create_ablations(X_data, y_data, missing_value=-100.0, random_seed=42):
    """
    Creates deterministic ablation variants of input data (validation or test).
    Also returns a missing_mask indicating which samples already have pre-existing
    missing sensor data, so downstream evaluation can filter appropriately.
    
    Args:
        X_data: Input tensor [N, seq_len, 4] - channels: [Current, Pressure, Radiation, Prev_A1]
        y_data: Labels tensor [N, 1]
        missing_value: Value to use for missing sensor data (default -100)
        random_seed: Seed for reproducible shuffling (uses LOCAL generator, doesn't affect global state)
    
    Returns:
        ablations: Dict with keys:
            Sensor ablations (channels 0, 1, 2):
            - 'clean': Original data
            - '{sensor}_full': Sensor channel fully missing (-100)
            - '{sensor}_block': Sensor channel 50% block missing (middle)
            
            Prev_A1 ablations (channel 3 - binary 0/1):
            - 'prev_a1_shuffle': Shuffled across samples
            - 'prev_a1_const_0': All zeros
            - 'prev_a1_const_1': All ones
            - 'prev_a1_flip': Flipped values 0->1, 1->0
        
        missing_mask: Boolean tensor [N, 3] where mask[i, ch] = True if sample i
                      already has missing data in sensor channel ch (0=Current, 1=Pressure, 2=Radiation).
    """
    # Use LOCAL generator to avoid affecting global torch random state
    # (doesn't interfere with model weight initialization or training)
    generator = torch.Generator().manual_seed(random_seed)
    
    seq_len = X_data.shape[1]
    n_samples = X_data.shape[0]
    block_start = seq_len // 4
    block_end = 3 * seq_len // 4  # 50% middle block
    
    # Detect pre-existing missing data BEFORE creating ablation variants
    missing_mask = get_missing_mask(X_data, missing_value=missing_value)
    
    ablations = {'clean': (X_data.clone(), y_data.clone())}
    
    # --- SENSOR ABLATIONS (channels 0, 1, 2) ---
    sensor_names = ['current', 'pressure', 'radiation']
    
    for ch_idx, name in enumerate(sensor_names):
        # Full window ablation
        X_full = X_data.clone()
        X_full[:, :, ch_idx] = missing_value
        ablations[f'{name}_full'] = (X_full, y_data.clone())
        
        # Block ablation (middle 50%)
        X_block = X_data.clone()
        X_block[:, block_start:block_end, ch_idx] = missing_value
        ablations[f'{name}_block'] = (X_block, y_data.clone())
    
    # --- PREV_A1 ABLATIONS (channel 3 - binary 0/1) ---
    prev_a1_idx = 3
    
    # 1. Shuffle across samples (preserves distribution, breaks correlation)
    X_shuffle = X_data.clone()
    perm = torch.randperm(n_samples, generator=generator)  # Uses local generator
    X_shuffle[:, :, prev_a1_idx] = X_data[perm, :, prev_a1_idx]
    ablations['prev_a1_shuffle'] = (X_shuffle, y_data.clone())
    
    # 2. Constant 0 (what if previous state was always "stable"?)
    X_const_0 = X_data.clone()
    X_const_0[:, :, prev_a1_idx] = 0.0
    ablations['prev_a1_const_0'] = (X_const_0, y_data.clone())
    
    # 3. Constant 1 (what if previous state was always "increase"?)
    X_const_1 = X_data.clone()
    X_const_1[:, :, prev_a1_idx] = 1.0
    ablations['prev_a1_const_1'] = (X_const_1, y_data.clone())
    
    # 4. Flip values 0→1, 1→0 (tests if model learned correct polarity)
    X_flip = X_data.clone()
    X_flip[:, :, prev_a1_idx] = 1.0 - X_data[:, :, prev_a1_idx]
    ablations['prev_a1_flip'] = (X_flip, y_data.clone())
    
    return ablations, missing_mask


def create_validation_ablations(X_val, y_val, missing_value=-100.0):
    """
    Creates deterministic ablation variants of validation data.
    Wrapper around create_ablations for backward compatibility.
    
    Returns:
        (ablations, missing_mask)
    """
    return create_ablations(X_val, y_val, missing_value=missing_value, random_seed=42)


def create_test_ablations(X_test, y_test, missing_value=-100.0):
    """
    Creates deterministic ablation variants of test data.
    Uses different seed than validation for independence.
    
    Returns:
        (ablations, missing_mask)
    """
    return create_ablations(X_test, y_test, missing_value=missing_value, random_seed=97)


def evaluate_robustness(models, forward_fn, ablations, missing_mask=None, 
                        filter_missing=True, device='cpu'):
    """
    Evaluates model on all ablation variants.
    
    Two modes controlled by filter_missing:
    
    filter_missing=True (use for VALIDATION data):
        For sensor ablation variants, excludes samples where the target channel
        is already missing OR any other sensor channel is already missing.
        Reports clean-subset metrics as primary, multi-missing separately.
    
    filter_missing=False (use for TEST data with natural missingness):
        Evaluates ALL samples. Still reports missingness stats for information.
    
    Prev_A1 ablations and 'clean' always use ALL samples regardless of filter_missing.
    
    Args:
        models: Dict of model layers
        forward_fn: Forward function
        ablations: Dict of {name: (X, y)} ablation variants
        missing_mask: Optional [N, 3] bool tensor from get_missing_mask / create_ablations.
                      If None, all samples are used (backward compatible).
        filter_missing: If True, filter sensor ablations to clean subset.
                        If False, use all samples but report stats.
        device: Device to evaluate on
    
    Returns:
        dict of {ablation_name: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'n_clean': int,
            'n_multi_missing': int,
            'multi_missing_acc': float or None
        }}
    """
    sensor_ablation_map = {
        'current': 0,
        'pressure': 1,
        'radiation': 2,
    }
    
    results = {}
    
    for layer in models.values():
        layer.eval()
    
    for name, (X, y) in ablations.items():
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            logits = forward_fn(models, X)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            
            preds_np = preds.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()
        
        # Determine if this is a sensor ablation
        ablated_ch = None
        if missing_mask is not None:
            for prefix, ch_idx in sensor_ablation_map.items():
                if name.startswith(prefix):
                    ablated_ch = ch_idx
                    break
        
        if ablated_ch is not None and missing_mask is not None:
            # Compute clean mask: exclude target-already-missing AND other-missing
            target_missing = missing_mask[:, ablated_ch].cpu().numpy()
            other_chs = [j for j in range(3) if j != ablated_ch]
            other_missing = missing_mask[:, other_chs].any(dim=1).cpu().numpy()
            any_issue = target_missing | other_missing
            clean_idx = ~any_issue
            
            n_clean = int(clean_idx.sum())
            n_excluded = int(any_issue.sum())
            
            if filter_missing and n_clean > 0:
                # VALIDATION MODE: use clean subset only
                acc = (preds_np[clean_idx] == y_np[clean_idx]).mean()
                prec = precision_score(y_np[clean_idx], preds_np[clean_idx], zero_division=0)
                rec = recall_score(y_np[clean_idx], preds_np[clean_idx], zero_division=0)
                
                if n_excluded > 0:
                    multi_acc = (preds_np[any_issue] == y_np[any_issue]).mean()
                else:
                    multi_acc = None
            elif filter_missing and n_clean == 0:
                acc, prec, rec = 0.0, 0.0, 0.0
                multi_acc = (preds_np == y_np).mean()
            else:
                # TEST MODE: use ALL samples, just report stats
                acc = (preds_np == y_np).mean()
                prec = precision_score(y_np, preds_np, zero_division=0)
                rec = recall_score(y_np, preds_np, zero_division=0)
                
                if n_excluded > 0:
                    multi_acc = (preds_np[any_issue] == y_np[any_issue]).mean()
                else:
                    multi_acc = None
            
            results[name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'n_clean': n_clean, 'n_multi_missing': n_excluded,
                'multi_missing_acc': multi_acc
            }
        else:
            # 'clean' variant, prev_a1 ablations, or no mask: use ALL samples
            n_total = len(y_np)
            acc = (preds_np == y_np).mean()
            prec = precision_score(y_np, preds_np, zero_division=0)
            rec = recall_score(y_np, preds_np, zero_division=0)
            
            results[name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'n_clean': n_total, 'n_multi_missing': 0,
                'multi_missing_acc': None
            }
    
    return results


def _write_robustness_section(f, robustness_results, section_title):
    """Helper to write robustness metrics for a single dataset (val or test).
    
    Now includes missing-aware columns: N_clean, N_multi, Multi_Acc for sensor ablations.
    """
    f.write("=" * 120 + "\n")
    f.write(f"{section_title}\n")
    f.write("=" * 120 + "\n\n")
    
    f.write(f"{'Ablation Type':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Acc Delta':>12} {'N_clean':>8} {'N_multi':>8} {'Multi_Acc':>10}\n")
    f.write("-" * 100 + "\n")
    
    clean_acc = robustness_results.get('clean', {}).get('accuracy', 0)
    
    for name, metrics in robustness_results.items():
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        delta = acc - clean_acc if name != 'clean' else 0.0
        delta_str = f"{delta:+.4f}" if name != 'clean' else "---"
        
        n_clean = metrics.get('n_clean', '---')
        n_multi = metrics.get('n_multi_missing', 0)
        multi_acc = metrics.get('multi_missing_acc', None)
        
        n_clean_str = f"{n_clean:>8}" if isinstance(n_clean, int) else f"{'---':>8}"
        n_multi_str = f"{n_multi:>8}" if n_multi > 0 else f"{'---':>8}"
        multi_acc_str = f"{multi_acc:>10.4f}" if multi_acc is not None else f"{'N/A':>10}"
        
        f.write(f"{name:<20} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {delta_str:>12} {n_clean_str} {n_multi_str} {multi_acc_str}\n")
    
    f.write("\n" + "-" * 100 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 100 + "\n")
    
    clean = robustness_results.get('clean', {})
    f.write(f"Clean Baseline:  Acc={clean.get('accuracy', 0):.4f}  "
            f"Prec={clean.get('precision', 0):.4f}  Rec={clean.get('recall', 0):.4f}\n\n")
    
    # Sensor full-missing averages (using clean-subset metrics)
    full_metrics = {k: v for k, v in robustness_results.items() if '_full' in k}
    if full_metrics:
        avg_acc = np.mean([v['accuracy'] for v in full_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in full_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in full_metrics.values()])
        f.write(f"Sensor Full-Missing Avg (clean subset):  Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}\n")
    
    # Sensor block-missing averages
    block_metrics = {k: v for k, v in robustness_results.items() if '_block' in k}
    if block_metrics:
        avg_acc = np.mean([v['accuracy'] for v in block_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in block_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in block_metrics.values()])
        f.write(f"Sensor Block-Missing Avg (clean subset): Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}\n")
    
    # Prev_A1 ablation averages (all samples, not filtered)
    prev_a1_metrics = {k: v for k, v in robustness_results.items() if 'prev_a1' in k}
    if prev_a1_metrics:
        avg_acc = np.mean([v['accuracy'] for v in prev_a1_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in prev_a1_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in prev_a1_metrics.values()])
        f.write(f"Prev_A1 Ablation Avg (all samples):      Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}\n")
    
    f.write("\nNote: Sensor ablation metrics computed on CLEAN subset only (no other sensor pre-missing).\n")
    f.write("      Prev_A1 ablations use ALL samples. Multi_Acc shows accuracy on excluded samples.\n")
    f.write("\n")


def plot_robustness_ablation(robustness_results, title_suffix="", output_folder=None, filename=None):
    """
    Bar chart of accuracy drop per ablation variant from the robustness suite.
    
    Shows drop = clean_accuracy - ablated_accuracy for each variant.
    Annotates sensor ablation bars with clean sample count if available.
    
    Args:
        robustness_results: Dict from evaluate_robustness()
        title_suffix: String appended to chart title (e.g., "Validation" or "Test")
        output_folder: Optional folder to save figure
        filename: Optional filename for saved figure
    """
    clean_acc = robustness_results.get('clean', {}).get('accuracy', 0)
    
    # Exclude 'clean' from the bar chart (it's the baseline)
    ablation_names = [n for n in robustness_results if n != 'clean']
    drops = [clean_acc - robustness_results[n]['accuracy'] for n in ablation_names]
    n_cleans = [robustness_results[n].get('n_clean', None) for n in ablation_names]
    n_excluded = [robustness_results[n].get('n_multi_missing', 0) for n in ablation_names]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(ablation_names))
    colors = ['green' if d > 0.05 else 'orange' if d > 0.01 else 'gray' for d in drops]
    
    bars = ax.bar(x, drops, color=colors, alpha=0.8, edgecolor='black')
    
    for i, (bar, nc, ne) in enumerate(zip(bars, n_cleans, n_excluded)):
        if nc is not None:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'n={nc}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if ne and ne > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.012,
                    f'(excl {ne})', ha='center', va='bottom', fontsize=7, color='red')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High importance (5%)')
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium importance (1%)')
    
    ax.set_xlabel('Ablation Variant', fontsize=12)
    ax.set_ylabel('Accuracy Drop (Clean - Ablated)', fontsize=12)
    ax.set_title(f'Robustness Ablation: Accuracy Drop per Variant ({title_suffix})\n'
                 f'Baseline (clean) Acc = {clean_acc:.4f}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder and filename:
        save_figure(fig, output_folder, filename)
    plt.show()


def save_robustness_report(robustness_results, validation_checks, output_folder, 
                          epoch=None, test_robustness=None):
    """
    Saves formatted robustness metrics AND validation check results to a text file.
    
    Args:
        robustness_results: Validation set ablation results
        validation_checks: Validation check results dict
        output_folder: Output directory
        epoch: Optional epoch number for filename
        test_robustness: Optional test set ablation results
    """
    filename = f"robustness_report_epoch{epoch}.txt" if epoch else "robustness_report_final.txt"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, 'w') as f:
        # ==================== VALIDATION SET ROBUSTNESS ====================
        _write_robustness_section(f, robustness_results, "VALIDATION SET ABLATION RESULTS")
        
        # ==================== TEST SET ROBUSTNESS ====================
        if test_robustness:
            f.write("\n")
            _write_robustness_section(f, test_robustness, "TEST SET ABLATION RESULTS")
        
        # ==================== VALIDATION CHECK RESULTS ====================
        if validation_checks:
            f.write("\n" + "=" * 90 + "\n")
            f.write("VALIDATION CHECK RESULTS\n")
            f.write("=" * 90 + "\n\n")
            
            total_checks = 0
            passed_checks = 0
            
            for category, checks in validation_checks.items():
                f.write(f"\n{category.upper().replace('_', ' ')}\n")
                f.write("-" * 40 + "\n")
                
                for check_name, passed in checks.items():
                    status = "PASS" if passed else "FAIL"
                    f.write(f"  {check_name:<35} [{status}]\n")
                    total_checks += 1
                    if passed:
                        passed_checks += 1
            
            f.write("\n" + "-" * 40 + "\n")
            if total_checks > 0:
                f.write(f"TOTAL: {passed_checks}/{total_checks} checks passed ")
                f.write(f"({100*passed_checks/total_checks:.1f}%)\n")
        
        f.write("=" * 90 + "\n")
    
    print(f"Saved: {filepath}")
    return filepath


# ==========================================
# 6.6 Validation Test Cases
# ==========================================

def test_data_split(X_train, X_val):
    """Verify shuffled 85/15 train/val split is correct."""
    results = {}
    
    n_train = len(X_train)
    n_val = len(X_val)
    total = n_train + n_val
    
    # Check split ratio is approximately 85/15
    train_ratio = n_train / total if total > 0 else 0
    results['train_ratio_approx_85pct'] = 0.80 < train_ratio < 0.90
    results['train_exists'] = n_train > 0
    results['val_exists'] = n_val > 0
    
    return results


def test_channel_count(X_train, models, model_type='cnn'):
    """Verify 4-channel architecture."""
    results = {}
    
    results['data_has_4_channels'] = X_train.shape[2] == 4
    
    if model_type == 'cnn':
        results['conv1_expects_4_channels'] = models['conv1'].in_channels == 4
    else:
        results['lstm_expects_4_channels'] = models['lstm'].input_size == 4
    
    return results


def test_dropout_function(X_sample, n_iters=500, prob_full=0.2, prob_block=0.3, missing_value=-100.0):
    """Verify dropout function behavior is correct."""
    results = {}
    
    full_count, block_count, clean_count = 0, 0, 0
    single_channel_only = True
    blocks_contiguous = True
    
    for _ in range(n_iters):
        x_aug = apply_sensor_dropout(
            X_sample.clone(),
            prob_full_missing=prob_full,
            prob_block_missing=prob_block,
            missing_value=missing_value
        )
        
        # Check first sample in batch
        x = x_aug[0]
        missing_mask = (x[:, :3] == missing_value)
        
        channels_affected = missing_mask.any(dim=0).sum().item()
        if channels_affected > 1:
            single_channel_only = False
        
        if channels_affected == 1:
            ch_idx = missing_mask.any(dim=0).nonzero().squeeze().item()
            missing_count = missing_mask[:, ch_idx].sum().item()
            
            if missing_count == x.shape[0]:
                full_count += 1
            else:
                block_count += 1
                indices = missing_mask[:, ch_idx].nonzero().squeeze()
                if indices.dim() > 0 and len(indices) > 1:
                    if (indices[-1] - indices[0] + 1) != len(indices):
                        blocks_contiguous = False
        else:
            clean_count += 1
    
    results['only_one_channel_dropped'] = single_channel_only
    results['blocks_are_contiguous'] = blocks_contiguous
    results['full_prob_approx_20pct'] = 0.10 < full_count/n_iters < 0.30
    results['block_prob_approx_30pct'] = 0.20 < block_count/n_iters < 0.40
    results['clean_prob_approx_50pct'] = 0.40 < clean_count/n_iters < 0.60
    
    return results


def test_ablation_suite(ablations, X_data, missing_value=-100.0):
    """Verify ablation variants are correct."""
    results = {}
    seq_len = X_data.shape[1]
    prev_a1_idx = 3
    
    # Check clean data matches original
    results['clean_matches_original'] = torch.allclose(ablations['clean'][0], X_data)
    
    # Check sensor ablations (channels 0, 1, 2)
    for ch_idx, name in enumerate(['current', 'pressure', 'radiation']):
        X_abl = ablations[f'{name}_full'][0]
        results[f'{name}_full_all_missing'] = (X_abl[:, :, ch_idx] == missing_value).all().item()
    
    start, end = seq_len // 4, 3 * seq_len // 4
    for ch_idx, name in enumerate(['current', 'pressure', 'radiation']):
        X_abl = ablations[f'{name}_block'][0]
        results[f'{name}_block_middle_missing'] = (X_abl[:, start:end, ch_idx] == missing_value).all().item()
    
    # Check prev_a1 ablations (channel 3)
    # Shuffle: should have same values but different order
    X_shuffle = ablations['prev_a1_shuffle'][0]
    orig_sum = X_data[:, :, prev_a1_idx].sum().item()
    shuffle_sum = X_shuffle[:, :, prev_a1_idx].sum().item()
    results['prev_a1_shuffle_preserves_sum'] = abs(orig_sum - shuffle_sum) < 0.01
    
    # Const 0: all zeros
    X_const_0 = ablations['prev_a1_const_0'][0]
    results['prev_a1_const_0_all_zeros'] = (X_const_0[:, :, prev_a1_idx] == 0).all().item()
    
    # Const 1: all ones
    X_const_1 = ablations['prev_a1_const_1'][0]
    results['prev_a1_const_1_all_ones'] = (X_const_1[:, :, prev_a1_idx] == 1).all().item()
    
    # Flip: should be 1 - original
    X_flip = ablations['prev_a1_flip'][0]
    expected_flip = 1.0 - X_data[:, :, prev_a1_idx]
    results['prev_a1_flip_correct'] = torch.allclose(X_flip[:, :, prev_a1_idx], expected_flip)
    
    return results


def test_model_forward(models, forward_fn, X_sample, missing_value=-100.0):
    """Verify model handles -100 values without errors."""
    results = {}
    
    try:
        logits = forward_fn(models, X_sample)
        results['clean_forward_ok'] = not (torch.isnan(logits).any() or torch.isinf(logits).any())
    except Exception:
        results['clean_forward_ok'] = False
    
    try:
        X_test = X_sample.clone()
        X_test[:, :, 0] = missing_value
        logits = forward_fn(models, X_test)
        results['ablated_forward_ok'] = not (torch.isnan(logits).any() or torch.isinf(logits).any())
    except Exception:
        results['ablated_forward_ok'] = False
    
    return results


def test_robustness_sanity(robustness_results):
    """Verify robustness metrics are reasonable."""
    results = {}
    
    clean_acc = robustness_results['clean']['accuracy']
    
    all_valid = True
    for name, metrics in robustness_results.items():
        if not (0 <= metrics['accuracy'] <= 1 and 
                0 <= metrics['precision'] <= 1 and 
                0 <= metrics['recall'] <= 1):
            all_valid = False
    results['all_metrics_valid_range'] = all_valid
    
    max_ablated = max(m['accuracy'] for n, m in robustness_results.items() if n != 'clean')
    results['clean_is_best_or_close'] = clean_acc >= max_ablated - 0.10
    
    return results


def run_validation_checks(X_train, X_val, 
                          models, forward_fn, ablations, robustness_results,
                          model_type='cnn', missing_value=-100.0):
    """Run all validation checks and return consolidated results."""
    all_results = {}
    
    all_results['data_split'] = test_data_split(X_train, X_val)
    all_results['channel_count'] = test_channel_count(X_train, models, model_type)
    all_results['dropout_function'] = test_dropout_function(X_train[:32], missing_value=missing_value)
    all_results['ablation_suite'] = test_ablation_suite(ablations, X_val, missing_value=missing_value)
    all_results['model_forward'] = test_model_forward(models, forward_fn, X_val[:32], missing_value=missing_value)
    all_results['robustness_sanity'] = test_robustness_sanity(robustness_results)
    
    return all_results


# ==========================================
# 7. Main Execution (Looping over ALL Test Files)
# ==========================================

if __name__ == "__main__":
    
    # UPDATE YOUR PATHS HERE
    TRAIN_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\training" 
    TEST_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\testing"
    
    ##Noisy test data. For model validation.
    #TEST_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\Data For Model\Test Data"  #

    # TRAIN_DIR = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\gun_conditioning_spike_cleaned\v8 spikes cleaned until max\training" 
    # TEST_DIR = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\gun_conditioning_spike_cleaned\v8 spikes cleaned until max\testing"
    
    # ==========================================
    # HYPERPARAMETERS (all in one place for easy tuning)
    # ==========================================
    HYPERPARAMS = {
        'sequence_length': 30,
        'learning_rate': 0.00005,
        'epochs': 50,
        'batch_size': 128,
        'n_filters1': 128,
        'n_filters2': 256,
        'kernel_size': 3,
        'pool_size': 2,
        'hidden_dim': 64,  # For LSTM
        # Dropout/Missing data augmentation
        'prob_full_missing': 0.2,          # 20% chance of full window dropout
        'prob_block_missing': 0.3,         # 30% chance of block dropout
        'dropout_min_block_pct': 0.2,      # Min block = 20% of seq_length
        'dropout_max_block_pct': 0.8,      # Max block = 80% of seq_length
        'missing_value': -100.0,           # Placeholder for missing/dropped data
    }
    
    SEQ_LEN = HYPERPARAMS['sequence_length']
    MODEL_TYPE = 'cnn'  # 'cnn' or 'lstm'
    
    # ==========================================
    # CREATE OUTPUT FOLDER
    # ==========================================
    OUTPUT_FOLDER = create_output_folder(prefix="polgun_cnn")

    FEATURE_COLS = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]

    # Noise thresholds from TRAIN folder (MAD baseline)
    NOISE_THRESHOLDS = estimate_folder_noise_thresholds(TRAIN_DIR, FEATURE_COLS, k=1.0)

    
    # 1. Calculate Constraints
    MAX_VOLTAGE_LIMIT = find_global_max_voltage(TRAIN_DIR)
    AVG_STEP_SIZE = calculate_average_step_size(TRAIN_DIR)
    
    # 2. Prepare Data (shuffled split, NaN-aware normalization)
    data, pos_weight, scaler = prepare_all_data(
        TRAIN_DIR, TEST_DIR, SEQ_LEN,
        noise_thresholds=NOISE_THRESHOLDS,
        filter_quiet_negatives=True
    )

    
    # 3. Train (with sensor dropout augmentation)
    models, fwd_fn, history, initial_weights, final_weights, weight_history = train_model(
        MODEL_TYPE, data, pos_weight, 
        epochs=HYPERPARAMS['epochs'], 
        batch_size=HYPERPARAMS['batch_size'],
        learning_rate=HYPERPARAMS['learning_rate'],
        n_filters1=HYPERPARAMS['n_filters1'],
        n_filters2=HYPERPARAMS['n_filters2'],
        kernel_size=HYPERPARAMS['kernel_size'],
        pool_size=HYPERPARAMS['pool_size'],
        hidden_dim=HYPERPARAMS['hidden_dim'],
        prob_full_missing=HYPERPARAMS['prob_full_missing'],
        prob_block_missing=HYPERPARAMS['prob_block_missing'],
        min_block_pct=HYPERPARAMS['dropout_min_block_pct'],
        max_block_pct=HYPERPARAMS['dropout_max_block_pct'],
        missing_value=HYPERPARAMS['missing_value']
    )

    
    # ==========================================
    # 3.5 ROBUSTNESS ABLATION SUITE (Validation + Test)
    # ==========================================
    print("\n>>> Running Robustness Ablation Suite...")
    
    # --- VALIDATION SET ABLATIONS ---
    print("\n--- Validation Set Ablations ---")
    val_ablations, val_missing_mask = create_validation_ablations(
        data['X_val'], data['y_val'], 
        missing_value=HYPERPARAMS['missing_value']
    )
    val_robustness = evaluate_robustness(models, fwd_fn, val_ablations, missing_mask=val_missing_mask, filter_missing=True)
    
    # --- TEST SET ABLATIONS ---
    print("\n--- Test Set Ablations ---")
    test_ablations, test_missing_mask = create_test_ablations(
        data['X_test'], data['y_test'], 
        missing_value=HYPERPARAMS['missing_value']
    )
    test_robustness = evaluate_robustness(models, fwd_fn, test_ablations, missing_mask=test_missing_mask, filter_missing=False)
    
    # Run validation checks
    validation_checks = run_validation_checks(
        data['X_train'], data['X_val'],
        models, fwd_fn, val_ablations, val_robustness,
        model_type=MODEL_TYPE,
        missing_value=HYPERPARAMS['missing_value']
    )
    
    # Plot robustness ablation results
    plot_robustness_ablation(val_robustness, title_suffix="Validation - Clean Subset",
                            output_folder=OUTPUT_FOLDER, filename="robustness_ablation_val")
    plot_robustness_ablation(test_robustness, title_suffix="Test - All Samples",
                            output_folder=OUTPUT_FOLDER, filename="robustness_ablation_test")
    
    # Save comprehensive robustness report (includes both val and test)
    save_robustness_report(val_robustness, validation_checks, OUTPUT_FOLDER, 
                          test_robustness=test_robustness)
    
    # ==========================================
    # 4. NEW: CHANNEL IMPORTANCE VISUALIZATIONS
    # ==========================================
    print("\n>>> Generating Channel Importance Visualizations...")
    
    # Static visualizations (Initial vs Final)
    plot_channel_importance_analysis(
        initial_weights, final_weights, 
        output_folder=OUTPUT_FOLDER,
        channel_names=CHANNEL_NAMES
    )
    
    # Temporal visualizations (Per-epoch progression)
    plot_channel_importance_over_epochs(
        weight_history,
        output_folder=OUTPUT_FOLDER,
        channel_names=CHANNEL_NAMES
    )

    # ==========================================
    # 4.5 ACTIVATION STATISTICS (Conv1)
    # ==========================================
    print("\n>>> Analyzing Activation Statistics...")
    if MODEL_TYPE == 'cnn':
        analyze_activation_statistics(
            models,
            data['X_val'],
            channel_names=CHANNEL_NAMES,
            output_folder=OUTPUT_FOLDER,
        )
    else:
        print("Activation statistics are currently implemented for the CNN model only. Skipping for LSTM.")
    # ==========================================
    # 4.1 NEW: EXCITATION/INHIBITION ANALYSIS
    # ==========================================
    print("\n>>> Generating Excitation/Inhibition Analysis...")
    
    # Static E/I analysis (returns E/I ratios for consistency check)
    ei_ratios = plot_excitation_inhibition_analysis(
        initial_weights, final_weights,
        output_folder=OUTPUT_FOLDER,
        channel_names=CHANNEL_NAMES
    )
    
    # E/I over epochs
    plot_excitation_inhibition_over_epochs(
        weight_history,
        output_folder=OUTPUT_FOLDER,
        channel_names=CHANNEL_NAMES
    )
    
    # E/I Consistency check against expected physics
    print_ei_consistency_check(ei_ratios, channel_names=CHANNEL_NAMES)
    
    # ==========================================
    # 5. EXISTING VISUALIZATIONS (kept for compatibility)
    # ==========================================
    viz_layer(initial_weights, final_weights, "conv1", include_bias=False)
    viz_layer(initial_weights, final_weights, "linear", include_bias=True)
    plot_conv1_input_channel_hists(models, initial_weights=initial_weights, final_weights=final_weights)

    #########################################################################
    # Save weights comparison to text file (in output folder)
    #########################################################################
    w_init = initial_weights["conv1"]["weight"].detach().cpu().numpy()
    w_final = final_weights["conv1"]["weight"].detach().cpu().numpy()
    b_init = initial_weights["conv1"]["bias"].detach().cpu().numpy()
    b_final = final_weights["conv1"]["bias"].detach().cpu().numpy()
   
    out_path = os.path.join(OUTPUT_FOLDER, "conv1_weights_comparison.txt")

    with open(out_path, 'w') as f, redirect_stdout(f):
        print("conv1.weight shape:", w_init.shape)
        print("conv1.bias shape:  ", b_init.shape)
        print("conv1 weight count (no bias):", w_init.size)
        print("conv1 bias count:", b_init.size)

        O, I, K = w_init.shape
        for in_ch, ch_name in enumerate(CHANNEL_NAMES):
            init_ch = w_init[:, in_ch, :]
            final_ch = w_final[:, in_ch, :]

            print("\n" + "="*80)
            print(f"INPUT CHANNEL {in_ch}: {ch_name}  |  weights per channel = {init_ch.size} (128 filters × 3 kernel)")
            print("="*80)

            for out_ch in range(init_ch.shape[0]):
                ki = init_ch[out_ch]
                kf = final_ch[out_ch]
                print(
                    f"filter {out_ch:03d} | "
                    f"init [{ki[0]: .8f}, {ki[1]: .8f}, {ki[2]: .8f}]  ->  "
                    f"final [{kf[0]: .8f}, {kf[1]: .8f}, {kf[2]: .8f}]"
                )   

        print("\n" + "="*80)
        print("CONV1 BIAS: initial -> final (per output filter)")
        print("="*80)
        for out_ch in range(b_init.shape[0]):
            print(f"filter {out_ch:03d} | init_bias {b_init[out_ch]: .8f} -> final_bias {b_final[out_ch]: .8f}")
    print(f"Saved: {out_path}")

    # ==========================================
    # 6. Plot Training Curves (with saving)
    # ==========================================
    plot_training_curves(history, output_folder=OUTPUT_FOLDER)

    # ==========================================
    # 7. Evaluate on Test Set
    # ==========================================
    evaluate_test_set(models, fwd_fn, data, pos_weight)
    
    # ==========================================
    # 8. SAVE CHECKPOINT
    # ==========================================
    print("\n>>> Saving model checkpoint...")
    save_checkpoint(
        output_folder=OUTPUT_FOLDER,
        models=models,
        scaler=scaler,
        hyperparams=HYPERPARAMS,
        history=history,
        weight_history=weight_history,
        initial_weights=initial_weights,
        final_weights=final_weights,
        pos_weight=pos_weight,
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        max_voltage=MAX_VOLTAGE_LIMIT,
        avg_step=AVG_STEP_SIZE,
        channel_names=CHANNEL_NAMES,
        model_type=MODEL_TYPE
    )

    # ==========================================
    # 9. VALIDATION TESTS
    # ==========================================
    print("\n>>> Running Validation Tests...")
    
    # 9.1 Channel Ablation Study
    print("\n--- Channel Ablation Study ---")
    ablation_results = channel_ablation_study(
        models, fwd_fn,
        data['X_test'], data['y_test'],
        channel_names=CHANNEL_NAMES,
        output_folder=OUTPUT_FOLDER
    )
    
    # 9.2 Synthetic Test Cases
    print("\n--- Synthetic Test Cases ---")
    synthetic_cases = create_synthetic_test_cases(scaler, SEQ_LEN)
    synthetic_results = run_synthetic_tests(models, fwd_fn, synthetic_cases)
    
    # 9.3 Temporal Sensitivity Test
    print("\n--- Temporal Sensitivity Test ---")
    temporal_results = temporal_sensitivity_test(
        models, fwd_fn, scaler, SEQ_LEN,
        output_folder=OUTPUT_FOLDER
    )

    #  # --- NEW: Feature-space visualizations (use VAL set) ---
    # plot_3d_real_labels(
    #     data['X_val'], data['y_val'], scaler,
    #     title="VAL: Real Labels (3D)",
    #     start_from_zero=True,
    #     elev=10, azim=210
    # )   

    # plot_3d_predicted_labels(
    #     models, fwd_fn, data['X_val'], data['y_val'], scaler,
    #     title="VAL: Predicted Labels (3D)",
    #     start_from_zero=True,
    #     elev=10, azim=210
    # )
     
    # TRAIN
#     plot_3d_real_labels(data['X_train'], data['y_train'], scaler, title="TRAIN: Real (3D)")
#     plot_3d_predicted_labels(models, fwd_fn, data['X_train'], data['y_train'], scaler, title="TRAIN: Pred (3D)")

# # VAL
#     plot_3d_real_labels(data['X_val'], data['y_val'], scaler, title="VAL: Real (3D)")
#     plot_3d_predicted_labels(models, fwd_fn, data['X_val'], data['y_val'], scaler, title="VAL: Pred (3D)")

# # TEST (optional but ideal)
#     plot_3d_real_labels(data['X_test'], data['y_test'], scaler, title="TEST: Real (3D)")
#     plot_3d_predicted_labels(models, fwd_fn, data['X_test'], data['y_test'], scaler, title="TEST: Pred (3D)")



#     plot_decision_boundary_slices_shaded(
#         models, fwd_fn, data, scaler, SEQ_LEN,
#         overlay="pred",
#         use_set="test"
#     )



#     plot_decision_boundary_slices(
#         models, fwd_fn, data, scaler, SEQ_LEN,
#         n_grid=50,
#         band_frac=0.10,   # try 0.05 to make slice stricter
#         overlay="both"    # "real", "pred", or "both"
#     )

    
    # 5. Run Simulation Loop (ALL FILES)
    print(f"\n>>> Starting Simulation Loop on folder: {TEST_DIR}")
    test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))
    
    if not test_files:
        print("No CSV files found in Test Directory.")
    

    for i, test_file in enumerate(test_files):
        if "_SIMULATED" in test_file:
            continue

        # Trend plot (6-panel) with figure saving
        plot_test_file_trends(
            test_file,
            models,
            fwd_fn,
            scaler,
            sequence_length=SEQ_LEN,
            output_folder=OUTPUT_FOLDER
        )


    for i, test_file in enumerate(test_files):
        # Safety Check: Skip files that are previous simulation outputs
        if "_SIMULATED" in test_file:
            continue
            
        print(f"\n[{i+1}/{len(test_files)}] Simulating: {os.path.basename(test_file)}")
        
        test_single_file_simulation(
            test_file, 
            models, 
            fwd_fn, 
            scaler, 
            sequence_length=SEQ_LEN,
            step_size=AVG_STEP_SIZE,
            initial_A=0, 
            max_limit_A=MAX_VOLTAGE_LIMIT,
            output_folder=OUTPUT_FOLDER
        )

    print("\nAll Simulations Complete.")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("RUN COMPLETE - OUTPUT SUMMARY")
    print("="*70)
    print(f"All outputs saved to: {OUTPUT_FOLDER}/")
    print(f"\nFigures ({OUTPUT_FOLDER}/figures/):")
    print(f"  - channel_importance_*.png      (L2 norm analysis)")
    print(f"  - excitation_inhibition_*.png   (E/I sign-aware analysis)")
    print(f"  - ei_ratio_over_epochs.png      (E/I dynamics)")
    print(f"  - ablation_study_results.png    (Channel importance validation)")
    print(f"  - temporal_sensitivity_test.png (Spike position sensitivity)")
    print(f"  - training_curves.png")
    print(f"  - trend_*.png, simulation_*.png (Per test file)")
    print(f"\nCheckpoints:")
    print(f"  - {OUTPUT_FOLDER}/checkpoint.pth      (lightweight)")
    print(f"  - {OUTPUT_FOLDER}/full_checkpoint.pth (complete)")
    print(f"\nLogs:")
    print(f"  - {OUTPUT_FOLDER}/conv1_weights_comparison.txt")
    print("="*70)
    print("\nTo load this model later:")
    print(f'  models, fwd_fn, scaler, ckpt = load_checkpoint("{OUTPUT_FOLDER}/full_checkpoint.pth")')
    print("="*70)