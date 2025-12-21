##LSTM or CNN gemini LEREC v5 with BatchNorm + MaxPool + max voltage limit + dynamic step size
## The plots shown on the test file voltage simulation are normalized.

## included class decision boundary plots for 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # <--- ADDED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

import warnings


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
# 1. Physics Constraints (Helper Functions)
# ==========================================
# GOAL: Scan training data BEFORE training to learn physical limits of the machine.
# ==========================================

def find_global_max_voltage(folder_path, voltage_col='glassmanDataXfer:hvPsVoltageMeasM'):
    """
    WHAT: Determines the absolute safety limit (Max Voltage) of the Gun.
    HOW:  Scans every CSV in the training folder, reads ONLY the voltage column,
          and finds the global maximum. This is passed to the Simulation as a hard Interlock.
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
    WHAT: Calculates the physics of the ramp (How many Volts does a 'Step Up' actually add?).
    HOW:  It aligns row t (Command=1) with row t+1 (Voltage).
          It calculates V(t+1) - V(t) for every successful ramp and averages them.
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



def robust_noise_sigma_mad_diff(series: pd.Series):
    """
    Robust noise sigma estimate using MAD of first differences.
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

    df = df.copy()

# -------- DEBUG: BEFORE time handling ----------
    
    if 'VoltageChange' in df.columns:
        vc_before = pd.to_numeric(df['VoltageChange'], errors='coerce').values
        print(f"[DEBUG BEFORE] rows={len(df)} | VoltageChange NaNs={np.isnan(vc_before).sum()}")
    else:
        print(f"[DEBUG BEFORE] rows={len(df)} | VoltageChange column missing")
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
    if 'VoltageChange' in df.columns:
        vc_tmp = pd.to_numeric(df['VoltageChange'], errors='coerce').values
        print(f"[DEBUG] After time handling: rows={len(df)} | VoltageChange NaNs={np.isnan(vc_tmp).sum()}")
    else:
        print(f"[DEBUG] After time handling: rows={len(df)} | VoltageChange column missing")
# ---------------------------------------------------------------


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


    
    # 2. Handle Inputs and Masks (NO VOLTAGE INCLUDED IN FEATURES)
    feature_cols = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]
    input_data = []
    
    for col in feature_cols:
        # --- MASK LOGIC ---
        mask = (~df[col].isna()).astype(float)
        val = df[col].fillna(0).astype(float)
        
        input_data.append(val.values)
        input_data.append(mask.values)
        
    # 3. Add Previous Output (A1) to Input Features
    a1_history = df['VoltageChange'].fillna(0).values
    input_data.append(a1_history)

    # Stack features: [Current, Mask, Pressure, Mask, Rad, Mask, Prev_A1]
    X_raw = np.stack(input_data, axis=1)
    
    # 4. Apply Scaler (Only to values - indices 0, 2, 4)
    X_raw[:, [0, 2, 4]] = scaler.transform(X_raw[:, [0, 2, 4]])
    
    # 5. Create Sequences
    if len(X_raw) <= sequence_length:
        return np.empty((0, sequence_length, 7)), np.empty((0, 1))

    X_seq, y_seq = [], []
    
    y_raw_target = df['VoltageChange'].values 

    print(f"[DEBUG] Max possible windows={max(0, len(X_raw)-sequence_length)} | "
      f"NaN targets={np.isnan(y_raw_target).sum()}")

    
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
        return np.empty((0, sequence_length, 7)), np.empty((0, 1))
    
        # --- Debug summary per file ---
    if filter_quiet_negatives and noise_thresholds:
        if total_zero_targets > 0:
            pct = 100.0 * skipped_quiet_zero / total_zero_targets
            print(f"[Quiet-Neg Filter] {file_label}: skipped {skipped_quiet_zero}/{total_zero_targets} "
                  f"quiet 0-targets ({pct:.1f}%)")
        else:
            print(f"[Quiet-Neg Filter] {file_label}: no 0-targets found to evaluate.")

        
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

def prepare_all_data(train_folder, test_folder, sequence_length,
                     noise_thresholds=None,
                     filter_quiet_negatives=True):

    """
    WHAT: Orchestrates data loading, scaling, and splitting.
    """
    print(f"Loading Data...")
    train_dfs = read_folder_csvs(train_folder)
    test_dfs = read_folder_csvs(test_folder)
    
    # --- FIT SCALER ON TRAIN ONLY ---
    # Logic: Prevents "Data Leakage". Test data must look like 'new, unseen' data.
    print("Fitting Scaler...")
    train_concat = pd.concat(train_dfs, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_concat[["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]].fillna(0))
    
    print("Processing Train Sequences...")
    X_train_list, y_train_list = [], []
    for df in train_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler, sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=True
        )

        if len(x_s) > 0:
            X_train_list.append(x_s)
            y_train_list.append(y_s)
            
    X_train_all = np.concatenate(X_train_list)
    y_train_all = np.concatenate(y_train_list)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.15, shuffle=True, random_state=97)
    
    print("Processing Test Sequences...")
    X_test_list, y_test_list = [], []
    for df in test_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler, sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=True
        )

        if len(x_s) > 0:
            X_test_list.append(x_s)
            y_test_list.append(y_s)
            
    if X_test_list:
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        X_test = np.empty((0, sequence_length, 7))
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


    print("\n" + "="*40)
    print("DATASET SUMMARY")
    print("="*40)
    print(f"Training Set:   {data['X_train'].shape}  (Samples, Seq_Len, Channels)")
    print(f"Validation Set: {data['X_val'].shape}")
    print(f"Test Set:       {data['X_test'].shape}")
    print("-" * 40)
    print(f"Class Balance (Train):")
    print(f"  - Stable (0):   {int(n_neg)}")
    print(f"  - Increase (1): {int(n_pos)}")
    print(f"  - Calculated Pos_Weight: {pos_weight.item():.4f}")
    print("="*40 + "\n")
    return data, pos_weight, scaler

# ==========================================
# 3. Model Architecture (CNN + BN + Pool)
# ==========================================


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


def create_cnn_model(input_dim, seq_len):
    """
    WHAT: Defines the Neural Network structure.
    ARCH: Conv1d -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear
    """
    # Define Filter Sizes to avoid mismatches
    n_filters1 = 128
    n_filters2 = 256
    
    # Block 1: Feature Extraction (Low level)
    conv1 = nn.Conv1d(input_dim, n_filters1, kernel_size=3, padding=1)
    bn1 = nn.BatchNorm1d(n_filters1) # Normalizes activations for stability
    
    # Block 2: Feature Extraction (High level)
    conv2 = nn.Conv1d(n_filters1, n_filters2, kernel_size=3, padding=1)
    bn2 = nn.BatchNorm1d(n_filters2)
    
    # Pooling: Shrinks time dimension by half (Downsampling)
    pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    # --- LINEAR LAYER SIZING MATH ---
    # Logic: We must calculate exactly how many neurons are left after flattening.
    # Pool 1 reduces length: 25 -> 12
    len_after_pool1 = seq_len // 2
    # Pool 2 reduces length: 12 -> 6
    len_after_pool2 = len_after_pool1 // 2
    
    # Total inputs = Filters * Remaining Time Steps
    linear_input_size = n_filters2 * len_after_pool2
    #linear_input_size = n_filters2 * seq_len  # No pooling used
    
   # print(f"Model Init: SeqLen {seq_len} -> Reduced to {len_after_pool2} steps -> Linear Input {linear_input_size}")
    
    linear = nn.Linear(linear_input_size, 1)

    return {'conv1': conv1, 'bn1': bn1, 'conv2': conv2, 'bn2': bn2, 'pool':pool, 'linear': linear}   

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

def train_model(model_type, data, pos_weight, epochs=60, batch_size=128):
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    print('input for model', X_train[30:60])
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    
   # --- MODEL SWITCHING LOGIC RESTORED ---
    if model_type == 'lstm':
        print("Initializing LSTM Model...")
        models = create_lstm_model(input_dim, hidden_dim=64)
        forward_fn = forward_lstm
    else:
        print("Initializing CNN Model...")
        models = create_cnn_model(input_dim, seq_len)
        forward_fn = forward_cnn
    params = [p for layer in models.values() for p in layer.parameters()]

    optimizer = optim.Adam(params, lr=0.000005)
    
        # --- DEBUG: print model parameter summary ---
    summarize_model_parameters(models, model_type=model_type)

    # LOSS FUNCTION:
    # BCEWithLogitsLoss combines Sigmoid + CrossEntropy. 
    # pos_weight scales the loss for the minority class (Ramps).
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

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
            
    return models, forward_fn, history



# ==========================================
# 5. Visualizations (RESTORED)
# ==========================================


# ==========================================
# 5. Evaluation Metrics 
# ==========================================

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
    X shape: (N, Seq, 7)
    Returns raw (inverse-scaled) physical values.
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = X

    last = X_np[:, -1, :]  # last timestep
    vals_scaled = last[:, [0, 2, 4]]  # scaled current, pressure, radiation

    # Inverse transform back to physical units
    vals_real = scaler.inverse_transform(vals_scaled)

    current = vals_real[:, 0]
    pressure = vals_real[:, 1]
    radiation = vals_real[:, 2]
    return current, pressure, radiation

from mpl_toolkits.mplot3d import Axes3D

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
    Build a synthetic constant sequence compatible with your 7-channel input.
    We fix masks=1 and prev_A1=0.
    """
    scaled = scaler.transform([[curr, pres, rad]])[0]
    step = np.array([scaled[0], 1.0, scaled[1], 1.0, scaled[2], 1.0, 0.0], dtype=np.float32)
    seq = np.tile(step, (seq_len, 1))  # (Seq, 7)
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



def plot_training_curves(history):
    """
    Plots the Loss and Accuracy curves after training.
    """
    epochs = range(len(history['val_loss']))
    plt.figure(figsize=(12, 5))
    
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
    plt.show()

def build_sequences_with_indices_for_plot(df, scaler, sequence_length):
    """
    Lightweight sequence builder for plotting.
    Matches your feature construction:
    [Current, Mask, Pressure, Mask, Rad, Mask, Prev_A1]
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

    # Build inputs with masks
    input_data = []
    for col in feature_cols:
        mask = (~df[col].isna()).astype(float)
        val = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        input_data.append(val.values)
        input_data.append(mask.values)

    # Prev A1 history
    a1_history = pd.to_numeric(df['VoltageChange'], errors='coerce').fillna(0).values
    input_data.append(a1_history)

    X_raw = np.stack(input_data, axis=1)  # (T, 7)

    # Scale ONLY value columns
    X_raw[:, [0, 2, 4]] = scaler.transform(X_raw[:, [0, 2, 4]])

    y_raw = pd.to_numeric(df['VoltageChange'], errors='coerce').values

    if len(X_raw) <= sequence_length:
        return np.empty((0, sequence_length, 7)), np.empty((0, 1)), np.array([], dtype=int), df

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
        return np.empty((0, sequence_length, 7)), np.empty((0, 1)), np.array([], dtype=int), df

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



def plot_test_file_trends(filepath, models, forward_fn, scaler, sequence_length):
    
    # 6-row subplot:
    # 1) Real VoltageChange (0/1)
    # 2) Pred VoltageChange (0/1)
    # 3) Real Voltage (continuous)
    # 4) Current vs time
    # 5) Pressure vs time
    # 6) Radiation vs time
    # """

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
    plt.show()



# ==========================================
# 6. Simulation (UPDATED: Real vs Sim Plot)
# ==========================================

def test_single_file_simulation(filepath, models, forward_fn, scaler, sequence_length, step_size, initial_A=0, max_limit_A=None):
    print(f"\n--- Processing Simulation: {filepath} ---")
    
    try:
        df = pd.read_csv(filepath)
        
    except FileNotFoundError:
        print("File not found.")
        return

    # Process Input
    print('raw rows',df.shape)
    X_seq, y_seq = process_single_df_to_sequences(df, scaler, sequence_length)
    print('processed maybe',X_seq.shape)
    print('processed',y_seq.shape)
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

    plt.figure(figsize=(12, 6))

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
    plt.show()


# ==========================================
# 7. Main Execution (Looping over ALL Test Files)
# ==========================================

if __name__ == "__main__":
    
    # UPDATE YOUR PATHS HERE
    TRAIN_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\training" 
    TEST_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\testing"

    # TRAIN_DIR = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\gun_conditioning_spike_cleaned\v8 spikes cleaned until max\training" 
    # TEST_DIR = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\gun_conditioning_spike_cleaned\v8 spikes cleaned until max\testing"
    SEQ_LEN = 20

    FEATURE_COLS = ["GunCurrent.Avg","peg-BL-cc:pressureM","RadiationTotal"]

    # Noise thresholds from TRAIN folder (MAD baseline)
    NOISE_THRESHOLDS = estimate_folder_noise_thresholds(TRAIN_DIR, FEATURE_COLS, k=1.0)

    
    # 1. Calculate Constraints
    MAX_VOLTAGE_LIMIT = find_global_max_voltage(TRAIN_DIR)
    AVG_STEP_SIZE = calculate_average_step_size(TRAIN_DIR)
    # 2. Prepare Data
    data, pos_weight, scaler = prepare_all_data(
        TRAIN_DIR, TEST_DIR, SEQ_LEN,
        noise_thresholds=NOISE_THRESHOLDS,
        filter_quiet_negatives=True
    )


    
    # 3. Train
    models, fwd_fn, history = train_model('cnn', data, pos_weight, epochs=60, batch_size=128)
    
    # 4. Plot Training Curves
    plot_training_curves(history)

    # 5. Evaluate on Test Set
    evaluate_test_set(models, fwd_fn, data, pos_weight)

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

        # NEW: 4-panel trend plot
        plot_test_file_trends(
            test_file,
            models,
            fwd_fn,
            scaler,
            sequence_length=SEQ_LEN
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
            max_limit_A=MAX_VOLTAGE_LIMIT
        )

    print("\nAll Simulations Complete.")