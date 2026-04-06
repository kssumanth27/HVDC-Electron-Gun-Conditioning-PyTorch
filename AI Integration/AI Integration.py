####### STILL IN TESTING PHASE./ NOT THE FINAL CODE

##  Virtual Test Environment with Latency Simulation
##  Standalone script: loads a trained CNN/LSTM from full_checkpoint.pth,
##  runs row-by-row inference on a test CSV, and simulates real-system
##  latencies (T1-T10) with measured preprocessing / inference times.

import os
import sys
import io
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 0. LATENCY CONFIGURATION  (edit these to match your hardware)
# ============================================================
LATENCY_CONFIG = {
    "T1_psc_response_ms":    100,   # 10 Hz PSC response time  (simulated)
    "T2_readback_ms":        100,   # Readback time – 10 Hz + ethernet  (simulated)
    "T3_daq_trigger_ms":      25,   # Data-card trigger time  (simulated)
    "T4_network_storage_ms":  25,   # Storage over network  (simulated)
    # T5  – data extraction          → measured live
    # T6  – preprocessing           → measured live
    "T7_model_trigger_ms":     5,   # Stored model called / triggered  (simulated)
    # T8  – model reads data        → measured live
    # T9  – inference               → measured live
    "T10_command_send_ms":    20,   # Command sent to control system  (simulated)
}

CYCLE_BUDGET_MS = 1000.0   # 1-second real-time budget

DECISION_THRESHOLD = 0.0   # logit > this → decision = 1 (increase voltage)
                           # Default 0.0 matches training (sigmoid > 0.5)
                           # Raise to e.g. 0.5 for more conservative decisions
                           # Lower to e.g. -0.5 for more aggressive decisions

FORCE_INCREASE_FIRST_N = 2  # Force decision=1 for the first N inference steps
                             # Set to 0 to disable forced override

OUTPUT_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Codes\AI Integration results"

SENSOR_COLS   = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
VOLTAGE_COL   = "glassmanDataXfer:hvPsVoltageMeasM"
MISSING_VALUE = 0.0


# ============================================================
# 1. MODEL ARCHITECTURE  (minimal – needed to load checkpoint)
# ============================================================

def create_lstm_model(input_dim, hidden_dim):
    lstm   = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                     num_layers=2, batch_first=True)
    linear = nn.Linear(hidden_dim, 1)
    return {'lstm': lstm, 'linear': linear}


def forward_lstm(models, x):
    lstm_out, _ = models['lstm'](x)
    return models['linear'](lstm_out[:, -1, :])


def create_cnn_model(input_dim, seq_len,
                     n_filters1=128, n_filters2=256,
                     kernel_size=3, pool_size=2):
    conv1 = nn.Conv1d(input_dim, n_filters1,
                      kernel_size=kernel_size, padding=kernel_size // 2)
    bn1   = nn.BatchNorm1d(n_filters1)

    conv2 = nn.Conv1d(n_filters1, n_filters1,
                      kernel_size=kernel_size, padding=kernel_size // 2)
    bn2   = nn.BatchNorm1d(n_filters1)

    conv3 = nn.Conv1d(n_filters1, n_filters2,
                      kernel_size=kernel_size, padding=kernel_size // 2)
    bn3   = nn.BatchNorm1d(n_filters2)

    pool  = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    len_after_pool = seq_len // pool_size
    linear = nn.Linear(n_filters2 * len_after_pool, 1)

    return {'conv1': conv1, 'bn1': bn1,
            'conv2': conv2, 'bn2': bn2,
            'conv3': conv3, 'bn3': bn3,
            'pool': pool, 'linear': linear}


def forward_cnn(models, x):
    x = x.permute(0, 2, 1)
    # Block 1
    x = F.relu(models['bn1'](models['conv1'](x)))
    # Block 2
    x = models['pool'](F.relu(models['bn2'](models['conv2'](x))))
    # Block 3
    x = F.relu(models['bn3'](models['conv3'](x)))
    x = x.reshape(x.size(0), -1)
    return models['linear'](x)


# ============================================================
# 2. CHECKPOINT LOADER
# ============================================================

def load_checkpoint(checkpoint_path):
    """Load full_checkpoint.pth → model dict, forward fn, scaler, metadata."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_          = np.array(ckpt['scaler_mean'])
    scaler.scale_         = np.array(ckpt['scaler_scale'])
    scaler.var_           = np.array(ckpt['scaler_var'])
    scaler.n_features_in_ = ckpt['scaler_n_features']

    hp        = ckpt['hyperparameters']
    seq_len   = hp['sequence_length']
    input_dim = 4

    model_type = ckpt.get('model_type', 'cnn')

    if model_type == 'lstm':
        models     = create_lstm_model(input_dim,
                                       hidden_dim=hp.get('hidden_dim', 64))
        forward_fn = forward_lstm
    else:
        models     = create_cnn_model(input_dim, seq_len,
                                      n_filters1=hp.get('n_filters1', 128),
                                      n_filters2=hp.get('n_filters2', 256),
                                      kernel_size=hp.get('kernel_size', 3),
                                      pool_size=hp.get('pool_size', 2))
        forward_fn = forward_cnn

    for name, sd in ckpt['model_state_dict'].items():
        if name in models:
            models[name].load_state_dict(sd)
    for layer in models.values():
        layer.eval()

    print(">>> Checkpoint loaded successfully!")
    print(f"    Model type     : {model_type}")
    print(f"    Epochs trained : {len(ckpt['training_history']['train_loss'])}")
    print(f"    Sequence length: {seq_len}")
    print(f"    Max voltage    : {ckpt.get('max_voltage_limit', 'N/A')}")
    print(f"    Avg step size  : {ckpt.get('avg_step_size', 'N/A')}")

    # --- Full hyperparameters ---
    print("\n" + "=" * 60)
    print("HYPERPARAMETERS")
    print("=" * 60)
    for k, v in hp.items():
        print(f"  {k:30s}: {v}")

    # --- Scaler values per sensor channel ---
    ch_names = ckpt.get('channel_names', SENSOR_COLS)
    print("\n" + "=" * 60)
    print("SCALER (Z-score normalisation parameters)")
    print("=" * 60)
    for i, name in enumerate(ch_names[:len(scaler.mean_)]):
        print(f"  {name:35s}  mean={scaler.mean_[i]:.6f}  "
              f"std={scaler.scale_[i]:.6f}")

    # --- Layer-by-layer architecture ---
    print("\n" + "=" * 60)
    print(f"MODEL ARCHITECTURE [{model_type.upper()}]")
    print("=" * 60)
    total_params = 0
    for layer_name, layer in models.items():
        layer_params = 0
        print(f"\n  Layer: {layer_name}  ({layer.__class__.__name__})")
        for pname, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            layer_params += n
            total_params += n
            print(f"    {pname:20s}  shape={str(tuple(p.shape)):20s}  "
                  f"params={n:,}")
        if layer_params == 0:
            print("    (no trainable parameters)")
        else:
            print(f"    Subtotal: {layer_params:,}")
    print("\n" + "-" * 60)
    print(f"  TOTAL TRAINABLE PARAMETERS: {total_params:,}")
    print("=" * 60)

    return models, forward_fn, scaler, ckpt


# ============================================================
# 3. TEST-CSV LOADER
# ============================================================

def load_test_csv(csv_path):
    """Read test CSV → DataFrame with time column converted to seconds."""
    print(f"\nLoading test file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}   Columns: {len(df.columns)}")

    time_col = next((c for c in df.columns if c.lower() == 'time'), None)
    if time_col:
        try:
            df[time_col] = pd.to_numeric(df[time_col])
        except (ValueError, TypeError):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df[time_col] = (df[time_col] - df[time_col].min()).dt.total_seconds()
            except Exception as e:
                print(f"  Warning: could not parse time column – {e}")
                time_col = None

        if time_col:
            df = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col)
            dt = df[time_col].diff().dropna()
            if len(dt) > 0:
                freq = dt.mode().iloc[0] if len(dt.mode()) > 0 else dt.median()
                print(f"  Time column      : '{time_col}'")
                print(f"  Dominant timestep: {freq:.4f} s")

    for col in SENSOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        else:
            print(f"  Warning: sensor column '{col}' not found – filling with NaN")
            df[col] = np.nan

    if 'VoltageChange' in df.columns:
        df['VoltageChange'] = pd.to_numeric(df['VoltageChange'], errors='coerce')
        vc = df['VoltageChange'].values.copy()
        vc[vc == -1] = 0
        df['VoltageChange'] = vc
    else:
        print("  Warning: 'VoltageChange' column not found – ground truth unavailable")
        df['VoltageChange'] = np.nan

    df = df.reset_index(drop=True)
    print(f"  Rows after cleanup: {len(df)}")
    return df, time_col


# ============================================================
# 4. PREPROCESSING HELPER  (per-step, matches training code)
# ============================================================

def preprocess_window(sensor_block, prev_a1_block, scaler):
    """
    Takes raw sensor values (seq_len, 3) and prev_a1 values (seq_len,),
    returns a ready-to-infer tensor of shape (1, seq_len, 4).
    """
    sensor = sensor_block.copy()
    nan_mask = np.isnan(sensor)
    sensor_clean = np.nan_to_num(sensor, nan=0.0)
    sensor_scaled = scaler.transform(sensor_clean)
    for ch in range(3):
        sensor_scaled[nan_mask[:, ch], ch] = MISSING_VALUE

    window = np.concatenate([sensor_scaled,
                             prev_a1_block.reshape(-1, 1)], axis=1)
    return torch.FloatTensor(window).unsqueeze(0)    # (1, seq_len, 4)


# ============================================================
# 5. LATENCY HELPERS
# ============================================================

def simulated_delay(label, ms):
    """Sleep for a simulated hardware delay and return elapsed ms."""
    time.sleep(ms / 1000.0)
    return ms


# ============================================================
# 6. MAIN SIMULATION LOOP
# ============================================================

def run_simulation(models, forward_fn, scaler, ckpt, df, time_col):

    hp       = ckpt['hyperparameters']
    seq_len  = hp['sequence_length']
    avg_step = ckpt.get('avg_step_size', 10.0)
    max_volt = ckpt.get('max_voltage_limit', float('inf'))

    cfg = LATENCY_CONFIG
    sim_fixed_ms = (cfg['T1_psc_response_ms']
                    + cfg['T2_readback_ms']
                    + cfg['T3_daq_trigger_ms']
                    + cfg['T4_network_storage_ms']
                    + cfg['T7_model_trigger_ms']
                    + cfg['T10_command_send_ms'])

    # Sensor array for the whole file (raw, unscaled)
    sensor_raw = df[SENSOR_COLS].values                  # (N, 3)
    gt_vc      = df['VoltageChange'].values              # (N,)  ground truth

    # Bootstrap Prev_A1 from CSV for the first seq_len rows
    csv_prev_a1 = df['VoltageChange'].fillna(0).values.astype(float)

    n_rows      = len(df)
    first_infer = seq_len      # index of the first row we predict
    n_steps     = n_rows - seq_len

    if n_steps <= 0:
        print("ERROR: test file has fewer rows than sequence_length. Nothing to do.")
        return None

    # ---- logging accumulators ----
    decision_log  = []          # model decisions (grows each step)
    step_records  = []
    sim_voltage   = 0.0         # simulated voltage starts at 0

    all_t5 = []
    all_t6 = []
    all_t8 = []
    all_t9 = []
    all_cycle = []
    warnings_count = 0

    print("\n" + "=" * 70)
    print(f"  STARTING VIRTUAL TEST  |  {n_steps} steps  |  seq_len={seq_len}")
    print("=" * 70)

    for step in range(n_steps):
        row_idx = first_infer + step          # the row we're predicting

        print(f"\n--- Step {step}/{n_steps-1}  (row {row_idx}) ---")

        # ============ T1 : PSC response ============
        t1 = cfg['T1_psc_response_ms']
        print(f"  T1  – PSC responding to voltage command ........... "
              f"{t1} ms  (simulated)")
        simulated_delay("T1", t1)

        # ============ T2 : Readback ============
        t2 = cfg['T2_readback_ms']
        print(f"  T2  – Voltage readback over ethernet .............. "
              f"{t2} ms  (simulated)")
        simulated_delay("T2", t2)

        # ============ T3 : DAQ trigger ============
        t3 = cfg['T3_daq_trigger_ms']
        print(f"  T3  – Data-card triggered ......................... "
              f"{t3} ms  (simulated)")
        simulated_delay("T3", t3)

        # ============ T4 : Network storage ============
        t4 = cfg['T4_network_storage_ms']
        print(f"  T4  – Sensor data stored over network ............. "
              f"{t4} ms  (simulated)")
        simulated_delay("T4", t4)

        # ============ T5 : Data extraction (MEASURED) ============
        t5_start  = time.perf_counter()

        win_start = row_idx - seq_len
        win_end   = row_idx                    # exclusive

        sensor_window = sensor_raw[win_start:win_end].copy()  # (seq_len, 3)

        # Build Prev_A1 for this window:
        #   rows before 'first_infer' → use CSV ground truth
        #   rows >= first_infer       → use decision_log
        prev_a1_window = np.zeros(seq_len, dtype=float)
        for k in range(seq_len):
            abs_row = win_start + k
            if abs_row < first_infer:
                prev_a1_window[k] = csv_prev_a1[abs_row]
            else:
                dl_idx = abs_row - first_infer
                if dl_idx < len(decision_log):
                    prev_a1_window[k] = decision_log[dl_idx]
                else:
                    prev_a1_window[k] = 0.0

        t5_ms = (time.perf_counter() - t5_start) * 1000.0
        all_t5.append(t5_ms)
        print(f"  T5  – Data extraction (CSV + decision log) ........ "
              f"{t5_ms:.3f} ms  (measured)")

        # ============ T6 : Preprocessing (MEASURED) ============
        t6_start = time.perf_counter()

        x_tensor = preprocess_window(sensor_window, prev_a1_window, scaler)

        t6_ms = (time.perf_counter() - t6_start) * 1000.0
        all_t6.append(t6_ms)
        print(f"  T6  – Preprocessing (scale + window build) ........ "
              f"{t6_ms:.3f} ms  (measured)")

        # ============ T7 : Model trigger ============
        t7 = cfg['T7_model_trigger_ms']
        print(f"  T7  – Stored model triggered ...................... "
              f"{t7} ms  (simulated)")
        simulated_delay("T7", t7)

        # ============ T8 : Model reads data (MEASURED) ============
        t8_start  = time.perf_counter()
        x_ready   = x_tensor  # already a tensor; in a real system this is the transfer step
        t8_ms     = (time.perf_counter() - t8_start) * 1000.0
        all_t8.append(t8_ms)
        print(f"  T8  – Model read processed data ................... "
              f"{t8_ms:.3f} ms  (measured)")

        # ============ T9 : Inference (MEASURED) ============
        t9_start = time.perf_counter()
        with torch.inference_mode():
            logit = forward_fn(models, x_ready)
        t9_ms = (time.perf_counter() - t9_start) * 1000.0
        all_t9.append(t9_ms)

        logit_val      = logit.item()
        model_decision = 1 if logit_val > DECISION_THRESHOLD else 0
        forced         = step < FORCE_INCREASE_FIRST_N
        decision       = 1 if forced else model_decision
        decision_log.append(float(decision))

        print(f"  T9  – Model inference ............................. "
              f"{t9_ms:.3f} ms  (measured)")
        if forced:
            print(f"        logit = {logit_val:+.4f}  →  model_decision = {model_decision}"
                  f"  →  decision = 1  [FORCED OVERRIDE {step+1}/{FORCE_INCREASE_FIRST_N}]")
        else:
            print(f"        logit = {logit_val:+.4f}  →  decision = {decision}")

        # ============ T10 : Send command ============
        t10 = cfg['T10_command_send_ms']
        print(f"  T10 – Sending decision to control system .......... "
              f"{t10} ms  (simulated)")
        simulated_delay("T10", t10)

        # ============ Voltage simulation ============
        if decision == 1:
            sim_voltage = min(sim_voltage + avg_step, max_volt)
        print(f"        Simulated voltage: {sim_voltage:.2f}")

        # ============ Cycle total ============
        cycle_ms = (sim_fixed_ms + t5_ms + t6_ms + t8_ms + t9_ms)
        all_cycle.append(cycle_ms)
        over = cycle_ms > CYCLE_BUDGET_MS
        if over:
            warnings_count += 1
        flag = "  *** WARNING: EXCEEDS 1 s ***" if over else "  OK"
        print(f"  TOTAL CYCLE: {cycle_ms:.2f} ms{flag}")

        # Ground truth
        gt = gt_vc[row_idx] if not np.isnan(gt_vc[row_idx]) else None
        gt_str = f"{int(gt)}" if gt is not None else "N/A"
        match  = (decision == gt) if gt is not None else None
        print(f"  Ground truth: {gt_str}   Match: {match}")

        step_records.append({
            'step':            step,
            'row_idx':         row_idx,
            'logit':           logit_val,
            'model_decision':  model_decision,
            'decision':        decision,
            'forced_override':  forced,
            'ground_truth':    gt,
            'simulated_voltage': sim_voltage,
            'T5_extraction_ms': t5_ms,
            'T6_preprocess_ms': t6_ms,
            'T8_data_feed_ms':  t8_ms,
            'T9_inference_ms':  t9_ms,
            'cycle_total_ms':   cycle_ms,
            'over_budget':      over,
        })

    return step_records, all_t5, all_t6, all_t8, all_t9, all_cycle, warnings_count


# ============================================================
# 7. SUMMARY & EXPORT
# ============================================================

def print_summary(step_records, all_t5, all_t6, all_t8, all_t9, all_cycle, warnings_count):

    decisions = [r['decision']     for r in step_records]
    gts       = [r['ground_truth'] for r in step_records if r['ground_truth'] is not None]
    preds_gt  = [(r['decision'], int(r['ground_truth']))
                 for r in step_records if r['ground_truth'] is not None]

    print("\n" + "=" * 70)
    print("  SIMULATION SUMMARY")
    print("=" * 70)

    # ---- Accuracy / classification ----
    if preds_gt:
        y_pred = [p for p, _ in preds_gt]
        y_true = [g for _, g in preds_gt]
        acc  = sum(1 for p, g in zip(y_pred, y_true) if p == g) / len(y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print(f"\n  Steps with ground truth : {len(preds_gt)}")
        print(f"  Accuracy                : {acc:.4f}  ({100*acc:.2f}%)")
        print(f"  Precision               : {prec:.4f}")
        print(f"  Recall                  : {rec:.4f}")
        print(f"  F1 Score                : {f1:.4f}")
        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        print(f"               Pred 0    Pred 1")
        print(f"    True 0    {cm[0,0]:>6}    {cm[0,1]:>6}")
        print(f"    True 1    {cm[1,0]:>6}    {cm[1,1]:>6}")
    else:
        print("  No ground-truth available – skipping accuracy metrics.")

    # ---- Decision counts ----
    n1 = sum(decisions)
    n0 = len(decisions) - n1
    print(f"\n  Total decisions         : {len(decisions)}")
    print(f"    Stable (0)            : {n0}")
    print(f"    Increase (1)          : {n1}")
    print(f"  Final simulated voltage : {step_records[-1]['simulated_voltage']:.2f}")

    # ---- Timing statistics ----
    def _stats(arr, label):
        a = np.array(arr)
        print(f"\n  {label}:")
        print(f"    Min   : {a.min():.3f} ms")
        print(f"    Max   : {a.max():.3f} ms")
        print(f"    Mean  : {a.mean():.3f} ms")
        print(f"    Median: {np.median(a):.3f} ms")

    _stats(all_t5,    "T5 – Data extraction (measured)")
    _stats(all_t6,    "T6 – Preprocessing (measured)")
    _stats(all_t8,    "T8 – Data feed to model (measured)")
    _stats(all_t9,    "T9 – Inference (measured)")
    _stats(all_cycle, "Total cycle time")

    sim_sum = sum(LATENCY_CONFIG.values())
    print(f"\n  Simulated (fixed) portion per step : {sim_sum} ms")
    print(f"  Measured  (variable) avg per step  : "
          f"{np.mean(all_t5) + np.mean(all_t6) + np.mean(all_t8) + np.mean(all_t9):.3f} ms")
    print(f"\n  Steps exceeding 1-second budget    : "
          f"{warnings_count} / {len(step_records)}")
    print("=" * 70)


def export_csv(step_records, output_path):
    df_out = pd.DataFrame(step_records)
    df_out.to_csv(output_path, index=False)
    print(f"\n  Results exported to: {output_path}")


# ============================================================
# 8. VOLTAGE COMPARISON PLOT
# ============================================================

def plot_voltage_comparison(step_records, df, seq_len, output_folder):
    """
    Plots min-max normalised simulated voltage (from decision_log)
    vs real voltage (from CSV column) and saves as PNG.
    """
    sim_arr = np.array([r['simulated_voltage'] for r in step_records])

    # Real voltage from CSV
    real_arr = None
    for col_name in [VOLTAGE_COL, 'hvps.lerec:voltageM']:
        if col_name in df.columns:
            raw = pd.to_numeric(df[col_name], errors='coerce').values
            real_segment = raw[seq_len: seq_len + len(sim_arr)]
            if len(real_segment) == len(sim_arr):
                real_arr = real_segment
            break

    fig, ax = plt.subplots(figsize=(14, 5))

    # Min-max normalise simulated voltage
    s_min, s_max = np.nanmin(sim_arr), np.nanmax(sim_arr)
    if s_max > s_min:
        sim_norm = (sim_arr - s_min) / (s_max - s_min)
    else:
        sim_norm = np.zeros_like(sim_arr)
    ax.plot(sim_norm, label='Simulated Voltage (rescaled)', linewidth=2)

    # Min-max normalise real voltage
    if real_arr is not None:
        r_min, r_max = np.nanmin(real_arr), np.nanmax(real_arr)
        if np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
            real_norm = (real_arr - r_min) / (r_max - r_min)
        else:
            real_norm = np.zeros_like(real_arr)
        ax.plot(real_norm, label='Real Voltage (rescaled)',
                linestyle='--', linewidth=2, alpha=0.7)
    else:
        print("  Warning: real voltage column not found – plotting simulated only")

    ax.set_title("Voltage Comparison – Simulated vs Real (min-max normalised)")
    ax.set_xlabel("Time Steps (from first prediction)")
    ax.set_ylabel("Normalised Voltage (0–1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(output_folder, "voltage_comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Voltage plot saved to: {png_path}")


# ============================================================
# 9. CONSOLE TEE (writes to both stdout and a log file)
# ============================================================

class TeeWriter:
    """Duplicates writes to the original stream and a StringIO buffer."""
    def __init__(self, original, buffer):
        self.original = original
        self.buffer   = buffer
    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)
    def flush(self):
        self.original.flush()
        self.buffer.flush()


# ============================================================
# 10. ENTRY POINT
# ============================================================

def main():
    # ---- Paths (CLI args or interactive prompt) ----
    if len(sys.argv) >= 3:
        ckpt_path = sys.argv[1]
        csv_path  = sys.argv[2]
    else:
        ckpt_path = input("Path to full_checkpoint.pth: ").strip().strip('"')
        csv_path  = input("Path to test CSV file     : ").strip().strip('"')

    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found – {ckpt_path}")
        return
    if not os.path.isfile(csv_path):
        print(f"ERROR: test CSV not found – {csv_path}")
        return

    # ---- Create timestamped output folder inside OUTPUT_DIR ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = f"AI Integration_{timestamp}"
    out_folder  = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output folder: {out_folder}")

    # ---- Start console tee (captures everything to a log buffer) ----
    log_buf    = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = TeeWriter(old_stdout, log_buf)

    print(f"Output folder: {out_folder}")
    print(f"Decision threshold: {DECISION_THRESHOLD}")

    # ---- Load ----
    models, forward_fn, scaler, ckpt = load_checkpoint(ckpt_path)
    df, time_col = load_test_csv(csv_path)

    # ---- Run ----
    result = run_simulation(models, forward_fn, scaler, ckpt, df, time_col)
    if result is None:
        sys.stdout = old_stdout
        return
    step_records, all_t5, all_t6, all_t8, all_t9, all_cycle, warnings_count = result

    # ---- Summary ----
    print_summary(step_records, all_t5, all_t6, all_t8, all_t9, all_cycle, warnings_count)

    # ---- CSV export ----
    out_csv = os.path.join(out_folder, "virtual_test_results.csv")
    export_csv(step_records, out_csv)

    # ---- Voltage comparison plot ----
    seq_len = ckpt['hyperparameters']['sequence_length']
    plot_voltage_comparison(step_records, df, seq_len, out_folder)

    # ---- Save console log ----
    sys.stdout = old_stdout
    log_path = os.path.join(out_folder, "console_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_buf.getvalue())
    print(f"  Console log saved to: {log_path}")
    print(f"\nAll outputs saved in: {out_folder}")


if __name__ == "__main__":
    main()
