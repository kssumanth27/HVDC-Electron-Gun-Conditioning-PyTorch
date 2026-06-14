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
import io
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # <--- ADDED
import pandas as pd
import numpy as np
import matplotlib

# When running inside the multi-seed grid, force a non-interactive backend so
# figures save silently and never pop up. We also monkey-patch plt.show() to a
# no-op for the same reason. Set BEFORE importing pyplot.
_POLGUN_SINGLE_RUN = os.environ.get("POLGUN_SINGLE_RUN") == "1"
if _POLGUN_SINGLE_RUN:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

if _POLGUN_SINGLE_RUN:
    # Replace plt.show with a no-op. Figures still get saved via save_figure().
    plt.show = lambda *args, **kwargs: None

import glob
import json
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, matthews_corrcoef,
                             ConfusionMatrixDisplay, precision_recall_curve,
                             average_precision_score)

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import copy
import warnings
import random
import time
import platform


def set_all_seeds(seed):
    """Set all RNG seeds for reproducible weight initialization/training.

    Note: this does NOT control the train/val split seed, which is passed
    explicitly to ``prepare_all_data(..., split_seed=...)``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Default seed for ad-hoc single runs that don't go through the multi-seed
# entrypoint. The multi-seed runner overrides these per run.
_DEFAULT_WEIGHT_INIT_SEED = int(os.environ.get("POLGUN_WEIGHT_INIT_SEED", "545"))
set_all_seeds(_DEFAULT_WEIGHT_INIT_SEED)

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

def get_missing_mask(X_data, missing_value=0.0, sensor_channels=(0, 1, 2)):
    """
    Detects which samples already have missing (naturally absent) sensor data.
    
    Args:
        X_data: Tensor of shape [N, seq_len, n_channels]
        missing_value: The sentinel value used for missing data (default 0.0)
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
                         min_block_pct=0.2, max_block_pct=0.8, missing_value=0.0):
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


class TeeWriter:
    """Writes to both the original stdout and an io.StringIO buffer."""
    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer
    def write(self, msg):
        self.original.write(msg)
        self.buffer.write(msg)
    def flush(self):
        self.original.flush()


def save_text_report(output_folder, filename, content):
    """Save a string to a .txt file inside the output folder."""
    filepath = os.path.join(output_folder, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filepath}")


def save_test_probability_report(output_folder, models, forward_fn, data,
                                 test_ablations=None, threshold=0.5,
                                 filename="test_probabilities", include_clean=True):
    """
    Save per-sample probabilities for the clean test set and/or test ablations.

    The report includes one row per sample:
      sample index | true label | predicted probability | hard prediction

    Args:
        include_clean: if True, write the clean TEST SET section.
        test_ablations: optional dict of {name: (X, y)} ablation tensors. When
            provided, each is written as its own section.
        filename: output text-file stem (so the clean-only and ablation-only
            reports can be saved to different files at different times).

    If neither the clean section nor any ablations are requested, nothing is
    written.
    """
    first_model = next(iter(models.values()))
    device = next(first_model.parameters()).device
    sections = []

    def _append_section(section_name, X, y):
        X = X.to(device)
        y = y.to(device)
        with torch.inference_mode():
            logits = forward_fn(models, X)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            y_true = y.detach().cpu().numpy().ravel()
            preds = (probs >= threshold).astype(int)

        header = [
            "=" * 90,
            f"{section_name}",
            "=" * 90,
            f"Threshold: {threshold:.4f}",
            f"Samples: {len(y_true)}",
            "sample_idx\ttrue_label\tpred_probability\tpred_class",
        ]
        rows = [
            f"{idx}\t{int(y_val)}\t{prob:.8f}\t{int(pred)}"
            for idx, (y_val, prob, pred) in enumerate(zip(y_true, probs, preds))
        ]
        sections.append("\n".join(header + rows))

    if include_clean:
        _append_section("TEST SET - CLEAN", data['X_test'], data['y_test'])

    if test_ablations:
        for ablation_name, (X_ab, y_ab) in test_ablations.items():
            _append_section(f"TEST ABLATION - {ablation_name}", X_ab, y_ab)

    if not sections:
        return

    save_text_report(output_folder, filename, "\n\n".join(sections) + "\n")


# ==========================================
# 0.3 Validation PR / MCC threshold selection + decision artifacts
# ==========================================

def _safe_mcc(y_true, y_pred):
    """Matthews correlation coefficient, guarded against degenerate predictions."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if y_pred.sum() in (0, len(y_pred)):
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def compute_validation_pr_threshold(models, forward_fn, data, output_folder=None,
                                    criterion="max_mcc", default_threshold=0.5):
    """
    Build the validation PR curve, score every candidate threshold with F1 and MCC,
    and commit a single decision threshold t* using the chosen ``criterion``.

    Saves in ``output_folder``:
      - ``val_pr_curve_points.csv``        (threshold, precision, recall, f1, mcc, predicted_positive_count)
      - ``val_threshold_selection.json``   (chosen threshold, argmax_f1/mcc rows, AP, class counts)
      - ``figures/val_pr_curve.png``       (PR curve + precision/recall/F1/MCC vs threshold)

    Returns (chosen_threshold, selection_info).
    """
    X_val, y_val = data['X_val'], data['y_val']
    if len(X_val) == 0:
        print("Validation set is empty; cannot compute PR curve.")
        return float(default_threshold), {}

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_val)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    y_true = y_val.cpu().numpy().ravel().astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        print(f"Validation set has only one class (pos={n_pos}, neg={n_neg}); "
              f"falling back to {default_threshold}.")
        return float(default_threshold), {}

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    average_precision = float(average_precision_score(y_true, probs))

    n_t = len(thresholds)
    f1_arr = np.zeros(n_t, dtype=float)
    mcc_arr = np.zeros(n_t, dtype=float)
    pred_pos_count = np.zeros(n_t, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, t in enumerate(thresholds):
            y_pred = (probs >= t).astype(int)
            pred_pos_count[i] = int(y_pred.sum())
            f1_arr[i] = f1_score(y_true, y_pred, zero_division=0)
            mcc_arr[i] = _safe_mcc(y_true, y_pred)

    pr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:n_t],
        "recall": recall[:n_t],
        "f1": f1_arr,
        "mcc": mcc_arr,
        "predicted_positive_count": pred_pos_count,
    })
    if output_folder is not None:
        csv_path = os.path.join(output_folder, "val_pr_curve_points.csv")
        pr_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    idx_max_f1 = int(np.argmax(f1_arr)) if n_t > 0 else 0
    idx_max_mcc = int(np.argmax(mcc_arr)) if n_t > 0 else 0

    def _row(idx):
        return {
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_arr[idx]),
            "mcc": float(mcc_arr[idx]),
            "predicted_positive_count": int(pred_pos_count[idx]),
        }

    row_max_f1 = _row(idx_max_f1)
    row_max_mcc = _row(idx_max_mcc)

    y_pred_05 = (probs >= default_threshold).astype(int)
    metrics_at_default = {
        "threshold": float(default_threshold),
        "precision": float(precision_score(y_true, y_pred_05, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_05, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_05, zero_division=0)),
        "mcc": _safe_mcc(y_true, y_pred_05),
        "predicted_positive_count": int(y_pred_05.sum()),
    }

    crit = (criterion or "").lower().strip()
    if crit == "max_f1":
        chosen = row_max_f1
    elif crit == "fixed_0_5":
        chosen = metrics_at_default
    else:
        crit = "max_mcc"
        chosen = row_max_mcc
    chosen_threshold = float(chosen["threshold"])

    selection_info = {
        "criterion": crit,
        "chosen_threshold": chosen_threshold,
        "chosen_row": chosen,
        "argmax_f1": row_max_f1,
        "argmax_mcc": row_max_mcc,
        "metrics_at_0_5": metrics_at_default,
        "average_precision": average_precision,
        "n_validation_samples": int(len(probs)),
        "validation_class_counts": {"stable_0": n_neg, "increase_1": n_pos},
    }
    if output_folder is not None:
        json_path = os.path.join(output_folder, "val_threshold_selection.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(selection_info, f, indent=2)
        print(f"Saved: {json_path}")

    # --- PR curve + threshold sweep plot ---
    if output_folder is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax_pr = axes[0]
        ax_pr.plot(recall, precision, color="steelblue", linewidth=2,
                   label=f"PR curve (AP={average_precision:.4f})")
        ax_pr.scatter([row_max_f1["recall"]], [row_max_f1["precision"]],
                      color="darkorange", s=80, zorder=5,
                      label=f"argmax F1 (t={row_max_f1['threshold']:.3f}, F1={row_max_f1['f1']:.3f})")
        ax_pr.scatter([row_max_mcc["recall"]], [row_max_mcc["precision"]],
                      color="crimson", s=80, marker="D", zorder=5,
                      label=f"argmax MCC (t={row_max_mcc['threshold']:.3f}, MCC={row_max_mcc['mcc']:.3f})")
        ax_pr.scatter([metrics_at_default["recall"]], [metrics_at_default["precision"]],
                      color="black", s=60, marker="x", zorder=5,
                      label=f"t=0.5 (F1={metrics_at_default['f1']:.3f})")
        # Mark the committed t* with a vertical/horizontal cross-hair
        ax_pr.axhline(y=chosen["precision"], color="purple", linestyle=":",
                      linewidth=1, alpha=0.6)
        ax_pr.axvline(x=chosen["recall"], color="purple", linestyle=":",
                      linewidth=1, alpha=0.6,
                      label=f"committed t*={chosen_threshold:.3f}")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_xlim(0.0, 1.02)
        ax_pr.set_ylim(0.0, 1.02)
        ax_pr.set_title("Validation Precision-Recall Curve")
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(loc="lower left", fontsize=8)

        ax_t = axes[1]
        ax_t.plot(thresholds, precision[:n_t], color="steelblue", linewidth=1.5,
                  label="Precision", alpha=0.7)
        ax_t.plot(thresholds, recall[:n_t], color="green", linewidth=1.5,
                  label="Recall", alpha=0.7)
        ax_t.plot(thresholds, f1_arr, color="darkorange", linewidth=2.0, label="F1")
        ax_t.plot(thresholds, mcc_arr, color="crimson", linewidth=2.0, label="MCC")
        ax_t.axvline(default_threshold, color="black", linestyle=":",
                     linewidth=1, label="t=0.5")
        ax_t.axvline(row_max_f1["threshold"], color="darkorange", linestyle="--",
                     linewidth=1, alpha=0.7, label=f"argmax F1 (t={row_max_f1['threshold']:.3f})")
        ax_t.axvline(row_max_mcc["threshold"], color="crimson", linestyle="--",
                     linewidth=1, alpha=0.7, label=f"argmax MCC (t={row_max_mcc['threshold']:.3f})")
        ax_t.axvline(chosen_threshold, color="purple", linestyle="-",
                     linewidth=1.5, alpha=0.7, label=f"committed t* ({crit})")
        ax_t.set_xlabel("Threshold")
        ax_t.set_ylabel("Metric value")
        ax_t.set_xlim(0.0, 1.0)
        ax_t.set_ylim(min(-0.1, float(np.min(mcc_arr)) - 0.05) if n_t > 0 else -0.1, 1.02)
        ax_t.set_title(f"Validation Metrics vs Threshold  (committed: {crit}, t*={chosen_threshold:.4f})")
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(loc="lower center", fontsize=8, ncol=2)

        plt.tight_layout()
        save_figure(fig, output_folder, "val_pr_curve")
        plt.close(fig)

    print("\n" + "=" * 60)
    print("VALIDATION PR / THRESHOLD SUMMARY")
    print("=" * 60)
    print(f"  Average Precision (AP): {average_precision:.4f}")
    print(f"  argmax F1:  t={row_max_f1['threshold']:.4f}  F1={row_max_f1['f1']:.4f}  MCC={row_max_f1['mcc']:.4f}")
    print(f"  argmax MCC: t={row_max_mcc['threshold']:.4f}  F1={row_max_mcc['f1']:.4f}  MCC={row_max_mcc['mcc']:.4f}")
    print(f"  t=0.5:      F1={metrics_at_default['f1']:.4f}  MCC={metrics_at_default['mcc']:.4f}")
    print(f"  Committed criterion: {crit}  ->  t* = {chosen_threshold:.6f}")
    print("=" * 60 + "\n")

    return chosen_threshold, selection_info


def plot_mcc_curve(history, output_folder=None):
    """Standalone MCC (val) per-epoch plot. MCC range is [-1, 1]."""
    if 'val_mcc' not in history or len(history['val_mcc']) == 0:
        return
    epochs = range(1, len(history['val_mcc']) + 1)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['val_mcc'], label='Val MCC', color='crimson', linewidth=2)
    if 'val_f1' in history and len(history['val_f1']) == len(history['val_mcc']):
        plt.plot(epochs, history['val_f1'], label='Val F1', color='darkorange',
                 linewidth=2, linestyle='--')
    plt.axhline(y=0.0, color='grey', linestyle=':', linewidth=1, label='MCC = 0 (random)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC / F1')
    plt.title('Validation MCC (and F1) per Epoch')
    plt.ylim([-1.05, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_folder is not None:
        save_figure(fig, output_folder, "mcc_curve")
    plt.close(fig)


def _decision_type_array(y_true, y_pred):
    """Per-sample 'TN' / 'FP' / 'FN' / 'TP' labels."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    out = np.empty(len(y_true), dtype=object)
    out[(y_true == 0) & (y_pred == 0)] = "TN"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    return out


def save_decision_probabilities_csv(output_folder, filename, probs, y_true, threshold,
                                    section_name=""):
    """
    Save a per-sample CSV with true label, probability, predicted class, TN/TP/FP/FN
    label, threshold, margin (|prob - t|), and threshold-normalized confidence.
    """
    probs = np.asarray(probs).ravel().astype(float)
    y_true = np.asarray(y_true).ravel().astype(int)
    preds = (probs >= threshold).astype(int)
    decision_type = _decision_type_array(y_true, preds)
    is_correct = (preds == y_true)

    signed_margin = probs - threshold
    abs_margin = np.abs(signed_margin)
    signed_margin_pp = signed_margin * 100.0
    abs_margin_pp = abs_margin * 100.0
    t_clipped = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    norm_conf = np.where(
        probs >= threshold,
        (probs - t_clipped) / (1.0 - t_clipped),
        (t_clipped - probs) / t_clipped,
    )
    norm_conf = np.clip(norm_conf, 0.0, 1.0)

    df = pd.DataFrame({
        "sample_idx": np.arange(len(probs)),
        "true_label": y_true,
        "pred_probability": probs,
        "pred_class": preds,
        "decision_type": decision_type,
        "is_correct": is_correct,
        "decision_threshold": float(threshold),
        "signed_margin_from_threshold": signed_margin,
        "abs_margin_from_threshold": abs_margin,
        "signed_margin_percentage_points": signed_margin_pp,
        "abs_margin_percentage_points": abs_margin_pp,
        "normalized_decision_confidence": norm_conf,
        "section": section_name,
    })
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")
    return df


def plot_threshold_decision_boundary(output_folder, filename, probs, y_true,
                                     threshold, title_prefix=""):
    """
    Probability-distribution plots separated by TN, TP, FN, FP, with a vertical
    line at the chosen threshold. Saved as a single figure with four panels +
    an overall histogram.
    """
    probs = np.asarray(probs).ravel().astype(float)
    y_true = np.asarray(y_true).ravel().astype(int)
    preds = (probs >= threshold).astype(int)
    dtypes = _decision_type_array(y_true, preds)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    bins = np.linspace(0.0, 1.0, 41)
    type_colors = {"TN": "#3a7d44", "FP": "#d62828", "FN": "#f77f00", "TP": "#1d3557"}

    # Top-left: overall histogram with threshold line
    ax = axes[0, 0]
    ax.hist(probs, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"t* = {threshold:.4f}")
    ax.set_title("All Samples")
    ax.set_xlabel("Predicted P(increase)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-right: correct vs wrong
    ax = axes[0, 1]
    correct_mask = preds == y_true
    ax.hist(probs[correct_mask], bins=bins, color="#2a9d8f", edgecolor="black",
            alpha=0.7, label=f"Correct (n={int(correct_mask.sum())})")
    ax.hist(probs[~correct_mask], bins=bins, color="#e63946", edgecolor="black",
            alpha=0.7, label=f"Wrong (n={int((~correct_mask).sum())})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"t* = {threshold:.4f}")
    ax.set_title("Correct vs Wrong")
    ax.set_xlabel("Predicted P(increase)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-right-most: stacked decision-type histogram
    ax = axes[0, 2]
    bottom = np.zeros(len(bins) - 1)
    for dt in ["TN", "FP", "FN", "TP"]:
        mask = dtypes == dt
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(probs[mask], bins=bins)
        ax.bar(bins[:-1], counts, width=bins[1] - bins[0], align="edge",
               bottom=bottom, color=type_colors[dt], edgecolor="black",
               alpha=0.85, label=f"{dt} (n={int(mask.sum())})")
        bottom += counts
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"t* = {threshold:.4f}")
    ax.set_title("Stacked by Decision Type")
    ax.set_xlabel("Predicted P(increase)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom row, panels 0..2: TN, FP, FN
    for i, dt in enumerate(["TN", "FP", "FN"]):
        ax = axes[1, i]
        mask = dtypes == dt
        if mask.sum() > 0:
            ax.hist(probs[mask], bins=bins, color=type_colors[dt],
                    edgecolor="black", alpha=0.85)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2)
        ax.set_title(f"{dt} (n={int(mask.sum())})")
        ax.set_xlabel("Predicted P(increase)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # No dedicated bottom-row TP panel exists; the stacked plot in axes[0,2]
    # already shows TP. Add an FN-vs-TP overlay in a small inset on axes[1,2]
    # if we have positives, so users can inspect the positive-class margin.
    ax = axes[1, 2]
    if (dtypes == "TP").sum() > 0:
        # Overlay TP on top of FN (which is already plotted in this axis above).
        ax.hist(probs[dtypes == "TP"], bins=bins, color=type_colors["TP"],
                edgecolor="black", alpha=0.45,
                label=f"TP overlay (n={int((dtypes=='TP').sum())})")
        ax.set_title(f"FN (solid) + TP overlay")
        ax.legend(fontsize=8)

    fig.suptitle(f"{title_prefix} Threshold Decision Boundary (t*={threshold:.4f})",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_folder is not None:
        save_figure(fig, output_folder, filename)
    plt.close(fig)


def plot_relative_threshold_decision_boundary(output_folder, filename, probs, y_true,
                                             threshold, title_prefix=""):
    """
    Same decision-type view as plot_threshold_decision_boundary, but every sample
    is plotted as its OFFSET FROM THE THRESHOLD, not its raw probability.

    The x-axis is (P(increase) - threshold) in percentage points:
      - 0 on the x-axis = the threshold itself
      - +X  = the model's predicted probability is X percentage points above
              the threshold (and is therefore classified as the positive class)
      - -X  = the model's predicted probability is X percentage points below
              the threshold (and is therefore classified as the negative class)

    Example: with threshold t* = 0.89, a sample with predicted probability 0.90
    appears at +1.0 (i.e. 1 percentage point above the threshold).

    Every panel is annotated with the actual threshold value (e.g.
    ``Threshold t* = 0.8900 (89.00%)``) so the relative axis is unambiguous.
    """
    probs = np.asarray(probs).ravel().astype(float)
    y_true = np.asarray(y_true).ravel().astype(int)
    preds = (probs >= threshold).astype(int)
    dtypes = _decision_type_array(y_true, preds)
    margins_pp = (probs - threshold) * 100.0
    max_left = abs((0.0 - threshold) * 100.0)
    max_right = abs((1.0 - threshold) * 100.0)
    xlim = max(max_left, max_right)
    bins = np.linspace(-xlim, xlim, 51)
    type_colors = {"TN": "#3a7d44", "FP": "#d62828", "FN": "#f77f00", "TP": "#1d3557"}
    correct_mask = preds == y_true

    threshold_label = f"Threshold t* = {threshold:.4f} ({threshold * 100:.2f}%)"
    xlabel = (
        "Predicted probability MINUS threshold (percentage points)\n"
        f"0 = threshold ({threshold * 100:.2f}%)   |   "
        "negative = predicted class 0   |   positive = predicted class 1"
    )

    def _annotate_threshold(ax):
        """Draw the threshold line plus a small text label above it."""
        ax.axvline(0.0, color="red", linestyle="--", linewidth=2,
                   label=threshold_label)
        ymin, ymax = ax.get_ylim()
        ax.text(
            0.0, ymax * 0.97,
            f"  t* = {threshold:.4f}",
            color="red", fontsize=8, fontweight="bold",
            ha="left", va="top", rotation=0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="red", alpha=0.85),
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.hist(margins_pp, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_title("All Samples (relative to threshold)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _annotate_threshold(ax)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    ax.hist(margins_pp[correct_mask], bins=bins, color="#2a9d8f",
            edgecolor="black", alpha=0.7, label=f"Correct (n={int(correct_mask.sum())})")
    ax.hist(margins_pp[~correct_mask], bins=bins, color="#e63946",
            edgecolor="black", alpha=0.7, label=f"Wrong (n={int((~correct_mask).sum())})")
    ax.set_title("Correct vs Wrong (relative to threshold)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _annotate_threshold(ax)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 2]
    bottom = np.zeros(len(bins) - 1)
    for dt in ["TN", "FP", "FN", "TP"]:
        mask = dtypes == dt
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(margins_pp[mask], bins=bins)
        ax.bar(bins[:-1], counts, width=bins[1] - bins[0], align="edge",
               bottom=bottom, color=type_colors[dt], edgecolor="black",
               alpha=0.85, label=f"{dt} (n={int(mask.sum())})")
        bottom += counts
    ax.set_title("Stacked by Decision Type (relative to threshold)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _annotate_threshold(ax)
    ax.legend(fontsize=8, loc="upper right")

    for i, dt in enumerate(["TN", "FP", "FN"]):
        ax = axes[1, i]
        mask = dtypes == dt
        if mask.sum() > 0:
            ax.hist(margins_pp[mask], bins=bins, color=type_colors[dt],
                    edgecolor="black", alpha=0.85)
        ax.set_title(f"{dt} (n={int(mask.sum())}) — relative to threshold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        _annotate_threshold(ax)

    ax = axes[1, 2]
    if (dtypes == "TP").sum() > 0:
        ax.hist(margins_pp[dtypes == "TP"], bins=bins, color=type_colors["TP"],
                edgecolor="black", alpha=0.85)
    ax.set_title(f"TP (n={int((dtypes == 'TP').sum())}) — relative to threshold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _annotate_threshold(ax)

    fig.suptitle(
        f"{title_prefix} Decision Boundary — Relative to Threshold\n"
        f"x-axis = (predicted probability − threshold), in percentage points  |  "
        f"{threshold_label}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if output_folder is not None:
        save_figure(fig, output_folder, filename)
    plt.close(fig)


def evaluate_at_threshold(probs, y_true, threshold):
    """Compute accuracy, precision, recall, f1, mcc, pr_auc at a fixed threshold."""
    probs = np.asarray(probs).ravel().astype(float)
    y_true = np.asarray(y_true).ravel().astype(int)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "decision_threshold": float(threshold),
        "accuracy": float((preds == y_true).mean()) if len(y_true) else float("nan"),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "mcc": _safe_mcc(y_true, preds),
        "pr_auc": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }
    return metrics, probs, preds


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
# 1.5 Sensor Correlation Analysis
# ==========================================

def analyze_sensor_correlations(folder_path, sensor_cols=None, max_lag=50, output_folder=None):
    """
    Analyzes statistical correlations between sensor channels in the training data.
    
    Produces 3 analyses:
    1. Pearson + Spearman correlation matrix (global, all files concatenated)
    2. Per-file correlation distribution (stability across conditioning runs)
    3. Time-lagged cross-correlation (detects delayed relationships)
    
    Args:
        folder_path: Path to folder containing CSV files
        sensor_cols: List of sensor column names (defaults to the 3 standard sensors)
        max_lag: Maximum lag (in timesteps) for cross-correlation analysis
        output_folder: Optional folder to save figures
    
    Returns:
        results dict with keys: 'pearson', 'spearman', 'per_file_corrs', 'n_files',
        'xcorr' (dict of pair -> median cross-correlation array)
    """
    if sensor_cols is None:
        sensor_cols = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
    
    short_names = ["Current", "Pressure", "Radiation"]
    pair_names = ["Cur-Pres", "Cur-Rad", "Pres-Rad"]
    pair_indices = [(0, 1), (0, 2), (1, 2)]
    
    print(f"\n{'='*60}")
    print("SENSOR CORRELATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Scanning: {folder_path}")
    
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print("No CSV files found.")
        return {'n_files': 0, 'pearson': None, 'spearman': None,
                'per_file_corrs': None, 'xcorr': None}
    
    # Load only sensor columns from each file
    file_dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in sensor_cols)
            if len(df) > 0 and all(c in df.columns for c in sensor_cols):
                for c in sensor_cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                file_dfs.append(df)
        except Exception:
            pass
    
    n_files = len(file_dfs)
    print(f"Loaded {n_files} files with all 3 sensor columns.")
    
    if n_files == 0:
        print("No valid files. Skipping correlation analysis.")
        return {'n_files': 0, 'pearson': None, 'spearman': None,
                'per_file_corrs': None, 'xcorr': None}
    
    # ============================================
    # ANALYSIS 1: Global Pearson + Spearman Matrix
    # ============================================
    all_data = pd.concat(file_dfs, ignore_index=True)
    
    pearson_corr = all_data[sensor_cols].corr(method='pearson')
    spearman_corr = all_data[sensor_cols].corr(method='spearman')
    
    fig1, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    
    pearson_vals = pearson_corr.values
    im_p = ax_p.imshow(pearson_vals, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax_p.set_xticks(range(3))
    ax_p.set_yticks(range(3))
    ax_p.set_xticklabels(short_names, rotation=45, ha='right')
    ax_p.set_yticklabels(short_names)
    ax_p.set_title("Pearson Correlation")
    for i in range(3):
        for j in range(3):
            ax_p.text(j, i, f"{pearson_vals[i, j]:.3f}", ha='center', va='center',
                      fontsize=12, fontweight='bold')
    plt.colorbar(im_p, ax=ax_p, shrink=0.8)
    
    spearman_vals = spearman_corr.values
    im_s = ax_s.imshow(spearman_vals, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax_s.set_xticks(range(3))
    ax_s.set_yticks(range(3))
    ax_s.set_xticklabels(short_names, rotation=45, ha='right')
    ax_s.set_yticklabels(short_names)
    ax_s.set_title("Spearman Rank Correlation")
    for i in range(3):
        for j in range(3):
            ax_s.text(j, i, f"{spearman_vals[i, j]:.3f}", ha='center', va='center',
                      fontsize=12, fontweight='bold')
    plt.colorbar(im_s, ax=ax_s, shrink=0.8)
    
    plt.suptitle("Sensor Correlation Matrix (All Training Files)", fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig1, output_folder, "sensor_correlation_matrix")
    plt.show()
    
    # ============================================
    # ANALYSIS 2: Per-File Correlation Distribution
    # ============================================
    per_file_corrs = {pn: [] for pn in pair_names}
    
    for df in file_dfs:
        clean = df[sensor_cols].dropna()
        if len(clean) < 10:
            continue
        corr = clean.corr(method='pearson')
        for (i, j), pn in zip(pair_indices, pair_names):
            r = corr.iloc[i, j]
            if pd.notna(r):
                per_file_corrs[pn].append(r)
    
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, pn in zip(axes, pair_names):
        vals = per_file_corrs[pn]
        if vals:
            ax.hist(vals, bins=min(20, max(5, len(vals))), alpha=0.8, edgecolor='black')
            median_r = np.median(vals)
            ax.axvline(median_r, color='red', linestyle='--', linewidth=2,
                       label=f'median = {median_r:.3f}')
            ax.legend()
        ax.set_title(pn)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("# Files")
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Per-File Sensor Correlations (Stability Across Runs)", fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig2, output_folder, "sensor_correlation_per_file")
    plt.show()
    
    # ============================================
    # ANALYSIS 3: Time-Lagged Cross-Correlation
    # ============================================
    lags = np.arange(-max_lag, max_lag + 1)
    xcorr_results = {}
    
    for (i, j), pn in zip(pair_indices, pair_names):
        file_xcorrs = []
        col_a, col_b = sensor_cols[i], sensor_cols[j]
        
        for df in file_dfs:
            a = pd.to_numeric(df[col_a], errors='coerce').dropna().values
            b = pd.to_numeric(df[col_b], errors='coerce').dropna().values
            n = min(len(a), len(b))
            
            if n < 2 * max_lag + 10:
                continue
            
            a = a[:n]
            b = b[:n]
            a_std = a.std()
            b_std = b.std()
            if a_std < 1e-12 or b_std < 1e-12:
                continue
            
            a_norm = (a - a.mean()) / a_std
            b_norm = (b - b.mean()) / b_std
            
            ccf = np.zeros(len(lags))
            for k, lag in enumerate(lags):
                if lag >= 0:
                    ccf[k] = np.mean(a_norm[lag:n-max_lag] * b_norm[:n-max_lag-lag])
                else:
                    ccf[k] = np.mean(a_norm[:n-max_lag+lag] * b_norm[-lag:n-max_lag])
            
            file_xcorrs.append(ccf)
        
        if file_xcorrs:
            xcorr_results[pn] = np.median(np.array(file_xcorrs), axis=0)
        else:
            xcorr_results[pn] = np.zeros(len(lags))
    
    fig3, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, pn in zip(axes, pair_names):
        xcorr = xcorr_results[pn]
        ax.bar(lags, xcorr, width=1.0, alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        peak_lag = lags[np.argmax(np.abs(xcorr))]
        peak_val = xcorr[np.argmax(np.abs(xcorr))]
        ax.set_title(f"{pn}\npeak at lag={peak_lag} (r={peak_val:.3f})")
        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Cross-correlation")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Time-Lagged Cross-Correlation (Median Across Files)", fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig3, output_folder, "sensor_cross_correlation")
    plt.show()
    
    # ============================================
    # PRINT SUMMARY
    # ============================================
    print(f"\n{'-'*60}")
    print("CORRELATION SUMMARY")
    print(f"{'-'*60}")
    print(f"{'Pair':<12} | {'Pearson':>10} | {'Spearman':>10} | {'Strength':<15}")
    print(f"{'-'*60}")
    
    for (i, j), pn in zip(pair_indices, pair_names):
        pr = pearson_vals[i, j]
        sr = spearman_vals[i, j]
        abs_pr = abs(pr)
        if abs_pr > 0.7:
            strength = "STRONG"
        elif abs_pr > 0.4:
            strength = "MODERATE"
        elif abs_pr > 0.2:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"
        print(f"{pn:<12} | {pr:>+10.4f} | {sr:>+10.4f} | {strength:<15}")
    
    print(f"{'-'*60}")
    print(f"Files analyzed: {n_files}")
    print(f"Total rows: {len(all_data)}")
    print(f"{'='*60}\n")
    
    return {
        'n_files': n_files,
        'n_rows': len(all_data),
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'per_file_corrs': per_file_corrs,
        'xcorr': xcorr_results,
        'xcorr_lags': lags,
    }


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
            print(df.shape)
            # Store filename for later debug prints
            df.attrs['source_file'] = os.path.basename(filename)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
        
        print(len(dfs))
    return dfs


def process_single_df_to_sequences(df, scaler, sequence_length,
                                   noise_thresholds=None,
                                   filter_quiet_negatives=True,
                                   missing_value=0.0):
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
        # Keep NaN as NaN - will be replaced with 0 after normalization
        val = df[col].astype(float)
        input_data.append(val.values)
        
    # 3. Add Previous Output (A1) to Input Features
    a1_history = df['VoltageChange'].fillna(0).values
    input_data.append(a1_history)

    # Stack features: [Current, Pressure, Radiation, Prev_A1] = 4 channels
    X_raw = np.stack(input_data, axis=1)
    
    # 4. Apply Scaler (Only to sensor values - indices 0, 1, 2)
    # If scaler is None we return raw (unnormalized) sequences with NaN preserved.
    # The caller is expected to fit a scaler on the train split and then apply it
    # to both train and val (see prepare_all_data for the split-before-scaling path).
    if scaler is not None:
        # Store NaN mask before transform
        nan_mask = np.isnan(X_raw[:, [0, 1, 2]])

        # Replace NaN temporarily with 0 for transform
        X_sensor_temp = np.nan_to_num(X_raw[:, [0, 1, 2]], nan=0.0)
        X_raw[:, [0, 1, 2]] = scaler.transform(X_sensor_temp)

        # Restore NaN positions and replace with missing_value (missing indicator)
        for i in range(3):
            X_raw[nan_mask[:, i], i] = missing_value
    
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
                     filter_quiet_negatives=True,
                     missing_value=0.0,
                     split_seed=524):

    """
    WHAT: Orchestrates data loading, splitting, and scaling.

    Pipeline (no train/val leakage in normalization):
      1. Read train CSVs and process them into RAW (unnormalized) sequences --
         NaN preserved, no missing_value applied yet.
      2. Shuffled train_test_split (85/15) on the raw sequences using
         ``split_seed`` as the random_state.
      3. Fit a NaN-aware StandardScaler on the TRAIN sensor columns only.
      4. Apply that scaler to both train and val sensors, then replace NaN
         positions with ``missing_value``.
      5. Process test CSVs with the same train-only scaler.

    Calculates class weights for imbalanced data.

    Inputs: Train folder path, Test folder path, sequence length,
    Outputs: data dict (X_train, y_train, X_val, y_val, X_test, y_test),
             pos_weight, scaler (fit on TRAIN ONLY)

    X_* are torch FloatTensors of shape (Samples, Seq_Len, 4) (inputs for the model)
    y_* are torch FloatTensors of shape (Samples, 1) (Our final target: VoltageChange)
    """
    print(f"Loading Data...")

    # --- READ ALL TRAINING FILES ---
    train_dfs = read_folder_csvs(train_folder)
    test_dfs = read_folder_csvs(test_folder)

    print(f"Loaded {len(train_dfs)} training files")

    # --- PROCESS ALL TRAIN SEQUENCES (RAW, NO SCALING YET) ---
    # Pass scaler=None to keep NaN in place and skip normalization. We need raw
    # sensor values so we can split first and only then fit the scaler on the
    # training portion. Quiet-negative filtering operates on raw values and is
    # unaffected by scaling.
    print("Processing Train Sequences (raw, pre-split)...")
    X_train_list, y_train_list = [], []
    for df in train_dfs:
        x_s, y_s = process_single_df_to_sequences(
            df, scaler=None, sequence_length=sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=filter_quiet_negatives,
            missing_value=missing_value,
        )
        if len(x_s) > 0:
            X_train_list.append(x_s)
            y_train_list.append(y_s)

    if X_train_list:
        X_train_raw_all = np.concatenate(X_train_list)
        y_train_all = np.concatenate(y_train_list)
    else:
        X_train_raw_all = np.empty((0, sequence_length, 4))
        y_train_all = np.empty((0, 1))

    # --- SHUFFLED TRAIN/VAL SPLIT (BEFORE SCALING) ---
    print(f"Splitting into train/val with shuffled train_test_split "
          f"(85/15, random_state={split_seed})...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw_all, y_train_all,
        test_size=0.15,
        shuffle=True,
        random_state=int(split_seed),
    )

    # --- FIT NaN-AWARE SCALER ON TRAIN SENSORS ONLY ---
    # Sensor channels are 0/1/2; channel 3 (Prev_A1) is binary and must not be scaled.
    print("Fitting NaN-aware Scaler on TRAIN SPLIT ONLY (no val leakage)...")
    train_sensors_flat = X_train_raw[:, :, 0:3].reshape(-1, 3) if X_train_raw.size else np.empty((0, 3))
    scaler = StandardScaler()
    if train_sensors_flat.size > 0:
        scaler.mean_ = np.nanmean(train_sensors_flat, axis=0)
        scaler.scale_ = np.nanstd(train_sensors_flat, axis=0, ddof=0)
        # Guard against zero std (constant sensor) to avoid divide-by-zero
        scaler.scale_ = np.where(scaler.scale_ < 1e-12, 1.0, scaler.scale_)
    else:
        scaler.mean_ = np.zeros(3, dtype=float)
        scaler.scale_ = np.ones(3, dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 3
    scaler.n_samples_seen_ = np.sum(~np.isnan(train_sensors_flat), axis=0) \
        if train_sensors_flat.size > 0 else np.array([0, 0, 0])

    print(f"  Scaler mean (ignoring NaN, TRAIN ONLY): {scaler.mean_}")
    print(f"  Scaler std  (ignoring NaN, TRAIN ONLY): {scaler.scale_}")

    def _apply_scaler_to_seq(X_seq):
        """Apply the train-only scaler to sensor channels and replace NaN with missing_value."""
        if X_seq.size == 0:
            return X_seq
        X_out = X_seq.copy()
        flat = X_out[:, :, 0:3].reshape(-1, 3)
        nan_mask = np.isnan(flat)
        flat_clean = np.nan_to_num(flat, nan=0.0)
        flat_scaled = scaler.transform(flat_clean)
        flat_scaled[nan_mask] = missing_value
        X_out[:, :, 0:3] = flat_scaled.reshape(X_out.shape[0], X_out.shape[1], 3)
        return X_out

    X_train = _apply_scaler_to_seq(X_train_raw)
    X_val = _apply_scaler_to_seq(X_val_raw)

    # --- PROCESS TEST SEQUENCES (using train-only scaler) ---
    # Track which source CSV each test sequence came from so downstream code
    # can compute per-test-file metrics in addition to the pooled aggregate.
    print("Processing Test Sequences (using train-only scaler)...")
    X_test_list, y_test_list = [], []
    test_file_names = []
    test_sample_file_idx_list = []
    for src_idx, df in enumerate(test_dfs):
        x_s, y_s = process_single_df_to_sequences(
            df, scaler, sequence_length,
            noise_thresholds=noise_thresholds,
            filter_quiet_negatives=False,  # Don't filter test data
            missing_value=missing_value
        )
        src_name = df.attrs.get('source_file', f'test_file_{src_idx}')
        if len(x_s) > 0:
            X_test_list.append(x_s)
            y_test_list.append(y_s)
            file_index_for_this_source = len(test_file_names)
            test_file_names.append(src_name)
            test_sample_file_idx_list.append(
                np.full(x_s.shape[0], file_index_for_this_source, dtype=np.int64)
            )
        else:
            print(f"  [info] No test sequences generated from {src_name}")

    if X_test_list:
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
        test_sample_file_idx = np.concatenate(test_sample_file_idx_list)
    else:
        X_test = np.empty((0, sequence_length, 4))
        y_test = np.empty((0, 1))
        test_sample_file_idx = np.empty((0,), dtype=np.int64)

    # --- CLASS WEIGHTS ---
    # Logic: Dataset is imbalanced (mostly stable, few ramps).
    # We calculate Weight = Negatives / Positives to force model to learn the ramps.
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float)
    
    data = {
        'X_train': torch.FloatTensor(X_train), 'y_train': torch.FloatTensor(y_train),
        'X_val': torch.FloatTensor(X_val), 'y_val': torch.FloatTensor(y_val),
        'X_test': torch.FloatTensor(X_test), 'y_test': torch.FloatTensor(y_test),
        # Per-test-file metadata (parallel arrays):
        #   test_file_names      : list of unique source CSV filenames
        #   test_sample_file_idx : int array of length N_test, where each entry
        #                          indexes into test_file_names for that sample.
        'test_file_names': list(test_file_names),
        'test_sample_file_idx': test_sample_file_idx,
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
    print(f"Shuffled Split: 85% train / 15% val (random_state={split_seed})")
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
    fusion_channels: int = None,
    cnn_architecture: str = "mid_fusion_pointwise",
    branch2_filters: int = 128,
    fc_hidden: int = 500,
):
    """
    WHAT: Defines the Neural Network structure.
    ARCH: Conv1d -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear

    Output: All the layers in a dict for easy access.

    The default values for n_filters1, n_filters2, kernel_size, and pool_size
    are chosen to match the original architecture. They can be overridden
    by passing explicit values (e.g. from a hyperparameter dict).
    """
    
    if fusion_channels is None:
        fusion_channels = n_filters1

    if cnn_architecture == "mid_fusion_pointwise":
        if n_filters1 % input_dim != 0:
            raise ValueError(
                f"n_filters1={n_filters1} must be divisible by input_dim={input_dim} "
                "for grouped conv1 mid-fusion."
            )

        # Block 1: independent temporal filtering per input channel.
        print("yes midfusion pointwise")
        conv1 = nn.Conv1d(
            input_dim,
            n_filters1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_dim,
        )
        bn1 = nn.BatchNorm1d(n_filters1)

        # Block 1.5: pointwise fusion across channel-specific features.
        conv_fuse = nn.Conv1d(n_filters1, fusion_channels, kernel_size=1)
        bn_fuse = nn.BatchNorm1d(fusion_channels)

        # Block 2: temporal convolution after fusion.
        conv2 = nn.Conv1d(
            fusion_channels,
            n_filters2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        bn2 = nn.BatchNorm1d(n_filters2)
    elif cnn_architecture == "early_fusion":
        # Legacy architecture kept for loading older checkpoints.
        print("yes early fusion")
        conv1 = nn.Conv1d(input_dim, n_filters1, kernel_size=kernel_size, padding=kernel_size // 2)
        bn1 = nn.BatchNorm1d(n_filters1)
        conv_fuse = None
        bn_fuse = None
        conv2 = nn.Conv1d(n_filters1, n_filters2, kernel_size=kernel_size, padding=kernel_size // 2)
        bn2 = nn.BatchNorm1d(n_filters2)

    elif cnn_architecture == "split_fusion_3plus1":
        if input_dim != 4:
            raise ValueError(
                f"split_fusion_3plus1 requires exactly 4 input channels, got {input_dim}"
            )
        print("yes split_fusion_3plus1")
        pad = kernel_size // 2

        # Branch A — fused 3 sensors (channels 0-2)
        conv1_three = nn.Conv1d(3, n_filters1, kernel_size=kernel_size, padding=pad)
        bn1_three   = nn.BatchNorm1d(n_filters1)
        conv2_three = nn.Conv1d(n_filters1, branch2_filters, kernel_size=kernel_size, padding=pad)
        bn2_three   = nn.BatchNorm1d(branch2_filters)

        # Branch B — 4th channel alone
        conv1_one = nn.Conv1d(1, n_filters1, kernel_size=kernel_size, padding=pad)
        bn1_one   = nn.BatchNorm1d(n_filters1)
        conv2_one = nn.Conv1d(n_filters1, branch2_filters, kernel_size=kernel_size, padding=pad)
        bn2_one   = nn.BatchNorm1d(branch2_filters)

        # Merge: conv on concatenated branches (branch2_filters * 2 -> n_filters2)
        conv3 = nn.Conv1d(branch2_filters * 2, n_filters2, kernel_size=kernel_size, padding=pad)
        bn3   = nn.BatchNorm1d(n_filters2)

        # No pooling — time length stays seq_len throughout.
        flatten_size = n_filters2 * seq_len
        linear1 = nn.Linear(flatten_size, fc_hidden)
        linear2 = nn.Linear(fc_hidden, 1)

        return {
            'conv1_three': conv1_three, 'bn1_three': bn1_three,
            'conv2_three': conv2_three, 'bn2_three': bn2_three,
            'conv1_one': conv1_one,     'bn1_one': bn1_one,
            'conv2_one': conv2_one,     'bn2_one': bn2_one,
            'conv3': conv3,             'bn3': bn3,
            'linear1': linear1,         'linear2': linear2,
        }

    else:
        raise ValueError(f"Unknown cnn_architecture={cnn_architecture!r}")
    
    # --- Pooling + linear for mid_fusion_pointwise / early_fusion only ---
    pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
    
    len_after_pool1 = seq_len // pool_size
    len_after_pool2 = len_after_pool1 // pool_size
    
    linear_input_size = n_filters2 * len_after_pool2

    linear = nn.Linear(linear_input_size, 1)

    models = {
        'conv1': conv1,
        'bn1': bn1,
        'conv2': conv2,
        'bn2': bn2,
        'pool': pool,
        'linear': linear,
    }
    if conv_fuse is not None:
        models['conv_fuse'] = conv_fuse
        models['bn_fuse'] = bn_fuse
    return models


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

    # ---------- split_fusion_3plus1 path ----------
    if 'conv1_three' in models:
        # Branch A: first 3 channels (fused sensors)
        a = F.relu(models['bn1_three'](models['conv1_three'](x[:, :3, :])))
        a = F.relu(models['bn2_three'](models['conv2_three'](a)))

        # Branch B: 4th channel alone
        b = F.relu(models['bn1_one'](models['conv1_one'](x[:, 3:4, :])))
        b = F.relu(models['bn2_one'](models['conv2_one'](b)))

        # Concatenate along channel axis and fuse with conv3
        x = torch.cat([a, b], dim=1)
        x = F.relu(models['bn3'](models['conv3'](x)))

        # Two FC layers: flatten -> fc_hidden -> logit
        x = x.reshape(x.size(0), -1)
        x = F.relu(models['linear1'](x))
        return models['linear2'](x)

    # ---------- mid_fusion_pointwise / early_fusion path ----------
    x = models['conv1'](x)
    x = models['bn1'](x)
    x = F.relu(x)
    x = models['pool'](x)

    if 'conv_fuse' in models:
        x = models['conv_fuse'](x)
        x = models['bn_fuse'](x)
        x = F.relu(x)
    
    x = models['conv2'](x)
    x = models['bn2'](x)
    x = F.relu(x)
    x = models['pool'](x)
    
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

def _format_seconds(secs):
    """Format a duration in seconds as Hh Mm Ss / Mm Ss / Ss depending on size."""
    if secs is None or not np.isfinite(secs):
        return "n/a"
    secs = float(secs)
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs - 3600 * h - 60 * m
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    if m > 0:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


def _count_trainable_params(models):
    total = 0
    for layer in models.values():
        for p in layer.parameters():
            if p.requires_grad:
                total += p.numel()
    return total


def _get_model_device(models):
    """Return the actual torch.device used by the model parameters."""
    for layer in models.values():
        try:
            return next(layer.parameters()).device
        except StopIteration:
            continue
    return torch.device("cpu")


def _describe_device(actual_device=None):
    """Return a dict describing the actual training device + machine."""
    cuda_available = torch.cuda.is_available()
    if actual_device is None:
        actual_device = torch.device("cuda" if cuda_available else "cpu")
    actual_device = torch.device(actual_device)
    if actual_device.type == "cuda":
        device_name = torch.cuda.get_device_name(actual_device.index or torch.cuda.current_device())
    else:
        device_name = "CPU"
    return {
        "device_type": actual_device.type,
        "device_name": device_name,
        "actual_training_device": str(actual_device),
        "cuda_available": bool(cuda_available),
        "cuda_device_name_if_available": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if cuda_available else None
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def print_run_architecture_banner(model_type, models, hyperparams, n_train, n_val,
                                  n_test=None, input_dim=None, seq_len=None):
    """
    Print a single, clearly-labeled banner showing which model architecture is
    being used for this run plus the device and a parameter count. Mirrored
    into the run's saved test_metrics.json via build_training_info().
    """
    n_params = _count_trainable_params(models)
    dev = _describe_device(_get_model_device(models))
    print("\n" + "=" * 70)
    print("RUN ARCHITECTURE")
    print("=" * 70)
    print(f"  Model type            : {model_type.upper()}")
    if model_type == "lstm":
        print(f"  Variant               : LSTM (2-layer)")
        print(f"  hidden_dim            : {hyperparams.get('hidden_dim')}")
    else:
        print(f"  Variant               : {hyperparams.get('cnn_architecture')}")
        print(f"  n_filters1            : {hyperparams.get('n_filters1')}")
        print(f"  n_filters2            : {hyperparams.get('n_filters2')}")
        print(f"  fusion_channels       : {hyperparams.get('fusion_channels')}")
        print(f"  kernel_size           : {hyperparams.get('kernel_size')}")
        print(f"  pool_size             : {hyperparams.get('pool_size')}")
        if hyperparams.get('cnn_architecture') == 'split_fusion_3plus1':
            print(f"  branch2_filters       : {hyperparams.get('branch2_filters')}")
            print(f"  fc_hidden             : {hyperparams.get('fc_hidden')}")
    print(f"  input_dim x seq_len   : {input_dim} x {seq_len}")
    print(f"  trainable parameters  : {n_params:,}")
    print(f"  optimizer / loss      : Adam (lr={hyperparams.get('learning_rate')}) "
          f"/ BCEWithLogitsLoss")
    print(f"  batch_size / epochs   : {hyperparams.get('batch_size')} / "
          f"{hyperparams.get('epochs')}")
    print(f"  early_stop patience   : {hyperparams.get('early_stopping_patience')}")
    print(f"  threshold criterion   : {hyperparams.get('threshold_criterion')}")
    print(f"  train / val / test N  : {n_train} / {n_val} / "
          f"{n_test if n_test is not None else 'n/a'}")
    print("-" * 70)
    print(f"  actual training device: {dev['device_type']} ({dev['device_name']})")
    print(f"  CUDA available        : {dev['cuda_available']}"
          f"{' - ' + dev['cuda_device_name_if_available'] if dev['cuda_available'] else ''}")
    print(f"  torch / python        : {dev['torch_version']} / {dev['python_version']}")
    print(f"  platform              : {dev['platform']}")
    print(f"  processor / cpu_count : {dev['processor']} / {dev['cpu_count']}")
    print("=" * 70 + "\n")


def build_architecture_hyperparameter_report(
    models,
    model_type,
    hyperparams,
    data,
    pos_weight,
    weight_init_seed,
    split_seed,
    train_dir,
    test_dir,
    output_folder,
    history=None,
):
    """
    Build a plain-text report with the chosen architecture, full hyperparameters,
    layer/state_dict shapes, training device, seeds, and data sizes.
    """
    dev = _describe_device(_get_model_device(models))
    n_params = _count_trainable_params(models)
    history = history or {}
    training_info = history.get("training_info", {})

    lines = []
    lines.append("=" * 80)
    lines.append("ARCHITECTURE + HYPERPARAMETER REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Output folder: {output_folder}")
    lines.append("")

    lines.append("RUN IDENTIFIERS")
    lines.append("-" * 80)
    lines.append(f"weight_init_seed: {weight_init_seed}")
    lines.append(f"split_seed:       {split_seed}")
    lines.append(f"train_dir:        {train_dir}")
    lines.append(f"test_dir:         {test_dir}")
    lines.append("")

    lines.append("MODEL ARCHITECTURE SELECTED")
    lines.append("-" * 80)
    lines.append(f"model_type: {model_type}")
    if model_type == "lstm":
        lines.append("architecture_variant: LSTM (2-layer nn.LSTM + linear head)")
        lines.append(f"hidden_dim: {hyperparams.get('hidden_dim')}")
        lines.append("weight evidence: expected top-level model keys are ['lstm', 'linear']")
    else:
        lines.append(f"architecture_variant: {hyperparams.get('cnn_architecture')}")
        lines.append(f"n_filters1:      {hyperparams.get('n_filters1')}")
        lines.append(f"fusion_channels: {hyperparams.get('fusion_channels')}")
        lines.append(f"n_filters2:      {hyperparams.get('n_filters2')}")
        lines.append(f"kernel_size:     {hyperparams.get('kernel_size')}")
        lines.append(f"pool_size:       {hyperparams.get('pool_size')}")
        lines.append(f"branch2_filters: {hyperparams.get('branch2_filters')}")
        lines.append(f"fc_hidden:       {hyperparams.get('fc_hidden')}")
        lines.append("weight evidence: CNN runs have conv/bn layer keys; LSTM keys should be absent.")
    lines.append(f"trainable_parameters: {n_params:,}")
    lines.append(f"actual_training_device: {dev['device_type']} ({dev['device_name']})")
    lines.append(f"cuda_available: {dev['cuda_available']}")
    if dev["cuda_device_name_if_available"]:
        lines.append(f"cuda_device_name_if_available: {dev['cuda_device_name_if_available']}")
    lines.append(f"torch_version: {dev['torch_version']}")
    lines.append(f"python_version: {dev['python_version']}")
    lines.append(f"platform: {dev['platform']}")
    lines.append("")

    lines.append("DATASET SHAPES")
    lines.append("-" * 80)
    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        if key in data:
            lines.append(f"{key:8s}: {tuple(data[key].shape)}")
    lines.append(f"pos_weight: {float(pos_weight.item()) if hasattr(pos_weight, 'item') else pos_weight}")
    lines.append("")

    lines.append("TRAINING SUMMARY")
    lines.append("-" * 80)
    if training_info:
        for key in [
            "epochs_requested", "epochs_run", "best_epoch", "best_val_loss",
            "stopped_early", "early_stopping_patience",
            "total_training_human", "total_training_seconds",
            "mean_epoch_seconds", "std_epoch_seconds",
        ]:
            if key in training_info:
                lines.append(f"{key}: {training_info[key]}")
    else:
        lines.append("Training summary not available yet.")
    lines.append("")

    lines.append("MODEL LAYER OBJECTS")
    lines.append("-" * 80)
    lines.append(f"Top-level model keys: {list(models.keys())}")
    for layer_name, layer in models.items():
        layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        lines.append(f"{layer_name}: {layer.__class__.__name__} | trainable_params={layer_params:,}")
    lines.append("")

    lines.append("MODEL STATE_DICT / WEIGHT SHAPES")
    lines.append("-" * 80)
    for layer_name, layer in models.items():
        lines.append(f"[{layer_name}] {layer.__class__.__name__}")
        state = layer.state_dict()
        if not state:
            lines.append("  (no state_dict entries)")
            continue
        for param_name, tensor in state.items():
            lines.append(f"  {param_name:30s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    lines.append("")

    lines.append("ALL HYPERPARAMETERS")
    lines.append("-" * 80)
    for key in sorted(hyperparams.keys()):
        lines.append(f"{key}: {hyperparams[key]}")
    lines.append("")

    lines.append("HYPERPARAMETERS AS JSON")
    lines.append("-" * 80)
    lines.append(json.dumps(hyperparams, indent=2, default=str))
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    lines.append("")
    return "\n".join(lines)


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
    fusion_channels: int = None,
    cnn_architecture: str = "mid_fusion_pointwise",
    branch2_filters: int = 128,
    fc_hidden: int = 500,
    hidden_dim: int = 64,
    prob_full_missing: float = 0.2,
    prob_block_missing: float = 0.3,
    min_block_pct: float = 0.2,
    max_block_pct: float = 0.8,
    missing_value: float = 0.0,
    early_stopping_patience: int = 15,
):

    """
    Training loop for the model.
    Creates the models LSTM and CNN based on model_type.
    Uses Adam optimizer and BCEWithLogitsLoss with pos_weight for imbalance.

    NEW: Applies sensor dropout augmentation during training using apply_sensor_dropout().
    NEW: Early stopping on validation loss with configurable patience. After the
    loop ends (either naturally or via early stop), the model weights are
    reloaded from the epoch with the lowest val_loss so downstream evaluation,
    threshold selection, and checkpointing all use the best-epoch weights.

    Inputs: model_type ('cnn' or 'lstm'), data dict, pos_weight tensor, epochs, batch_size
            prob_full_missing: probability of full window dropout (20%)
            prob_block_missing: probability of block dropout (30%)
            Remaining 50%: no dropout
            early_stopping_patience: stop if val_loss has not improved for this
                many consecutive epochs. Set to 0 or None to disable.
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
            fusion_channels=fusion_channels,
            cnn_architecture=cnn_architecture,
            branch2_filters=branch2_filters,
            fc_hidden=fc_hidden,
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

    # --- RUN-LEVEL ARCHITECTURE BANNER (after model is built so param count is real) ---
    _banner_hyperparams = {
        "cnn_architecture": cnn_architecture,
        "n_filters1": n_filters1,
        "n_filters2": n_filters2,
        "fusion_channels": fusion_channels if fusion_channels is not None else n_filters1,
        "kernel_size": kernel_size,
        "pool_size": pool_size,
        "branch2_filters": branch2_filters,
        "fc_hidden": fc_hidden,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "threshold_criterion": None,
    }
    print_run_architecture_banner(
        model_type, models, _banner_hyperparams,
        n_train=int(len(X_train)),
        n_val=int(len(X_val)),
        n_test=None,
        input_dim=int(X_train.shape[2]) if X_train.ndim == 3 else None,
        seq_len=int(X_train.shape[1]) if X_train.ndim == 3 else None,
    )

    # LOSS FUNCTION:
    # BCEWithLogitsLoss combines Sigmoid + CrossEntropy. 
    # pos_weight scales the loss for the minority class (Ramps).
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_pr_auc': [],
        'val_f1': [], 'val_mcc': [],
        'epoch_times_sec': [],
    }

    # --- PER-EPOCH WEIGHT TRACKING ---
    weight_history = []  # List to store weight snapshots each epoch

    # --- TRAINING TIMER (wall-clock) ---
    _train_start_time = time.perf_counter()
    _train_start_wall = datetime.now().isoformat(timespec="seconds")

    # --- EARLY STOPPING STATE ---
    # Keep a snapshot of the best (lowest val_loss) state_dict so we can reload
    # it after training. epoch index is 0-based internally; we print 1-based.
    _use_early_stop = bool(early_stopping_patience and early_stopping_patience > 0)
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = copy.deepcopy({name: layer.state_dict() for name, layer in models.items()})
    if _use_early_stop:
        print(f">>> Early stopping enabled (patience={early_stopping_patience}, monitor=val_loss).")
    else:
        print(">>> Early stopping disabled.")

    for epoch in range(epochs):
        _epoch_start = time.perf_counter()

        # --- TRAIN LOOP (Mini-Batches) ---
        if model_type == 'lstm':
             models['lstm'].train() 
        else:
             for layer in models.values():
                 layer.train()
   
        
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
             for layer in models.values():
                 layer.eval()
        
        

        with torch.inference_mode():
            val_preds = forward_fn(models, X_val)
            v_loss = criterion(val_preds, y_val)
            pred_cls = (val_preds > 0).float()
            acc = (pred_cls == y_val).float().mean()

            val_probs = torch.sigmoid(val_preds).cpu().numpy().ravel()
            y_val_np = y_val.cpu().numpy().ravel().astype(int)
            val_preds_hard = pred_cls.cpu().numpy().ravel().astype(int)
            val_pr_auc = average_precision_score(y_val_np, val_probs)
            val_f1 = float(f1_score(y_val_np, val_preds_hard, zero_division=0))
            # matthews_corrcoef is undefined when one class is absent in preds; guard it.
            if val_preds_hard.sum() in (0, len(val_preds_hard)):
                val_mcc = 0.0
            else:
                val_mcc = float(matthews_corrcoef(y_val_np, val_preds_hard))

            history['val_loss'].append(v_loss.item())
            history['val_acc'].append(acc.item())
            history['val_pr_auc'].append(val_pr_auc)
            history['val_f1'].append(val_f1)
            history['val_mcc'].append(val_mcc)

        _epoch_elapsed = time.perf_counter() - _epoch_start
        history['epoch_times_sec'].append(float(_epoch_elapsed))

        # ETA: assume remaining epochs cost the running mean epoch time.
        _epochs_done = epoch + 1
        _epochs_left = max(epochs - _epochs_done, 0)
        _mean_epoch = float(np.mean(history['epoch_times_sec']))
        _eta_sec = _epochs_left * _mean_epoch
        _total_so_far = time.perf_counter() - _train_start_time

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f} | Val Loss {v_loss.item():.4f} | "
              f"Val Acc {acc.item():.4f} | Val PR-AUC {val_pr_auc:.4f} | "
              f"Val F1 {val_f1:.4f} | Val MCC {val_mcc:.4f} | "
              f"Epoch time {_format_seconds(_epoch_elapsed)} | "
              f"Elapsed {_format_seconds(_total_so_far)} | "
              f"ETA {_format_seconds(_eta_sec)}")

        # --- CAPTURE WEIGHTS THIS EPOCH ---
        epoch_snapshot = {'epoch': epoch}
        if 'conv1' in models:
            epoch_snapshot['conv1'] = copy.deepcopy(models['conv1'].state_dict())
        elif 'conv1_three' in models:
            epoch_snapshot['conv1_three'] = copy.deepcopy(models['conv1_three'].state_dict())
            epoch_snapshot['conv1_one'] = copy.deepcopy(models['conv1_one'].state_dict())
        else:
            epoch_snapshot['conv1'] = None
        weight_history.append(epoch_snapshot)

        # --- EARLY STOPPING CHECK (monitor: val_loss) ---
        current_val_loss = float(v_loss.item())
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch  # 0-based
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(
                {name: layer.state_dict() for name, layer in models.items()}
            )
        else:
            epochs_without_improvement += 1
            if _use_early_stop and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"\n>>> Early stopping triggered at epoch {epoch + 1}. "
                    f"No val_loss improvement for {early_stopping_patience} epochs."
                )
                break

    # --- RESTORE BEST-EPOCH WEIGHTS ---
    for name, layer in models.items():
        if name in best_state_dict:
            layer.load_state_dict(best_state_dict[name])
    print(
        f">>> Restored best weights from epoch {best_epoch + 1} "
        f"(val_loss={best_val_loss:.6f})."
    )

    # --- CAPTURE FINAL WEIGHTS (from best epoch) ---
    final_weights = {name: layer.state_dict() for name, layer in models.items()}
    print(">>> Final model weights correspond to the best validation-loss epoch.")
    print(f">>> Weight history captured for {len(weight_history)} epochs.")
    stopped_early = _use_early_stop and epochs_without_improvement >= early_stopping_patience

    # --- TIMING SUMMARY ---
    _total_train_sec = time.perf_counter() - _train_start_time
    _train_end_wall = datetime.now().isoformat(timespec="seconds")
    _epoch_times = history['epoch_times_sec']
    _mean_epoch_sec = float(np.mean(_epoch_times)) if _epoch_times else float('nan')
    _std_epoch_sec = float(np.std(_epoch_times, ddof=0)) if len(_epoch_times) > 1 else 0.0
    _device_info = _describe_device(_get_model_device(models))
    _n_params = _count_trainable_params(models)

    print(
        f">>> Training summary: best_epoch={best_epoch + 1}, "
        f"best_val_loss={best_val_loss:.6f}, "
        f"last_epoch_run={len(history['train_loss'])}, "
        f"stopped_early={stopped_early}"
    )
    print("\n" + "-" * 70)
    print("TRAINING WALL-CLOCK TIME")
    print("-" * 70)
    print(f"  Started   : {_train_start_wall}")
    print(f"  Finished  : {_train_end_wall}")
    print(f"  Total     : {_format_seconds(_total_train_sec)}  "
          f"({_total_train_sec:.2f} s)")
    print(f"  Mean/epoch: {_format_seconds(_mean_epoch_sec)}  "
          f"(std {_std_epoch_sec:.2f} s over {len(_epoch_times)} epochs)")
    print(f"  Actual training device: {_device_info['device_type']} "
          f"({_device_info['device_name']})")
    print(f"  CUDA available        : {_device_info['cuda_available']}"
          f"{' - ' + _device_info['cuda_device_name_if_available'] if _device_info['cuda_available'] else ''}")
    print(f"  Params    : {_n_params:,}")
    print("-" * 70 + "\n")

    # Stash a serializable training-info block so downstream code can persist it.
    history['training_info'] = {
        "model_type": model_type,
        "cnn_architecture": cnn_architecture if model_type != "lstm" else None,
        "n_trainable_params": int(_n_params),
        "epochs_requested": int(epochs),
        "epochs_run": int(len(history['train_loss'])),
        "best_epoch": int(best_epoch + 1),
        "best_val_loss": float(best_val_loss),
        "stopped_early": bool(stopped_early),
        "early_stopping_patience": int(early_stopping_patience or 0),
        "total_training_seconds": float(_total_train_sec),
        "total_training_human": _format_seconds(_total_train_sec),
        "mean_epoch_seconds": float(_mean_epoch_sec) if _epoch_times else None,
        "std_epoch_seconds": float(_std_epoch_sec) if _epoch_times else None,
        "epoch_times_seconds": [float(t) for t in _epoch_times],
        "train_start_iso": _train_start_wall,
        "train_end_iso": _train_end_wall,
        "device": _device_info,
        "input_dim": int(X_train.shape[2]) if X_train.ndim == 3 else None,
        "seq_len": int(X_train.shape[1]) if X_train.ndim == 3 else None,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }

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
            fusion_channels=hyperparams.get('fusion_channels'),
            cnn_architecture=hyperparams.get('cnn_architecture', 'early_fusion'),
            branch2_filters=hyperparams.get('branch2_filters', 128),
            fc_hidden=hyperparams.get('fc_hidden', 500),
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
    if model_type == 'cnn':
        print(f"    CNN architecture: {hyperparams.get('cnn_architecture', 'early_fusion')}")
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


def _conv1_weight_numpy(weight_tensor):
    if torch.is_tensor(weight_tensor):
        return weight_tensor.detach().cpu().numpy()
    return np.asarray(weight_tensor)


def get_conv1_channel_layout(weight_tensor, channel_names=None):
    """
    Map Conv1 weights to logical input channels for both architectures:
      - early fusion: weight shape [out_ch, n_channels, k]
      - grouped mid-fusion: weight shape [out_ch, 1, k], with filters grouped by channel
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    w = _conv1_weight_numpy(weight_tensor)
    n_channels = len(channel_names)
    out_ch, in_ch_per_filter, kernel_size = w.shape

    if in_ch_per_filter == n_channels:
        channel_blocks = [w[:, ch_idx, :] for ch_idx in range(n_channels)]
        heatmap = np.sum(np.abs(w), axis=2)
        return {
            "layout": "dense",
            "weight": w,
            "n_channels": n_channels,
            "out_channels": out_ch,
            "kernel_size": kernel_size,
            "filters_per_channel": out_ch,
            "channel_blocks": channel_blocks,
            "channel_vectors": [block.ravel() for block in channel_blocks],
            "filter_channel_heatmap": heatmap,
        }

    if in_ch_per_filter == 1 and out_ch % n_channels == 0:
        filters_per_channel = out_ch // n_channels
        channel_blocks = []
        heatmap = np.zeros((out_ch, n_channels), dtype=w.dtype)

        for ch_idx in range(n_channels):
            start = ch_idx * filters_per_channel
            end = start + filters_per_channel
            block = w[start:end, 0, :]
            channel_blocks.append(block)
            heatmap[start:end, ch_idx] = np.sum(np.abs(block), axis=1)

        return {
            "layout": "grouped",
            "weight": w,
            "n_channels": n_channels,
            "out_channels": out_ch,
            "kernel_size": kernel_size,
            "filters_per_channel": filters_per_channel,
            "channel_blocks": channel_blocks,
            "channel_vectors": [block.ravel() for block in channel_blocks],
            "filter_channel_heatmap": heatmap,
        }

    raise ValueError(
        f"Unsupported conv1 weight layout {w.shape} for {n_channels} logical channels."
    )

def plot_conv1_input_channel_hists(models, initial_weights=None, final_weights=None, bins=120):
    """
    If initial_weights/final_weights provided, overlays BEFORE vs AFTER in each plot.
    Otherwise plots current model weights only.
    """
    layout_after = get_conv1_channel_layout(get_conv1_weight_tensor(models), CHANNEL_NAMES)

    w_before = None
    if initial_weights is not None:
        w_before = get_conv1_channel_layout(
            get_conv1_weight_tensor({"conv1": initial_weights["conv1"]}),
            CHANNEL_NAMES,
        )

    # summary stats per channel
    print(f"Conv1 per-input-channel summary ({layout_after['layout']} layout):")
    for i, name in enumerate(CHANNEL_NAMES):
        v_after = layout_after["channel_vectors"][i]
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
        v_after = layout_after["channel_vectors"][i]
        plt.hist(v_after, bins=bins, alpha=0.6, label="after")

        if w_before is not None:
            v_before = w_before["channel_vectors"][i]
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

    if 'conv1' not in initial_weights:
        print("Skipping channel-importance analysis (no conv1 in weights).")
        return
    
    layout_init = get_conv1_channel_layout(initial_weights['conv1']['weight'], channel_names)
    layout_final = get_conv1_channel_layout(final_weights['conv1']['weight'], channel_names)

    n_channels = len(channel_names)
    
    # --- Compute per-channel metrics ---
    l2_init = np.zeros(n_channels)
    l2_final = np.zeros(n_channels)
    mean_abs_init = np.zeros(n_channels)
    mean_abs_final = np.zeros(n_channels)
    
    for i in range(n_channels):
        ch_init = layout_init["channel_vectors"][i]
        ch_final = layout_final["channel_vectors"][i]
        
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
    heatmap_init = layout_init["filter_channel_heatmap"]
    heatmap_final = layout_final["filter_channel_heatmap"]
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
        ch_init = layout_init["channel_vectors"][i]
        ch_final = layout_final["channel_vectors"][i]
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
        ch_init = layout_init["channel_vectors"][i]
        ch_final = layout_final["channel_vectors"][i]
        
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

    if 'conv1' not in weight_history[0] or weight_history[0].get('conv1') is None:
        print("Skipping channel-importance-over-epochs (no conv1 in weight_history; "
              "LSTM run?).")
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
            
        layout_epoch = get_conv1_channel_layout(snapshot['conv1']['weight'], channel_names)
        
        for ch_idx in range(n_channels):
            ch_weights = layout_epoch["channel_vectors"][ch_idx]
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

    if 'conv1' not in initial_weights:
        print("Skipping E/I analysis (no conv1 in weights).")
        return None
    
    layout_init = get_conv1_channel_layout(initial_weights['conv1']['weight'], channel_names)
    layout_final = get_conv1_channel_layout(final_weights['conv1']['weight'], channel_names)

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
        ch_init = layout_init["channel_vectors"][i]
        ch_final = layout_final["channel_vectors"][i]
        
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

    # Will hold a subsampled view of post-BN (pre-ReLU) activations for plotting
    z_bn_vals = None

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        X_dev = X.to(device)
        # Conv1 expects [N, C, T]
        x_perm = X_dev.permute(0, 2, 1)

        # --- Per-filter stats after BN + ReLU ---
        z = conv1(x_perm)
        z_bn = bn1(z)
        a = F.relu(z_bn)  # [N, F, L]

        # Cache post-BN, pre-ReLU activations for a global histogram
        z_bn_cpu = z_bn.detach().cpu().reshape(-1).numpy()
        if z_bn_cpu.size > 200000:
            idx = np.random.choice(z_bn_cpu.size, size=200000, replace=False)
            z_bn_cpu = z_bn_cpu[idx]
        z_bn_vals = z_bn_cpu

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
        weight = conv1.weight.detach()
        stride = conv1.stride[0]
        padding = conv1.padding[0]
        dilation = conv1.dilation[0]
        n_channels = min(len(channel_names), x_perm.shape[1])
        grouped_by_channel = (
            conv1.groups == conv1.in_channels == n_channels
            and weight.shape[1] == 1
            and conv1.out_channels % n_channels == 0
        )
        if not grouped_by_channel and weight.shape[1] != n_channels:
            raise ValueError(
                f"Unsupported Conv1 layout for activation analysis: weight shape {tuple(weight.shape)} "
                f"with logical n_channels={n_channels}."
            )
        filters_per_channel = conv1.out_channels // n_channels if grouped_by_channel else conv1.out_channels

        energy_per_channel = np.zeros(n_channels, dtype=np.float64)
        mean_abs_per_channel = np.zeros(n_channels, dtype=np.float64)
        std_abs_per_channel = np.zeros(n_channels, dtype=np.float64)
        # Activation-based excitation/inhibition metrics per channel (pre-BN)
        pos_sum_per_channel = np.zeros(n_channels, dtype=np.float64)
        neg_sum_per_channel = np.zeros(n_channels, dtype=np.float64)
        pos_count_per_channel = np.zeros(n_channels, dtype=np.float64)
        neg_count_per_channel = np.zeros(n_channels, dtype=np.float64)
        # For heatmap: mean |activation| per (filter, channel)
        heatmap_per_filter_channel = np.zeros((n_filters, n_channels), dtype=np.float64)
        # Split heatmaps: positive-only and |negative-only| contributions
        heatmap_pos_per_filter_channel = np.zeros((n_filters, n_channels), dtype=np.float64)
        heatmap_neg_per_filter_channel = np.zeros((n_filters, n_channels), dtype=np.float64)
        per_channel_contribs = []

        # Compute contributions channel by channel
        for ch_idx in range(n_channels):
            x_c = x_perm[:, ch_idx : ch_idx + 1, :]  # [N, 1, T]
            if grouped_by_channel:
                start = ch_idx * filters_per_channel
                end = start + filters_per_channel
                w_c = weight[start:end, :, :]
            else:
                start = 0
                end = n_filters
                w_c = weight[:, ch_idx : ch_idx + 1, :]
            contrib = F.conv1d(
                x_c,
                w_c,
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            per_channel_contribs.append(contrib)

            contrib_abs = contrib.abs()

            # Scalar summaries per channel (magnitude-based)
            energy_per_channel[ch_idx] = contrib.pow(2).sum().item()
            mean_abs_per_channel[ch_idx] = contrib_abs.mean().item()
            std_abs_per_channel[ch_idx] = contrib_abs.std(unbiased=False).item()

            # Excitation / inhibition metrics (sign-sensitive, pre-BN)
            pos = torch.clamp(contrib, min=0.0)
            neg = torch.clamp(contrib, max=0.0)
            pos_sum_per_channel[ch_idx] = pos.sum().item()
            neg_sum_per_channel[ch_idx] = neg.sum().item()  # negative or zero
            pos_count_per_channel[ch_idx] = (pos > 0).sum().item()
            neg_count_per_channel[ch_idx] = (neg < 0).sum().item()

            # Per-filter, per-channel mean |activation|
            contrib_abs_cpu = contrib_abs.detach().cpu()
            heatmap_per_filter_channel[start:end, ch_idx] = (
                contrib_abs_cpu.mean(dim=(0, 2)).numpy()
            )

            # Per-filter, per-channel split: positive-only and |negative-only|
            heatmap_pos_per_filter_channel[start:end, ch_idx] = (
                pos.detach().cpu().mean(dim=(0, 2)).numpy()
            )
            heatmap_neg_per_filter_channel[start:end, ch_idx] = (
                neg.abs().detach().cpu().mean(dim=(0, 2)).numpy()
            )

        total_energy = float(energy_per_channel.sum() + 1e-8)
        frac_energy_per_channel = 100.0 * energy_per_channel / total_energy

        # Activation-based E/I ratio per channel (pre-BN)
        ei_ratio_per_channel = pos_sum_per_channel / (np.abs(neg_sum_per_channel) + 1e-8)

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
    # Plot 0: Post-BN, pre-ReLU activation histogram (all filters/channels)
    # ============================
    if z_bn_vals is not None and z_bn_vals.size > 0:
        fig0, ax0 = plt.subplots(figsize=(8, 4))
        ax0.hist(z_bn_vals, bins=120, alpha=0.8, color="tab:purple")
        ax0.set_title("Conv1 Post-BN Pre-ReLU Activations (All Filters/Channels)")
        ax0.set_xlabel("Activation value (after BN, before ReLU)")
        ax0.set_ylabel("count")
        ax0.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_folder:
            save_figure(fig0, output_folder, "activation_post_bn_pre_relu_hist")
        plt.show()

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
    # Plot 2a: Per-channel activation histograms (clipped to 99th percentile)
    # ============================
    fig2a, axes2a = plt.subplots(2, 2, figsize=(10, 8))
    axes2a = axes2a.flatten()

    # We will also reuse the same vals for Plot 2b
    per_channel_vals = []

    for ch_idx in range(n_channels):
        contrib = per_channel_contribs[ch_idx]
        vals = contrib.abs().detach().cpu().reshape(-1).numpy()
        # Optional subsampling for very large arrays
        if vals.size > 200000:
            idx = np.random.choice(vals.size, size=200000, replace=False)
            vals = vals[idx]

        per_channel_vals.append(vals)

        ax = axes2a[ch_idx]
        # Clip x-axis to 99th percentile so the bulk of the distribution is visible
        if vals.size > 0:
            xlim = np.percentile(vals, 99)
            ax.hist(vals, bins=80, alpha=0.8, color="tab:blue")
            ax.set_xlim(0, xlim)
        else:
            ax.hist([], bins=80, alpha=0.8, color="tab:blue")

        ax.set_title(f"Channel {ch_idx}: {channel_names[ch_idx]} (clipped 99th pct)")
        ax.set_xlabel("|activation|")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (in case n_channels < 4)
    for j in range(n_channels, len(axes2a)):
        axes2a[j].axis("off")

    plt.suptitle("Conv1 Per-Channel Activation Distributions (Clipped)", fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig2a, output_folder, "activation_per_channel_histograms_clipped")
    plt.show()

    # ============================
    # Plot 2b: Per-channel activation histograms (log-log scale)
    # ============================
    fig2b, axes2b = plt.subplots(2, 2, figsize=(10, 8))
    axes2b = axes2b.flatten()

    for ch_idx in range(n_channels):
        vals = per_channel_vals[ch_idx]
        # Remove zeros before taking log scale on x
        vals_pos = vals[vals > 0]

        ax = axes2b[ch_idx]
        if vals_pos.size > 0:
            ax.hist(vals_pos, bins=80, alpha=0.8, color="tab:blue")
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.hist([], bins=80, alpha=0.8, color="tab:blue")

        ax.set_title(f"Channel {ch_idx}: {channel_names[ch_idx]} (log-log)")
        ax.set_xlabel("|activation| (log scale)")
        ax.set_ylabel("count (log scale)")
        ax.grid(True, which="both", alpha=0.3)

    for j in range(n_channels, len(axes2b)):
        axes2b[j].axis("off")

    plt.suptitle("Conv1 Per-Channel Activation Distributions (Log-Log)", fontsize=14)
    plt.tight_layout()
    if output_folder:
        save_figure(fig2b, output_folder, "activation_per_channel_histograms_log")
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
    # Plot 3c: Split positive/negative pre-BN heatmaps (side by side)
    # ============================
    vmax_split = max(heatmap_pos_per_filter_channel.max(),
                     heatmap_neg_per_filter_channel.max())

    fig3c, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 6))

    im_pos = ax_pos.imshow(
        heatmap_pos_per_filter_channel,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0, vmax=vmax_split,
    )
    ax_pos.set_xlabel("Input Channel")
    ax_pos.set_ylabel("Filter Index")
    ax_pos.set_xticks(np.arange(n_channels))
    ax_pos.set_xticklabels(channel_names, rotation=45, ha="right")
    ax_pos.set_title("Conv1 Mean Positive Activation\nper Filter-Channel (Excitatory)")
    plt.colorbar(im_pos, ax=ax_pos, shrink=0.7)

    im_neg = ax_neg.imshow(
        heatmap_neg_per_filter_channel,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=0, vmax=vmax_split,
    )
    ax_neg.set_xlabel("Input Channel")
    ax_neg.set_ylabel("Filter Index")
    ax_neg.set_xticks(np.arange(n_channels))
    ax_neg.set_xticklabels(channel_names, rotation=45, ha="right")
    ax_neg.set_title("Conv1 Mean |Negative| Activation\nper Filter-Channel (Inhibitory)")
    plt.colorbar(im_neg, ax=ax_neg, shrink=0.7)

    plt.tight_layout()
    if output_folder:
        save_figure(fig3c, output_folder, "activation_filter_channel_heatmap_pos_neg")
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
            f"std|a|={std_abs_per_channel[ch_idx]:.4e}  "
            f"sum_pos={pos_sum_per_channel[ch_idx]:.4e}  "
            f"sum_neg={neg_sum_per_channel[ch_idx]:.4e}  "
            f"E/I_ratio={ei_ratio_per_channel[ch_idx]:6.2f}"
        )

    print("\nPer-filter summary after BN + ReLU:")
    print(f"  Total filters: {n_filters}")
    print(f"  Dead filters (sparsity >= 90%): {n_dead}")
    print(f"  Weak filters (sparsity >= 70%): {n_weak}")
    print("=" * 80 + "\n")


def run_activation_sanity_checks(models, seq_len=30, channel_names=None, device="cpu"):
    """
    Run simple synthetic sanity checks for Conv1 activation statistics.

    These checks DO NOT update model weights. They only call
    analyze_activation_statistics() on crafted inputs to verify that:
      - Single-channel inputs produce energy and E/I concentrated on that channel
      - Positive vs negative inputs flip excitation/inhibition metrics
      - Scaling one channel increases its measured contribution

    Args:
        models: dict of layers (same format as used during training)
        seq_len: sequence length for synthetic tests
        channel_names: optional list of channel names
        device: 'cpu' or 'cuda' for running the tests
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    # Move model to the requested device (non-destructive for caller)
    for layer in models.values():
        if hasattr(layer, "to"):
            layer.to(device)

    print("\n" + "=" * 80)
    print("ACTIVATION SANITY CHECKS (SYNTHETIC INPUTS)")
    print("=" * 80)

    N = 32  # small batch size for tests
    n_channels = len(channel_names)

    # Helper to create a constant pattern tensor
    def make_constant_batch(value, channel_idx=None):
        X = torch.zeros(N, seq_len, n_channels, device=device)
        if channel_idx is None:
            X[:] = value
        else:
            X[:, :, channel_idx] = value
        return X

    # 1) Single-channel activation tests
    print("\n--- Sanity Test 1: Single-Channel Inputs ---")
    for ch_idx, name in enumerate(channel_names):
        print(f"\n[Single-Channel Test] Only channel {ch_idx} ({name}) is nonzero.")
        X_single = make_constant_batch(1.0, channel_idx=ch_idx)
        analyze_activation_statistics(
            models,
            X_single,
            channel_names=channel_names,
            output_folder=None,
        )

    # 2) Positive vs negative inputs for E/I behavior
    print("\n--- Sanity Test 2: Positive vs Negative Inputs (All Channels) ---")
    X_pos = make_constant_batch(1.0, channel_idx=None)
    X_neg = make_constant_batch(-1.0, channel_idx=None)

    print("\n[Positive Input] All channels = +1")
    analyze_activation_statistics(
        models,
        X_pos,
        channel_names=channel_names,
        output_folder=None,
    )

    print("\n[Negative Input] All channels = -1")
    analyze_activation_statistics(
        models,
        X_neg,
        channel_names=channel_names,
        output_folder=None,
    )

    # 3) Scaling one channel
    print("\n--- Sanity Test 3: Scaling One Channel (Current) ---")
    base = torch.randn(N, seq_len, n_channels, device=device) * 0.1
    X_base = base.clone()
    X_scaled = base.clone()
    # By convention, channel 0 is Current in CHANNEL_NAMES
    X_scaled[:, :, 0] *= 10.0

    print("\n[Base Input] Random small values on all channels")
    analyze_activation_statistics(
        models,
        X_base,
        channel_names=channel_names,
        output_folder=None,
    )

    print("\n[Scaled Input] Same as base, but 'Current' channel multiplied by 10")
    analyze_activation_statistics(
        models,
        X_scaled,
        channel_names=channel_names,
        output_folder=None,
    )

    print("\n" + "=" * 80)
    print("END OF ACTIVATION SANITY CHECKS")
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

    if 'conv1' not in weight_history[0] or weight_history[0].get('conv1') is None:
        print("Skipping E/I-over-epochs (no conv1 in weight_history; LSTM run?).")
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
            
        layout_epoch = get_conv1_channel_layout(snapshot['conv1']['weight'], channel_names)
        
        for ch_idx in range(n_channels):
            ch_weights = layout_epoch["channel_vectors"][ch_idx]
            
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

def channel_ablation_study(models, forward_fn, X_test, y_test,
                           channel_names=None, output_folder=None,
                           missing_value=0.0, threshold=0.5):
    """
    Ablate each channel using appropriate strategy and measure accuracy drop.
    
    Uses ACTUAL TEST DATA (X_test, y_test from your test CSV files).
    Runs on ALL samples (no filtering) since test data has natural missingness that
    varies per file. Reports missingness statistics so you know how many samples
    had the target channel already missing (redundant ablation) or other channels missing.
    
    For clean ablation metrics, use the robustness suite on validation data instead.
    
    4-channel architecture [Current, Pressure, Radiation, Prev_A1]
    Strategy by channel type:
    - Sensors (indices 0, 1, 2): Set to missing_value (0) to simulate missing sensor
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
    
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    
    # Set eval mode
    for layer in models.values():
        layer.eval()
    
    n_total = X_test.shape[0]
    
    # Compute baseline accuracy over all samples using the same committed
    # decision threshold used for test metrics/simulations.
    with torch.inference_mode():
        baseline_logits = forward_fn(models, X_test)
        baseline_probs = torch.sigmoid(baseline_logits)
        baseline_preds = (baseline_probs >= threshold).float()
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
    print(f"Decision threshold used for baseline + ablations: {threshold:.6f}")
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

            for layer in models.values():
                layer.eval()
            
            with torch.inference_mode():
                logits = forward_fn(models, X_ablated)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
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
                'decision_threshold': float(threshold),
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

            for layer in models.values():
                layer.eval()
            
            with torch.inference_mode():
                logits = forward_fn(models, X_ablated)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
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
                'decision_threshold': float(threshold),
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

            for layer in models.values():
                layer.eval()
            
            with torch.inference_mode():
                logits = forward_fn(models, X_ablated)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
                ablated_acc = (preds == y_test).float().mean().item()
            
            drop = baseline_acc - ablated_acc
            accuracy_drops.append(drop)
            importance = "HIGH" if drop > 0.05 else "MEDIUM" if drop > 0.01 else "LOW" if drop > 0 else "NONE/NEG"
            
            print(f"{ch_name:<16} | {method:<12} | {ablated_acc:>11.4f} | {drop:>+8.4f} | {importance:<10} | {n_total:>7} | {'N/A':>10} | {'N/A':>10} | {'N/A':>7}")
            
            results[ch_name] = {
                'baseline': baseline_acc,
                'ablated': ablated_acc,
                'drop': drop,
                'decision_threshold': float(threshold),
                'method': method,
                'importance': importance,
                'n_total': n_total,
                'n_target_already_missing': 0,
                'n_other_missing': 0,
                'n_fully_clean': n_total,
            }
    
    print("="*120)
    print("Note: Test data ablation uses ALL samples. Missingness columns are informational:")
    print("      N_tgt_miss = target channel was already 0 (ablation redundant for those samples)")
    print("      N_oth_miss = another sensor channel was already 0")
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

        for layer in models.values():
            layer.eval()
        
        with torch.inference_mode():
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

        for layer in models.values():
            layer.eval()

        with torch.inference_mode():
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

def evaluate_test_set(models, forward_fn, data, pos_weight, output_folder=None,
                      threshold=0.5):
    """
    Evaluate on the test set at a fixed decision ``threshold`` (default 0.5).
    Returns a dict with metrics + raw probabilities/predictions/true labels.
    """
    print("\n" + "="*30)
    print("EVALUATING ON TEST SET")
    print(f"Decision threshold: {threshold:.6f}")
    print("="*30)

    X_test, y_test = data['X_test'], data['y_test']

    if len(X_test) == 0:
        print("Test set is empty. Cannot evaluate.")
        return {}

    for layer in models.values():
        layer.eval()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.inference_mode():
        logits = forward_fn(models, X_test)
        loss = criterion(logits, y_test)
        test_loss = float(loss.item())

        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        y_true = y_test.cpu().numpy().ravel().astype(int)
        preds = (probs >= threshold).astype(int)

    accuracy = float((preds == y_true).mean())
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    mcc = _safe_mcc(y_true, preds)
    test_pr_auc = float(average_precision_score(y_true, probs))
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, probs)

    real_vals, real_counts = np.unique(y_true, return_counts=True)
    pred_vals, pred_counts = np.unique(preds, return_counts=True)
    real_dict = {int(k): int(v) for k, v in zip(real_vals, real_counts)}
    pred_dict = {int(k): int(v) for k, v in zip(pred_vals, pred_counts)}

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1:            {f1:.4f}")
    print(f"MCC:           {mcc:.4f}")
    print(f"PR-AUC:        {test_pr_auc:.4f}")
    print("-" * 30)
    print(f"Real Distribution:      {real_dict}")
    print(f"Predicted Distribution: {pred_dict}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(cm)
    print("="*30 + "\n")

    fig_pr = plt.figure(figsize=(7, 5))
    plt.step(pr_recall, pr_precision, where='post', color='blue', linewidth=2)
    plt.fill_between(pr_recall, pr_precision, step='post', alpha=0.15, color='blue')
    plt.axvline(rec, color='purple', linestyle=':', linewidth=1.2,
                label=f'committed t*={threshold:.4f}')
    plt.title(f'Test Precision-Recall Curve  (PR-AUC = {test_pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_folder:
        save_figure(fig_pr, output_folder, "precision_recall_curve")
    plt.close(fig_pr)

    return {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "decision_threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "pr_auc": test_pr_auc,
        "confusion_matrix": cm.tolist(),
        "real_distribution": real_dict,
        "predicted_distribution": pred_dict,
        "_test_probabilities": probs.tolist(),
        "_test_predictions": preds.tolist(),
        "_test_true_labels": y_true.tolist(),
    }


def _safe_filename_stem(name):
    """Sanitize a filename into a folder-safe stem.

    Drops the extension and replaces any character that isn't alphanumeric,
    underscore, hyphen, or dot with an underscore. Leading/trailing punctuation
    is stripped so we never produce an empty / hidden folder name.
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    safe_chars = []
    for ch in stem:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    cleaned = "".join(safe_chars).strip("._-")
    return cleaned if cleaned else "unnamed_file"


def evaluate_test_set_by_file(models, forward_fn, data, output_folder, threshold):
    """Compute the SAME test-set artifacts as the aggregate path, but per source CSV.

    For every unique test CSV under TEST_DIR (as recorded in
    ``data['test_file_names']``), this saves:
      - test_files/<safe_stem>/test_metrics.json
      - test_files/<safe_stem>/test_decision_probabilities.csv
      - test_files/<safe_stem>/figures/test_decision_boundary.png
      - test_files/<safe_stem>/figures/test_relative_decision_boundary.png

    It also writes top-level summary files at ``output_folder``:
      - test_file_metrics.csv
      - test_file_metrics.json

    The same ``threshold`` (validation-committed t*) is used for every file so
    metrics are directly comparable across files. No per-file thresholding is
    performed.

    Returns a list of per-file summary rows (also returned via the JSON file).
    """
    X_test = data.get('X_test')
    y_test = data.get('y_test')
    file_names = data.get('test_file_names', []) or []
    file_idx = data.get('test_sample_file_idx', None)

    if (X_test is None or len(X_test) == 0
            or not file_names or file_idx is None):
        print("Per-file test evaluation skipped: missing test data or "
              "source-file metadata.")
        return []

    print("\n" + "=" * 30)
    print(f"PER-FILE TEST EVALUATION ({len(file_names)} file(s))")
    print(f"Decision threshold: {threshold:.6f}")
    print("=" * 30)

    for layer in models.values():
        layer.eval()
    first_model = next(iter(models.values()))
    device = next(first_model.parameters()).device

    X_dev = X_test.to(device)
    with torch.inference_mode():
        logits = forward_fn(models, X_dev)
        probs_all = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y_all = y_test.detach().cpu().numpy().ravel().astype(int)
    file_idx_arr = np.asarray(file_idx).ravel().astype(int)

    per_file_root = os.path.join(output_folder, "test_files")
    os.makedirs(per_file_root, exist_ok=True)

    summary_rows = []
    summary_dict = {}

    for fi, fname in enumerate(file_names):
        mask = (file_idx_arr == fi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [skip] No test samples generated from {fname}")
            continue

        probs_f = probs_all[mask]
        y_f = y_all[mask]

        safe_stem = _safe_filename_stem(fname)
        file_folder = os.path.join(per_file_root, safe_stem)
        os.makedirs(os.path.join(file_folder, "figures"), exist_ok=True)

        metrics, _, _ = evaluate_at_threshold(probs_f, y_f, threshold)

        report = {
            "source_file": fname,
            "safe_folder_name": safe_stem,
            "decision_threshold": float(threshold),
            "metrics": metrics,
        }
        metrics_path = os.path.join(file_folder, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  [{fi+1}/{len(file_names)}] {fname}: "
              f"n={n}, acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, "
              f"mcc={metrics['mcc']:.4f}")
        print(f"      Saved: {metrics_path}")

        save_decision_probabilities_csv(
            file_folder, "test_decision_probabilities.csv",
            probs_f, y_f, threshold,
            section_name=f"TEST [{fname}]",
        )
        plot_threshold_decision_boundary(
            file_folder, "test_decision_boundary",
            probs_f, y_f, threshold,
            title_prefix=f"Test [{fname}]",
        )
        plot_relative_threshold_decision_boundary(
            file_folder, "test_relative_decision_boundary",
            probs_f, y_f, threshold,
            title_prefix=f"Test [{fname}]",
        )

        cm = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
        try:
            tn = int(cm[0][0]); fp = int(cm[0][1])
            fn_ = int(cm[1][0]); tp = int(cm[1][1])
        except Exception:
            tn = fp = fn_ = tp = 0

        pr_auc_val = metrics.get("pr_auc", float("nan"))
        try:
            pr_auc_clean = float(pr_auc_val)
        except (TypeError, ValueError):
            pr_auc_clean = float("nan")

        summary_row = {
            "source_file": fname,
            "safe_folder_name": safe_stem,
            "n_samples": int(metrics.get("n_samples", n)),
            "n_pos": int(metrics.get("n_pos", 0)),
            "n_neg": int(metrics.get("n_neg", 0)),
            "decision_threshold": float(threshold),
            "accuracy": float(metrics.get("accuracy", float("nan"))),
            "precision": float(metrics.get("precision", float("nan"))),
            "recall": float(metrics.get("recall", float("nan"))),
            "f1": float(metrics.get("f1", float("nan"))),
            "mcc": float(metrics.get("mcc", float("nan"))),
            "pr_auc": pr_auc_clean,
            "tn": tn,
            "fp": fp,
            "fn": fn_,
            "tp": tp,
            "output_subfolder": os.path.relpath(file_folder, output_folder),
        }
        summary_rows.append(summary_row)
        summary_dict[fname] = report

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_folder, "test_file_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        json_path = os.path.join(output_folder, "test_file_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "decision_threshold": float(threshold),
                "n_files_with_samples": len(summary_rows),
                "n_files_total": len(file_names),
                "summary_rows": summary_rows,
                "per_file_full": summary_dict,
            }, f, indent=2)
        print(f"Saved: {json_path}")
    else:
        print("No per-file rows produced; skipping top-level summary files.")

    return summary_rows


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
                             start_from_zero=True, elev=25, azim=35,
                             threshold=0.5):
    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

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
    ax.set_title(
        title + f"\n(Accuracy on this set: {(preds_np == y_np).mean():.3f}, "
        f"threshold={threshold:.4f})"
    )
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
                                  n_grid=60, band_frac=0.10, overlay="real",
                                  threshold=0.5):
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
    threshold:
      probability threshold used for predicted labels and boundary contours.
      The contour is drawn at logit(threshold), so threshold=0.5 is the usual
      logit=0 boundary.
    """
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    logit_threshold = float(np.log(threshold / (1.0 - threshold)))

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
        with torch.inference_mode():
            logits_all = forward_fn(models, X_ref)
            pred_all = (torch.sigmoid(logits_all) >= threshold).float().detach().cpu().numpy().flatten()

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

        with torch.inference_mode():
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

        # Decision boundary: probability = threshold, equivalently logit=logit(threshold)
        CS = plt.contour(cur_grid, pre_grid, logits.T, levels=[logit_threshold])
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
            f"Radiation ≈ {rad_fixed:.3f} (band ± {band:.3f})\n"
            f"{subtitle} | threshold={threshold:.4f}"
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
    Plots the Loss, Accuracy, and PR-AUC curves after training.
    Optionally saves to output_folder if provided.
    """
    epochs = range(1, len(history['val_loss']) + 1)
    has_pr_auc = 'val_pr_auc' in history and len(history['val_pr_auc']) > 0
    n_cols = 3 if has_pr_auc else 2
    fig = plt.figure(figsize=(6 * n_cols, 5))
    
    plt.subplot(1, n_cols, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val', linestyle='--')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, n_cols, 2)
    plt.plot(epochs, history['val_acc'], color='green', label='Val Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    if has_pr_auc:
        plt.subplot(1, n_cols, 3)
        plt.plot(epochs, history['val_pr_auc'], color='purple', label='Val PR-AUC')
        plt.title('Validation PR-AUC vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('PR-AUC (Average Precision)')
        plt.ylim([0.0, 1.05])
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if output_folder:
        save_figure(fig, output_folder, "training_curves")
    plt.show()


###########################################################################

#### Ignore from Line 1150 until Line 1330.
def build_sequences_with_indices_for_plot(df, scaler, sequence_length, missing_value=0.0):
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
                                         use_set="trainval", threshold=0.5):
    """
    Decision boundary slice with shaded class regions and scatter overlay.

    Shading meaning:
      - logit < 0  -> class 0 region
      - logit > 0  -> class 1 region

    overlay:
      "real" | "pred"
    use_set:
      "train" | "val" | "test" | "trainval"
    threshold:
      probability threshold used for predicted labels and shaded boundary.
    """
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    logit_threshold = float(np.log(threshold / (1.0 - threshold)))

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
        with torch.inference_mode():
            pred_probs = torch.sigmoid(forward_fn(models, X_ref))
            pred_all = (pred_probs >= threshold).float().cpu().numpy().flatten()

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

        with torch.inference_mode():
            logits = forward_fn(models, X_syn).cpu().numpy().reshape(n_grid, n_grid)

        # --- mask real points near this rad slice ---
        slice_mask = (rad_all >= (rad_fixed - band)) & (rad_all <= (rad_fixed + band))
        cur_slice = cur_all[slice_mask]
        pre_slice = pre_all[slice_mask]
        lab_slice = get_labels(slice_mask)

        plt.figure(figsize=(8, 6))

        # ---- SHADING ----
        # Two regions: probability < threshold (class 0), probability >= threshold (class 1)
        plt.contourf(cur_grid, pre_grid, logits.T,
                     levels=[-1e9, logit_threshold, 1e9],
                     colors=['green', 'grey'], alpha=0.18)

        # ---- BOUNDARY LINE ----
        CS = plt.contour(cur_grid, pre_grid, logits.T, levels=[logit_threshold])
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
            f"Radiation ≈ {rad_fixed:.3f} (band ± {band:.3f}) | "
            f"threshold={threshold:.4f}"
        )
        plt.xlabel("Current (real units)")
        plt.ylabel("Pressure (real units)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

##################################################################################################

def plot_test_file_trends(filepath, models, forward_fn, scaler, sequence_length,
                          output_folder=None, threshold=0.5):
    """
    6-row subplot:
    1) Real VoltageChange (0/1)
    2) Pred VoltageChange (0/1)  -- uses ``threshold`` on sigmoid probability
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
    X_seq, y_seq, idxs, df_clean = build_sequences_with_indices_for_plot(
        df, scaler, sequence_length, missing_value=HYPERPARAMS['missing_value']
    )

    if len(X_seq) == 0:
        print("File too short or no valid targets for plotting.")
        return

    X_tensor = torch.FloatTensor(X_seq)

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(float)

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

def test_single_file_simulation(filepath, models, forward_fn, scaler, sequence_length,
                                step_size, initial_A=0, max_limit_A=None,
                                missing_value=0.0, output_folder=None,
                                threshold=0.5):
    """
    Runs simulation on a single test file and plots Real vs Simulated voltage.
    Uses ``threshold`` (validation-committed t*) on sigmoid probability for
    decisions. Optionally saves the figure to output_folder if provided.
    """
    print(f"\n--- Processing Simulation: {filepath} ---")
    print(f"    Decision threshold: {threshold:.6f}")

    try:
        df = pd.read_csv(filepath)
        
    except FileNotFoundError:
        print("File not found.")
        return

    X_seq, y_seq = process_single_df_to_sequences(
        df, scaler, sequence_length, missing_value=missing_value
    )
    if len(X_seq) == 0:
        print("File too short or empty after processing.")
        return

    X_tensor = torch.FloatTensor(X_seq)

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (probs >= threshold).astype(float)

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


def test_single_file_simulation_with_ablation(filepath, models, forward_fn, scaler, sequence_length,
                                              step_size, ablation_name, missing_value=0.0,
                                              initial_A=0, max_limit_A=None, output_folder=None,
                                              threshold=0.5):
    """
    Runs simulation on a single test file under a synthetic multi-sensor missingness pattern.

    The underlying trained model is unchanged; only the input sensor channels are modified
    to reflect the requested ablation scenario before inference.

    Supported ablation_name values (cases 2–5):
      - 'missing_pres_rad'               -> Pressure, Radiation missing
      - 'missing_cur_rad'                -> Current, Radiation missing
      - 'missing_cur_pres'               -> Current, Pressure missing
      - 'missing_all_sensors_prev_only'  -> Current, Pressure, Radiation missing
    """
    print(f"\n--- Processing Ablation Simulation ({ablation_name}): {filepath} ---")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("File not found.")
        return

    # Process Input (same as baseline)
    X_seq, y_seq = process_single_df_to_sequences(df, scaler, sequence_length)
    if len(X_seq) == 0:
        print("File too short or empty after processing.")
        return

    # Apply synthetic missingness AFTER normalization, using the same sentinel as other ablations
    X_mod = np.array(X_seq, copy=True)
    if ablation_name == 'missing_pres_rad':
        X_mod[:, :, 1] = missing_value  # Pressure
        X_mod[:, :, 2] = missing_value  # Radiation
    elif ablation_name == 'missing_cur_rad':
        X_mod[:, :, 0] = missing_value  # Current
        X_mod[:, :, 2] = missing_value  # Radiation
    elif ablation_name == 'missing_cur_pres':
        X_mod[:, :, 0] = missing_value  # Current
        X_mod[:, :, 1] = missing_value  # Pressure
    elif ablation_name == 'missing_all_sensors_prev_only':
        X_mod[:, :, 0] = missing_value  # Current
        X_mod[:, :, 1] = missing_value  # Pressure
        X_mod[:, :, 2] = missing_value  # Radiation
    else:
        print(f"Unknown ablation name: {ablation_name}")
        return

    X_tensor = torch.FloatTensor(X_mod)

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (probs >= threshold).astype(float)

    simulated_A = []
    current_A = initial_A

    print(f"Ablation Sim Config -> Step: {step_size:.2f} | Limit: {max_limit_A} | "
          f"Ablation: {ablation_name} | Threshold: {threshold:.4f}")

    for raw_pred in predictions:
        decision = int(raw_pred)
        if decision == 1:
            current_A += step_size
        simulated_A.append(current_A)

    sim_arr = np.array(simulated_A, dtype=float)

    fig = plt.figure(figsize=(12, 6))

    real_norm = None

    # Real Voltage (if column exists) → normalize 0–1 (same logic as baseline)
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

    # Simulated Voltage → normalize 0–1 (own min/max)
    s_min = np.min(sim_arr)
    s_max = np.max(sim_arr)
    if s_max > s_min:
        sim_norm = (sim_arr - s_min) / (s_max - s_min)
    else:
        sim_norm = np.zeros_like(sim_arr)

    plt.plot(
        sim_norm,
        label=f'Simulated Voltage (normalized) - {ablation_name}',
        linewidth=2
    )

    plt.title(f"Simulation vs Real (Normalized): {os.path.basename(filepath)}\n"
              f"Step: {step_size:.2f} | Ablation: {ablation_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Voltage (0–1)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_folder:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_figure(fig, output_folder, f"simulation_{base_name}_{ablation_name}")
    plt.show()


def test_single_file_simulation_with_single_ablation(filepath, models, forward_fn, scaler, sequence_length,
                                                     step_size, ablation_name, missing_value=0.0,
                                                     initial_A=0, max_limit_A=None, output_folder=None,
                                                     threshold=0.5):
    """
    Runs simulation on a single test file under a single-sensor or Prev_A1 ablation.

    Supported ablation_name values (aligned with create_ablations):
      - 'current_full', 'pressure_full', 'radiation_full'
      - 'current_block', 'pressure_block', 'radiation_block'
      - 'prev_a1_shuffle', 'prev_a1_const_0', 'prev_a1_const_1', 'prev_a1_flip'
    """
    print(f"\n--- Processing Single-Channel Ablation Simulation ({ablation_name}): {filepath} ---")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("File not found.")
        return

    # Process Input (same as baseline)
    X_seq, y_seq = process_single_df_to_sequences(df, scaler, sequence_length)
    if len(X_seq) == 0:
        print("File too short or empty after processing.")
        return

    seq_len = X_seq.shape[1]
    block_start = seq_len // 4
    block_end = 3 * seq_len // 4  # middle 50%

    X_mod = np.array(X_seq, copy=True)

    # Sensor index mapping: 0=Current, 1=Pressure, 2=Radiation, 3=Prev_A1
    if ablation_name in ('current_full', 'pressure_full', 'radiation_full'):
        ch_map = {'current_full': 0, 'pressure_full': 1, 'radiation_full': 2}
        ch_idx = ch_map[ablation_name]
        X_mod[:, :, ch_idx] = missing_value
    elif ablation_name in ('current_block', 'pressure_block', 'radiation_block'):
        ch_map = {'current_block': 0, 'pressure_block': 1, 'radiation_block': 2}
        ch_idx = ch_map[ablation_name]
        X_mod[:, block_start:block_end, ch_idx] = missing_value
    elif ablation_name == 'prev_a1_const_0':
        X_mod[:, :, 3] = 0.0
    elif ablation_name == 'prev_a1_const_1':
        X_mod[:, :, 3] = 1.0
    elif ablation_name == 'prev_a1_flip':
        X_mod[:, :, 3] = 1.0 - X_mod[:, :, 3]
    elif ablation_name == 'prev_a1_shuffle':
        # Shuffle Prev_A1 across windows (batch dimension), analogous to create_ablations
        n_samples = X_mod.shape[0]
        perm = np.random.permutation(n_samples)
        X_mod[:, :, 3] = X_mod[perm, :, 3]
    else:
        print(f"Unknown single-ablation name: {ablation_name}")
        return

    X_tensor = torch.FloatTensor(X_mod)

    for layer in models.values():
        layer.eval()

    with torch.inference_mode():
        logits = forward_fn(models, X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (probs >= threshold).astype(float)

    simulated_A = []
    current_A = initial_A

    print(f"Single Ablation Sim Config -> Step: {step_size:.2f} | Limit: {max_limit_A} | "
          f"Ablation: {ablation_name} | Threshold: {threshold:.4f}")

    for raw_pred in predictions:
        decision = int(raw_pred)
        if decision == 1:
            current_A += step_size
        simulated_A.append(current_A)

    sim_arr = np.array(simulated_A, dtype=float)

    fig = plt.figure(figsize=(12, 6))

    real_norm = None

    # Real Voltage (if column exists) → normalize 0–1
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

    # Simulated Voltage → normalize 0–1 (own min/max)
    s_min = np.min(sim_arr)
    s_max = np.max(sim_arr)
    if s_max > s_min:
        sim_norm = (sim_arr - s_min) / (s_max - s_min)
    else:
        sim_norm = np.zeros_like(sim_arr)

    plt.plot(
        sim_norm,
        label=f'Simulated Voltage (normalized) - {ablation_name}',
        linewidth=2
    )

    plt.title(f"Simulation vs Real (Normalized): {os.path.basename(filepath)}\\n"
              f"Step: {step_size:.2f} | Ablation: {ablation_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Voltage (0–1)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_folder:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_figure(fig, output_folder, f"simulation_{base_name}_{ablation_name}")
    plt.show()


# ==========================================
# 6.5 Robustness Validation Suite
# ==========================================

def create_ablations(X_data, y_data, missing_value=0.0, random_seed=42):
    """
    Creates deterministic ablation variants of input data (validation or test).
    Also returns a missing_mask indicating which samples already have pre-existing
    missing sensor data, so downstream evaluation can filter appropriately.
    
    Args:
        X_data: Input tensor [N, seq_len, 4] - channels: [Current, Pressure, Radiation, Prev_A1]
        y_data: Labels tensor [N, 1]
        missing_value: Value to use for missing sensor data (default 0)
        random_seed: Seed for reproducible shuffling (uses LOCAL generator, doesn't affect global state)
    
    Returns:
        ablations: Dict with keys:
            Sensor ablations (channels 0, 1, 2):
            - 'clean': Original data
            - '{sensor}_full': Sensor channel fully missing (0)
            - '{sensor}_block': Sensor channel 50% block missing (middle)

            Multi-sensor missingness ablations (requested cases 2–5):
            - 'missing_pres_rad': Pressure & Radiation missing, only Current + Prev_A1 present
            - 'missing_cur_rad':  Current & Radiation missing, only Pressure + Prev_A1 present
            - 'missing_cur_pres': Current & Pressure missing, only Radiation + Prev_A1 present
            - 'missing_all_sensors_prev_only': Current, Pressure, Radiation missing, Prev_A1 only
            
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

    # --- MULTI-SENSOR MISSINGNESS ABLATIONS (cases 2–5) ---
    # Case 2: Pressure & Radiation missing → only Current + Prev_A1 present
    X_missing_pres_rad = X_data.clone()
    X_missing_pres_rad[:, :, 1] = missing_value  # Pressure
    X_missing_pres_rad[:, :, 2] = missing_value  # Radiation
    ablations['missing_pres_rad'] = (X_missing_pres_rad, y_data.clone())

    # Case 3: Current & Radiation missing → only Pressure + Prev_A1 present
    X_missing_cur_rad = X_data.clone()
    X_missing_cur_rad[:, :, 0] = missing_value  # Current
    X_missing_cur_rad[:, :, 2] = missing_value  # Radiation
    ablations['missing_cur_rad'] = (X_missing_cur_rad, y_data.clone())

    # Case 4: Current & Pressure missing → only Radiation + Prev_A1 present
    X_missing_cur_pres = X_data.clone()
    X_missing_cur_pres[:, :, 0] = missing_value  # Current
    X_missing_cur_pres[:, :, 1] = missing_value  # Pressure
    ablations['missing_cur_pres'] = (X_missing_cur_pres, y_data.clone())

    # Case 5: All three sensors missing → Prev_A1 only
    X_missing_all_sensors = X_data.clone()
    X_missing_all_sensors[:, :, 0] = missing_value  # Current
    X_missing_all_sensors[:, :, 1] = missing_value  # Pressure
    X_missing_all_sensors[:, :, 2] = missing_value  # Radiation
    ablations['missing_all_sensors_prev_only'] = (X_missing_all_sensors, y_data.clone())

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


def create_validation_ablations(X_val, y_val, missing_value=0.0):
    """
    Creates deterministic ablation variants of validation data.
    Wrapper around create_ablations for backward compatibility.
    
    Returns:
        (ablations, missing_mask)
    """
    return create_ablations(X_val, y_val, missing_value=missing_value, random_seed=42)


def create_test_ablations(X_test, y_test, missing_value=0.0):
    """
    Creates deterministic ablation variants of test data.
    Uses different seed than validation for independence.
    
    Returns:
        (ablations, missing_mask)
    """
    return create_ablations(X_test, y_test, missing_value=missing_value, random_seed=97)


def evaluate_robustness(models, forward_fn, ablations, missing_mask=None,
                        filter_missing=True, device='cpu', threshold=0.5):
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
    
    def _safe_pr_auc(y, probs):
        """Return PR-AUC if both classes present, else 0.0."""
        if len(np.unique(y)) < 2:
            return 0.0
        return average_precision_score(y, probs)

    for name, (X, y) in ablations.items():
        X, y = X.to(device), y.to(device)
        with torch.inference_mode():
            logits = forward_fn(models, X)
            probs_np = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (torch.sigmoid(logits) >= threshold).float()
            preds_np = preds.squeeze().cpu().numpy()
            y_np = y.squeeze().cpu().numpy()
        
        # Determine if this is a sensor or multi-sensor missingness ablation
        ablated_ch = None
        is_multi_missing = name.startswith('missing_')
        if missing_mask is not None and not is_multi_missing:
            for prefix, ch_idx in sensor_ablation_map.items():
                if name.startswith(prefix):
                    ablated_ch = ch_idx
                    break
        
        if (ablated_ch is not None or is_multi_missing) and missing_mask is not None:
            # Compute clean mask
            if is_multi_missing:
                # Multi-sensor ablations: require ALL three sensors present originally
                any_issue = missing_mask.any(dim=1).cpu().numpy()
            else:
                # Single-sensor ablation: exclude target-already-missing AND other-missing
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
                pr_auc = _safe_pr_auc(y_np[clean_idx], probs_np[clean_idx])
                
                if n_excluded > 0:
                    multi_acc = (preds_np[any_issue] == y_np[any_issue]).mean()
                else:
                    multi_acc = None
            elif filter_missing and n_clean == 0:
                acc, prec, rec, pr_auc = 0.0, 0.0, 0.0, 0.0
                multi_acc = (preds_np == y_np).mean()
            else:
                # TEST MODE: use ALL samples, just report stats
                acc = (preds_np == y_np).mean()
                prec = precision_score(y_np, preds_np, zero_division=0)
                rec = recall_score(y_np, preds_np, zero_division=0)
                pr_auc = _safe_pr_auc(y_np, probs_np)
                
                if n_excluded > 0:
                    multi_acc = (preds_np[any_issue] == y_np[any_issue]).mean()
                else:
                    multi_acc = None
            
            results[name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'pr_auc': pr_auc,
                'decision_threshold': float(threshold),
                'n_clean': n_clean, 'n_multi_missing': n_excluded,
                'multi_missing_acc': multi_acc
            }
        else:
            # 'clean' variant, prev_a1 ablations, or no mask: use ALL samples
            n_total = len(y_np)
            acc = (preds_np == y_np).mean()
            prec = precision_score(y_np, preds_np, zero_division=0)
            rec = recall_score(y_np, preds_np, zero_division=0)
            pr_auc = _safe_pr_auc(y_np, probs_np)
            
            results[name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'pr_auc': pr_auc,
                'decision_threshold': float(threshold),
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
    
    f.write(f"{'Ablation Type':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'PR-AUC':>10} {'Acc Delta':>12} {'N_clean':>8} {'N_multi':>8} {'Multi_Acc':>10}\n")
    f.write("-" * 110 + "\n")
    
    clean_acc = robustness_results.get('clean', {}).get('accuracy', 0)
    
    for name, metrics in robustness_results.items():
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        pr_auc = metrics.get('pr_auc', None)
        delta = acc - clean_acc if name != 'clean' else 0.0
        delta_str = f"{delta:+.4f}" if name != 'clean' else "---"
        pr_auc_str = f"{pr_auc:>10.4f}" if pr_auc is not None else f"{'N/A':>10}"
        
        n_clean = metrics.get('n_clean', '---')
        n_multi = metrics.get('n_multi_missing', 0)
        multi_acc = metrics.get('multi_missing_acc', None)
        
        n_clean_str = f"{n_clean:>8}" if isinstance(n_clean, int) else f"{'---':>8}"
        n_multi_str = f"{n_multi:>8}" if n_multi > 0 else f"{'---':>8}"
        multi_acc_str = f"{multi_acc:>10.4f}" if multi_acc is not None else f"{'N/A':>10}"
        
        f.write(f"{name:<20} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {pr_auc_str} {delta_str:>12} {n_clean_str} {n_multi_str} {multi_acc_str}\n")
    
    f.write("\n" + "-" * 100 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 100 + "\n")
    
    clean = robustness_results.get('clean', {})
    f.write(f"Clean Baseline:  Acc={clean.get('accuracy', 0):.4f}  "
            f"Prec={clean.get('precision', 0):.4f}  Rec={clean.get('recall', 0):.4f}  "
            f"PR-AUC={clean.get('pr_auc', 0):.4f}\n\n")
    
    def _avg_pr_auc(metrics_dict):
        vals = [v['pr_auc'] for v in metrics_dict.values() if 'pr_auc' in v]
        return np.mean(vals) if vals else None

    # Sensor full-missing averages (using clean-subset metrics)
    full_metrics = {k: v for k, v in robustness_results.items() if '_full' in k}
    if full_metrics:
        avg_acc = np.mean([v['accuracy'] for v in full_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in full_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in full_metrics.values()])
        avg_prauc = _avg_pr_auc(full_metrics)
        prauc_str = f"  PR-AUC={avg_prauc:.4f}" if avg_prauc is not None else ""
        f.write(f"Sensor Full-Missing Avg (clean subset):  Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}{prauc_str}\n")
    
    # Sensor block-missing averages
    block_metrics = {k: v for k, v in robustness_results.items() if '_block' in k}
    if block_metrics:
        avg_acc = np.mean([v['accuracy'] for v in block_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in block_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in block_metrics.values()])
        avg_prauc = _avg_pr_auc(block_metrics)
        prauc_str = f"  PR-AUC={avg_prauc:.4f}" if avg_prauc is not None else ""
        f.write(f"Sensor Block-Missing Avg (clean subset): Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}{prauc_str}\n")
    
    # Multi-sensor missingness averages (clean subset: cases 2–5)
    multi_keys = [
        'missing_pres_rad',
        'missing_cur_rad',
        'missing_cur_pres',
        'missing_all_sensors_prev_only',
    ]
    multi_metrics = {k: v for k, v in robustness_results.items() if k in multi_keys}
    if multi_metrics:
        avg_acc = np.mean([v['accuracy'] for v in multi_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in multi_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in multi_metrics.values()])
        avg_prauc = _avg_pr_auc(multi_metrics)
        prauc_str = f"  PR-AUC={avg_prauc:.4f}" if avg_prauc is not None else ""
        f.write(f"Multi-Sensor Missing Avg (clean subset): Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}{prauc_str}\n")
    
    # Prev_A1 ablation averages (all samples, not filtered)
    prev_a1_metrics = {k: v for k, v in robustness_results.items() if 'prev_a1' in k and not k in multi_keys}
    if prev_a1_metrics:
        avg_acc = np.mean([v['accuracy'] for v in prev_a1_metrics.values()])
        avg_prec = np.mean([v['precision'] for v in prev_a1_metrics.values()])
        avg_rec = np.mean([v['recall'] for v in prev_a1_metrics.values()])
        avg_prauc = _avg_pr_auc(prev_a1_metrics)
        prauc_str = f"  PR-AUC={avg_prauc:.4f}" if avg_prauc is not None else ""
        f.write(f"Prev_A1 Ablation Avg (all samples):      Acc={avg_acc:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}{prauc_str}\n")
    
    f.write("\nNote: Sensor ablation metrics computed on CLEAN subset only (no other sensor pre-missing).\n")
    f.write("      Prev_A1 ablations use ALL samples. Multi_Acc shows accuracy on excluded samples.\n")
    f.write("      Multi-sensor 'missing_*' ablations are evaluated on sequences where all three sensors were originally present.\n")

    # Optional legend for multi-sensor ablations (only print if present)
    multi_labels = {
        'missing_pres_rad': "Missing: Pressure, Radiation; Present: Current, Prev_A1",
        'missing_cur_rad': "Missing: Current, Radiation; Present: Pressure, Prev_A1",
        'missing_cur_pres': "Missing: Current, Pressure; Present: Radiation, Prev_A1",
        'missing_all_sensors_prev_only': "Missing: Current, Pressure, Radiation; Present: Prev_A1 only",
    }
    present_multi = [k for k in multi_keys if k in robustness_results]
    if present_multi:
        f.write("\nMulti-sensor ablation legend:\n")
        for k in present_multi:
            f.write(f"  - {k:<32} -> {multi_labels.get(k, '')}\n")

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


def plot_multi_missing_ablations(robustness_results, title_suffix="", output_folder=None, filename=None):
    """
    Focused bar chart for the four multi-sensor missingness ablations (cases 2–5).

    Shows drop = clean_accuracy - ablated_accuracy for:
      - 'missing_pres_rad'
      - 'missing_cur_rad'
      - 'missing_cur_pres'
      - 'missing_all_sensors_prev_only'
    """
    clean_acc = robustness_results.get('clean', {}).get('accuracy', 0)

    multi_keys = [
        'missing_pres_rad',
        'missing_cur_rad',
        'missing_cur_pres',
        'missing_all_sensors_prev_only',
    ]
    # Keep only keys that are present in results
    ablation_names = [k for k in multi_keys if k in robustness_results]
    if not ablation_names:
        return

    drops = [clean_acc - robustness_results[n]['accuracy'] for n in ablation_names]
    n_cleans = [robustness_results[n].get('n_clean', None) for n in ablation_names]
    n_excluded = [robustness_results[n].get('n_multi_missing', 0) for n in ablation_names]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(ablation_names))
    colors = ['green' if d > 0.05 else 'orange' if d > 0.01 else 'gray' for d in drops]

    bars = ax.bar(x, drops, color=colors, alpha=0.8, edgecolor='black')

    for bar, nc, ne in zip(bars, n_cleans, n_excluded):
        if nc is not None:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                    f'n={nc}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if ne and ne > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.012,
                    f'(excl {ne})', ha='center', va='bottom', fontsize=7, color='red')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High importance (5%)')
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium importance (1%)')

    ax.set_xlabel('Multi-Sensor Ablation', fontsize=12)
    ax.set_ylabel('Accuracy Drop (Clean - Ablated)', fontsize=12)
    ax.set_title(f'Multi-Sensor Missingness: Accuracy Drop ({title_suffix})\n'
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
        if 'conv1_three' in models:
            results['conv1_three_expects_3_channels'] = models['conv1_three'].in_channels == 3
            results['conv1_one_expects_1_channel'] = models['conv1_one'].in_channels == 1
        else:
            results['conv1_expects_4_channels'] = models['conv1'].in_channels == 4
            results['conv1_grouped_by_channel'] = models['conv1'].groups == models['conv1'].in_channels
            results['conv_fuse_present'] = 'conv_fuse' in models
            results['conv_fuse_is_pointwise'] = (
                'conv_fuse' in models and models['conv_fuse'].kernel_size == (1,)
            )
    else:
        results['lstm_expects_4_channels'] = models['lstm'].input_size == 4
    
    return results


def test_dropout_function(X_sample, n_iters=500, prob_full=0.2, prob_block=0.3, missing_value=0.0):
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


def test_ablation_suite(ablations, X_data, missing_value=0.0):
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

    # Check multi-sensor missingness ablations (cases 2–5)
    X_mpr = ablations['missing_pres_rad'][0]
    results['missing_pres_rad_pres_missing'] = (X_mpr[:, :, 1] == missing_value).all().item()
    results['missing_pres_rad_rad_missing'] = (X_mpr[:, :, 2] == missing_value).all().item()

    X_mcr = ablations['missing_cur_rad'][0]
    results['missing_cur_rad_cur_missing'] = (X_mcr[:, :, 0] == missing_value).all().item()
    results['missing_cur_rad_rad_missing'] = (X_mcr[:, :, 2] == missing_value).all().item()

    X_mcp = ablations['missing_cur_pres'][0]
    results['missing_cur_pres_cur_missing'] = (X_mcp[:, :, 0] == missing_value).all().item()
    results['missing_cur_pres_pres_missing'] = (X_mcp[:, :, 1] == missing_value).all().item()

    X_mall = ablations['missing_all_sensors_prev_only'][0]
    results['missing_all_sensors_cur_missing'] = (X_mall[:, :, 0] == missing_value).all().item()
    results['missing_all_sensors_pres_missing'] = (X_mall[:, :, 1] == missing_value).all().item()
    results['missing_all_sensors_rad_missing'] = (X_mall[:, :, 2] == missing_value).all().item()
    
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


def test_model_forward(models, forward_fn, X_sample, missing_value=0.0):
    """Verify model handles missing values without errors."""
    results = {}
    
    try:
        logits = forward_fn(models, X_sample)
        results['clean_forward_ok'] = not (torch.isnan(logits).any() or torch.isinf(logits).any())
        results['clean_output_shape_ok'] = tuple(logits.shape) == (X_sample.shape[0], 1)
    except Exception:
        results['clean_forward_ok'] = False
        results['clean_output_shape_ok'] = False
    
    try:
        X_test = X_sample.clone()
        X_test[:, :, 0] = missing_value
        logits = forward_fn(models, X_test)
        results['ablated_forward_ok'] = not (torch.isnan(logits).any() or torch.isinf(logits).any())
        results['ablated_output_shape_ok'] = tuple(logits.shape) == (X_sample.shape[0], 1)
    except Exception:
        results['ablated_forward_ok'] = False
        results['ablated_output_shape_ok'] = False
    
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


def test_sensor_correlations(corr_results, xcorr_tolerance=0.15):
    """
    Verify sensor correlation analysis produced valid results.
    
    Args:
        corr_results: Dict returned by analyze_sensor_correlations()
        xcorr_tolerance: Max allowed difference between xcorr at lag=0 and global Pearson r
    
    Returns:
        Dict of {check_name: bool} pass/fail results
    """
    results = {}
    
    # Data sanity
    results['files_loaded'] = corr_results['n_files'] > 0
    
    if corr_results['pearson'] is None:
        results['analysis_completed'] = False
        return results
    results['analysis_completed'] = True
    
    pearson = corr_results['pearson'].values
    spearman = corr_results['spearman'].values
    
    # Correlation matrix validity
    results['pearson_in_range'] = bool(np.all(pearson >= -1) and np.all(pearson <= 1))
    results['spearman_in_range'] = bool(np.all(spearman >= -1) and np.all(spearman <= 1))
    results['pearson_diagonal_is_one'] = bool(np.allclose(np.diag(pearson), 1.0, atol=1e-6))
    results['pearson_is_symmetric'] = bool(np.allclose(pearson, pearson.T, atol=1e-6))
    
    # Per-file consistency
    per_file = corr_results['per_file_corrs']
    if per_file is not None:
        all_in_range = True
        no_empty_pairs = True
        for pn, vals in per_file.items():
            if len(vals) == 0:
                no_empty_pairs = False
                continue
            if not all(-1 <= v <= 1 for v in vals):
                all_in_range = False
        results['per_file_values_in_range'] = all_in_range
        results['all_pairs_have_data'] = no_empty_pairs
        results['multiple_files_analyzed'] = all(len(v) > 1 for v in per_file.values())
    
    # Cross-correlation consistency
    xcorr = corr_results.get('xcorr')
    lags = corr_results.get('xcorr_lags')
    if xcorr is not None and lags is not None:
        pair_names = ["Cur-Pres", "Cur-Rad", "Pres-Rad"]
        pair_indices = [(0, 1), (0, 2), (1, 2)]
        lag0_idx = np.where(lags == 0)[0]
        
        if len(lag0_idx) > 0:
            lag0_idx = lag0_idx[0]
            for (i, j), pn in zip(pair_indices, pair_names):
                if pn in xcorr:
                    xcorr_at_0 = xcorr[pn][lag0_idx]
                    pearson_r = pearson[i, j]
                    results[f'xcorr_lag0_matches_pearson_{pn}'] = bool(
                        abs(xcorr_at_0 - pearson_r) < xcorr_tolerance
                    )
        
        # Peak should not be at the extreme edge of the lag range
        for pn in pair_names:
            if pn in xcorr:
                peak_idx = np.argmax(np.abs(xcorr[pn]))
                peak_lag = lags[peak_idx]
                max_lag = lags[-1]
                results[f'xcorr_peak_not_at_edge_{pn}'] = bool(abs(peak_lag) < max_lag * 0.9)
    
    return results


def run_validation_checks(X_train, X_val, 
                          models, forward_fn, ablations, robustness_results,
                          model_type='cnn', missing_value=0.0, corr_results=None):
    """Run all validation checks and return consolidated results."""
    all_results = {}
    
    all_results['data_split'] = test_data_split(X_train, X_val)
    all_results['channel_count'] = test_channel_count(X_train, models, model_type)
    all_results['dropout_function'] = test_dropout_function(X_train[:32], missing_value=missing_value)
    all_results['ablation_suite'] = test_ablation_suite(ablations, X_val, missing_value=missing_value)
    all_results['model_forward'] = test_model_forward(models, forward_fn, X_val[:32], missing_value=missing_value)
    all_results['robustness_sanity'] = test_robustness_sanity(robustness_results)
    if corr_results is not None:
        all_results['sensor_correlations'] = test_sensor_correlations(corr_results)
    
    return all_results


# ==========================================
# 6.9 Multi-seed runner + aggregator
# ==========================================

# Seed grids requested for the multi-seed sweep. Cartesian product = 25 runs.
WEIGHT_INIT_SEEDS = [33, 545, 97, 123, 999]
SPLIT_SEEDS       = [12, 97, 435, 8765, 88]


def _read_run_metrics(run_folder):
    """Read a single run's test_metrics.json + val_threshold_selection.json.

    Also pulls the training_info / architecture block (if present) so the
    aggregate CSV can carry per-run wall-clock time, mean epoch time, device,
    architecture name, and parameter count.
    """
    test_path = os.path.join(run_folder, "test_metrics.json")
    val_path = os.path.join(run_folder, "val_threshold_selection.json")
    meta_path = os.path.join(run_folder, "run_metadata.json")
    row = {
        "test_accuracy": np.nan,
        "test_precision": np.nan,
        "test_recall": np.nan,
        "test_f1": np.nan,
        "test_mcc": np.nan,
        "test_pr_auc": np.nan,
        "chosen_threshold": np.nan,
        "criterion": None,
        "status": "missing",
        # Training / architecture metadata (filled below if available).
        "model_type": None,
        "cnn_architecture": None,
        "n_trainable_params": np.nan,
        "epochs_run": np.nan,
        "best_epoch": np.nan,
        "stopped_early": np.nan,
        "total_training_seconds": np.nan,
        "mean_epoch_seconds": np.nan,
        "device_type": None,
        "device_name": None,
    }

    training_info_source = None

    if os.path.exists(test_path):
        try:
            with open(test_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            tm = rep.get("test_metrics", rep)
            row["test_accuracy"]  = float(tm.get("test_accuracy", np.nan))
            row["test_precision"] = float(tm.get("precision", np.nan))
            row["test_recall"]    = float(tm.get("recall", np.nan))
            row["test_f1"]        = float(tm.get("f1", np.nan))
            row["test_mcc"]       = float(tm.get("mcc", np.nan))
            row["test_pr_auc"]    = float(tm.get("pr_auc", np.nan))
            row["chosen_threshold"] = float(tm.get("decision_threshold", np.nan))
            row["status"] = "ok"
            row["model_type"] = rep.get("model_type", row["model_type"])
            training_info_source = rep.get("training_info")
        except Exception as exc:
            row["status"] = f"read_error: {exc}"

    if training_info_source is None and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if row["model_type"] is None:
                row["model_type"] = meta.get("model_type")
            training_info_source = meta.get("training_info")
        except Exception:
            pass

    if isinstance(training_info_source, dict):
        ti = training_info_source
        row["cnn_architecture"] = ti.get("cnn_architecture")
        row["n_trainable_params"] = float(ti.get("n_trainable_params", np.nan)) \
            if ti.get("n_trainable_params") is not None else np.nan
        row["epochs_run"] = float(ti.get("epochs_run", np.nan)) \
            if ti.get("epochs_run") is not None else np.nan
        row["best_epoch"] = float(ti.get("best_epoch", np.nan)) \
            if ti.get("best_epoch") is not None else np.nan
        row["stopped_early"] = bool(ti.get("stopped_early")) \
            if ti.get("stopped_early") is not None else np.nan
        row["total_training_seconds"] = float(ti.get("total_training_seconds", np.nan)) \
            if ti.get("total_training_seconds") is not None else np.nan
        row["mean_epoch_seconds"] = float(ti.get("mean_epoch_seconds", np.nan)) \
            if ti.get("mean_epoch_seconds") is not None else np.nan
        dev = ti.get("device") or {}
        row["device_type"] = dev.get("device_type")
        row["device_name"] = dev.get("device_name")
        if row["model_type"] is None:
            row["model_type"] = ti.get("model_type")

    if os.path.exists(val_path):
        try:
            with open(val_path, "r", encoding="utf-8") as f:
                vsel = json.load(f)
            row["criterion"] = vsel.get("criterion")
            if np.isnan(row["chosen_threshold"]):
                row["chosen_threshold"] = float(vsel.get("chosen_threshold", np.nan))
        except Exception:
            pass

    return row


def _aggregate_multi_seed_validation_metrics(root_folder):
    """Run the validation aggregator on the multi-seed grid root folder.

    This produces ``all_runs_validation_metrics.csv`` plus
    ``aggregate_validation_metrics_mean_sd.csv`` / ``.json`` next to the
    test-aggregate files, so each multi-seed run yields BOTH a test summary
    and a validation summary (including the average chosen threshold).

    The aggregator is implemented in the sibling ``aggregate_validation_metrics``
    module. We try to import it relative to this script's folder, then call its
    ``aggregate(root_folder)`` function. Any failure is logged but does not
    raise, since validation aggregation is a reporting convenience and must
    not break the main multi-seed flow.
    """
    print("\n" + "=" * 80)
    print("Aggregating VALIDATION metrics across multi-seed runs...")
    print("=" * 80)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            from aggregate_validation_metrics import aggregate as _val_aggregate
        except ImportError as exc:
            print(f"  [info] Could not import aggregate_validation_metrics: {exc}")
            print("  [info] Falling back to subprocess invocation.")
            agg_script = os.path.join(script_dir, "aggregate_validation_metrics.py")
            if os.path.exists(agg_script):
                subprocess.run(
                    [sys.executable, agg_script, root_folder],
                    check=False,
                )
            else:
                print(f"  [warn] aggregate_validation_metrics.py not found at "
                      f"{agg_script}; skipping validation aggregation.")
            return
        _val_aggregate(root_folder, output_folder=root_folder)
    except Exception as exc:
        print(f"  [warn] Validation aggregation failed: {exc}")


def aggregate_multi_seed_results(root_folder, run_records):
    """Write per-run and aggregate mean/std CSV summaries for the multi-seed grid."""
    metric_names = ["test_accuracy", "test_precision", "test_recall",
                    "test_f1", "test_mcc", "test_pr_auc"]
    rows = []
    for rec in run_records:
        row = {
            "run_index": rec["run_index"],
            "weight_init_seed": rec["weight_init_seed"],
            "split_seed": rec["split_seed"],
            "output_folder": rec["output_folder"],
        }
        row.update(_read_run_metrics(rec["output_folder"]))
        rows.append(row)

    all_df = pd.DataFrame(rows)
    all_csv = os.path.join(root_folder, "all_runs_metrics.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved: {all_csv}")

    ok_df = all_df[all_df["status"] == "ok"]
    summary_rows = []
    for m in metric_names:
        vals = pd.to_numeric(ok_df[m], errors="coerce").dropna()
        summary_rows.append({
            "metric": m,
            "n_runs": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "sd":   float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min":  float(vals.min()) if len(vals) else float("nan"),
            "max":  float(vals.max()) if len(vals) else float("nan"),
            "median": float(vals.median()) if len(vals) else float("nan"),
        })

    # Also report threshold stats
    t_vals = pd.to_numeric(ok_df["chosen_threshold"], errors="coerce").dropna()
    summary_rows.append({
        "metric": "chosen_threshold",
        "n_runs": int(len(t_vals)),
        "mean":   float(t_vals.mean()) if len(t_vals) else float("nan"),
        "sd":     float(t_vals.std(ddof=1)) if len(t_vals) > 1 else 0.0,
        "min":    float(t_vals.min()) if len(t_vals) else float("nan"),
        "max":    float(t_vals.max()) if len(t_vals) else float("nan"),
        "median": float(t_vals.median()) if len(t_vals) else float("nan"),
    })

    # Training-time and epoch stats across runs.
    for time_col in ["total_training_seconds", "mean_epoch_seconds",
                     "epochs_run", "best_epoch", "n_trainable_params"]:
        col_vals = pd.to_numeric(ok_df.get(time_col, pd.Series(dtype=float)),
                                  errors="coerce").dropna()
        summary_rows.append({
            "metric": time_col,
            "n_runs": int(len(col_vals)),
            "mean":   float(col_vals.mean()) if len(col_vals) else float("nan"),
            "sd":     float(col_vals.std(ddof=1)) if len(col_vals) > 1 else 0.0,
            "min":    float(col_vals.min()) if len(col_vals) else float("nan"),
            "max":    float(col_vals.max()) if len(col_vals) else float("nan"),
            "median": float(col_vals.median()) if len(col_vals) else float("nan"),
        })

    arch_modes = ok_df.get("cnn_architecture", pd.Series(dtype=object)).dropna()
    model_modes = ok_df.get("model_type", pd.Series(dtype=object)).dropna()
    device_modes = ok_df.get("device_name", pd.Series(dtype=object)).dropna()

    summary_csv = os.path.join(root_folder, "aggregate_metrics_mean_sd.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    total_train_seconds_sum = pd.to_numeric(
        ok_df.get("total_training_seconds", pd.Series(dtype=float)),
        errors="coerce").dropna().sum()

    summary_json = os.path.join(root_folder, "aggregate_metrics_mean_sd.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "n_expected_runs": len(run_records),
            "n_completed_runs": int((all_df["status"] == "ok").sum()),
            "weight_init_seeds": WEIGHT_INIT_SEEDS,
            "split_seeds": SPLIT_SEEDS,
            "model_types_seen": sorted(model_modes.unique().tolist())
                if not model_modes.empty else [],
            "cnn_architectures_seen": sorted(arch_modes.unique().tolist())
                if not arch_modes.empty else [],
            "devices_seen": sorted(device_modes.unique().tolist())
                if not device_modes.empty else [],
            "total_training_seconds_sum_across_runs": float(total_train_seconds_sum),
            "metrics": summary_rows,
        }, f, indent=2)
    print(f"Saved: {summary_json}")

    print("\nOverall test metrics across completed runs (mean +/- SD):")
    for r in summary_rows:
        print(f"  {r['metric']:>26s}: {r['mean']:.4f} +/- {r['sd']:.4f}  (n={r['n_runs']})")
    if total_train_seconds_sum > 0:
        print(f"\nTotal training time across all completed runs: "
              f"{_format_seconds(total_train_seconds_sum)} "
              f"({total_train_seconds_sum:.2f} s)")

    # Also aggregate VALIDATION metrics (including average chosen threshold)
    # at the same root, mirroring the test summary CSV/JSON.
    _aggregate_multi_seed_validation_metrics(root_folder)


def run_multi_seed_experiments():
    """Launch 25 isolated child runs (Cartesian seed grid) and aggregate metrics.

    Each child is launched as a subprocess with environment variables that put
    the script into single-run mode and feed it the chosen seeds and output
    folder. This keeps RNG state and matplotlib state cleanly isolated.
    """
    root_folder = os.path.abspath(create_output_folder(prefix="polgun_cnn_multiseed"))
    total_runs = len(WEIGHT_INIT_SEEDS) * len(SPLIT_SEEDS)
    grid_start = time.perf_counter()
    grid_start_wall = datetime.now().isoformat(timespec="seconds")
    print("\n" + "=" * 80)
    print(f"STARTING MULTI-SEED EXPERIMENT GRID: {total_runs} RUNS")
    print(f"Weight-init seeds: {WEIGHT_INIT_SEEDS}")
    print(f"Split seeds:       {SPLIT_SEEDS}")
    print(f"Root output:       {root_folder}")
    print(f"Grid start time:   {grid_start_wall}")
    print("=" * 80 + "\n")

    run_records = []
    run_index = 0
    for w_seed in WEIGHT_INIT_SEEDS:
        for s_seed in SPLIT_SEEDS:
            run_index += 1
            run_name = f"run_{run_index:02d}_init_{w_seed}_split_{s_seed}"
            run_folder = os.path.join(root_folder, run_name)
            os.makedirs(os.path.join(run_folder, "figures"), exist_ok=True)

            print("\n" + "=" * 80)
            print(f"RUN {run_index}/{total_runs}: weight_init_seed={w_seed}, split_seed={s_seed}")
            print(f"Output folder: {run_folder}")
            print("=" * 80)

            env = os.environ.copy()
            env["POLGUN_SINGLE_RUN"] = "1"
            env["POLGUN_WEIGHT_INIT_SEED"] = str(w_seed)
            env["POLGUN_SPLIT_SEED"] = str(s_seed)
            env["POLGUN_OUTPUT_FOLDER"] = run_folder
            env["PYTHONUNBUFFERED"] = "1"
            env["MPLBACKEND"] = "Agg"

            _run_start = time.perf_counter()
            try:
                subprocess.run([sys.executable, os.path.abspath(__file__)],
                               env=env, check=True)
                status = "completed"
            except subprocess.CalledProcessError as exc:
                print(f"!! Run {run_index} failed with exit code {exc.returncode}")
                status = f"failed (exit {exc.returncode})"
            _run_elapsed = time.perf_counter() - _run_start
            print(f">>> Run {run_index} wall-clock: {_format_seconds(_run_elapsed)}")

            run_records.append({
                "run_index": run_index,
                "weight_init_seed": w_seed,
                "split_seed": s_seed,
                "output_folder": run_folder,
                "status": status,
                "subprocess_wall_seconds": float(_run_elapsed),
            })

    aggregate_multi_seed_results(root_folder, run_records)
    grid_elapsed = time.perf_counter() - grid_start
    grid_end_wall = datetime.now().isoformat(timespec="seconds")
    print("\nAll multi-seed runs complete.")
    print(f"Root folder:      {root_folder}")
    print(f"Grid wall-clock:  {_format_seconds(grid_elapsed)} "
          f"({grid_elapsed:.2f} s, start {grid_start_wall} -> end {grid_end_wall})")


# ==========================================
# 7. Main Execution (Looping over ALL Test Files)
# ==========================================

if __name__ == "__main__":

    # If we're not in single-run mode, dispatch to the multi-seed grid.
    if os.environ.get("POLGUN_SINGLE_RUN") != "1":
        run_multi_seed_experiments()
        sys.exit(0)

    # ---- Single-run mode (one weight-init seed + one split seed) ----
    WEIGHT_INIT_SEED = int(os.environ.get("POLGUN_WEIGHT_INIT_SEED", "545"))
    SPLIT_SEED = int(os.environ.get("POLGUN_SPLIT_SEED", "97"))
    set_all_seeds(WEIGHT_INIT_SEED)
    print(f"\n>>> Single-run seeds: weight_init_seed={WEIGHT_INIT_SEED}, "
          f"split_seed={SPLIT_SEED}")

    # UPDATE YOUR PATHS HERE
    TRAIN_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\train poster" 
    TEST_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\polgun v8 until max conditioning\v8 spikes cleaned until max\test poster"
    
    ##Noisy test data. For model validation.
    #TEST_DIR = r"C:\Users\suman\Downloads\Stony Brook\PhD\Electron Gun\Data\Archive 2\Data For Model\Test Data"  #

    # TRAIN_DIR = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\gun_conditioning_spike_cleaned\v8 spikes cleaned until max\training" 
    # TEST_DIR = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\gun_conditioning_spike_cleaned\v8 spikes cleaned until max\testing"
    
    # ==========================================
    # HYPERPARAMETERS (all in one place for easy tuning)
    # ==========================================
    HYPERPARAMS = {
        'sequence_length': 30,
        'learning_rate': 0.0001,
        'epochs': 75,
        'batch_size': 128,
        'cnn_architecture': 'mid_fusion_pointwise',
        'n_filters1': 128,
        'fusion_channels': 128,
        'n_filters2': 256,
        'kernel_size': 7,
        'pool_size': 2,                      #'mid_fusion_pointwise' or 'split_fusion_3plus1' or 'early_fusion'
        'branch2_filters': 150,             # split_fusion_3plus1: filters per branch in conv2
        'fc_hidden': 500,                   # split_fusion_3plus1: hidden units in first FC layer
        'hidden_dim': 64,  # For LSTM
        # Dropout/Missing data augmentation
        'prob_full_missing': 0.0,          # 20% chance of full window dropout
        'prob_block_missing': 0.0,         # 30% chance of block dropout
        'dropout_min_block_pct': 0.0,      # Min block = 20% of seq_length
        'dropout_max_block_pct': 0.0,      # Max block = 80% of seq_length
        'missing_value': 0.0,              # Placeholder for missing/dropped data
        # Seeds (set per run by the multi-seed grid)
        'weight_init_seed': WEIGHT_INIT_SEED,
        'split_seed': SPLIT_SEED,
        # Decision threshold selected on validation; choices: max_mcc, max_f1, fixed_0_5
        'threshold_criterion': 'max_mcc',
        # Early stopping on val_loss; 0/None disables. Best-epoch weights are
        # reloaded into the model before any downstream evaluation.
        'early_stopping_patience': 15,
    }

    SEQ_LEN = HYPERPARAMS['sequence_length']
    MODEL_TYPE = 'lstm'  # 'cnn' or 'lstm'

    # ==========================================
    # CREATE OUTPUT FOLDER (or reuse one supplied by the multi-seed runner)
    # ==========================================
    _env_output_folder = os.environ.get("POLGUN_OUTPUT_FOLDER")
    if _env_output_folder:
        OUTPUT_FOLDER = _env_output_folder
        os.makedirs(os.path.join(OUTPUT_FOLDER, "figures"), exist_ok=True)
        print(f">>> Using output folder provided by runner: {OUTPUT_FOLDER}")
    else:
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
        filter_quiet_negatives=True,
        missing_value=HYPERPARAMS['missing_value'],
        split_seed=SPLIT_SEED,
    )

    # 2.5 Sensor Correlation Analysis
    corr_results = analyze_sensor_correlations(TRAIN_DIR, output_folder=OUTPUT_FOLDER)
    
    # 3. Train (with sensor dropout augmentation)
    models, fwd_fn, history, initial_weights, final_weights, weight_history = train_model(
        MODEL_TYPE, data, pos_weight, 
        epochs=HYPERPARAMS['epochs'], 
        batch_size=HYPERPARAMS['batch_size'],
        learning_rate=HYPERPARAMS['learning_rate'],
        n_filters1=HYPERPARAMS['n_filters1'],
        fusion_channels=HYPERPARAMS['fusion_channels'],
        n_filters2=HYPERPARAMS['n_filters2'],
        kernel_size=HYPERPARAMS['kernel_size'],
        pool_size=HYPERPARAMS['pool_size'],
        cnn_architecture=HYPERPARAMS['cnn_architecture'],
        branch2_filters=HYPERPARAMS['branch2_filters'],
        fc_hidden=HYPERPARAMS['fc_hidden'],
        hidden_dim=HYPERPARAMS['hidden_dim'],
        prob_full_missing=HYPERPARAMS['prob_full_missing'],
        prob_block_missing=HYPERPARAMS['prob_block_missing'],
        min_block_pct=HYPERPARAMS['dropout_min_block_pct'],
        max_block_pct=HYPERPARAMS['dropout_max_block_pct'],
        missing_value=HYPERPARAMS['missing_value'],
        early_stopping_patience=HYPERPARAMS['early_stopping_patience'],
    )

    # Save a human-readable text report of the chosen architecture,
    # hyperparameters, model keys, and weight shapes.
    save_text_report(
        OUTPUT_FOLDER,
        "architecture_hyperparameters",
        build_architecture_hyperparameter_report(
            models=models,
            model_type=MODEL_TYPE,
            hyperparams=HYPERPARAMS,
            data=data,
            pos_weight=pos_weight,
            weight_init_seed=WEIGHT_INIT_SEED,
            split_seed=SPLIT_SEED,
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            output_folder=OUTPUT_FOLDER,
            history=history,
        ),
    )

    # Save a standalone run_metadata.json with architecture + training-time info
    # so we still have this on disk even if a later stage of the pipeline fails.
    try:
        _run_meta = {
            "weight_init_seed": WEIGHT_INIT_SEED,
            "split_seed": SPLIT_SEED,
            "model_type": MODEL_TYPE,
            "output_folder": OUTPUT_FOLDER,
            "hyperparameters": HYPERPARAMS,
            "training_info": history.get("training_info", {}),
        }
        with open(os.path.join(OUTPUT_FOLDER, "run_metadata.json"), "w",
                  encoding="utf-8") as _f:
            json.dump(_run_meta, _f, indent=2)
        print(f"Saved: {os.path.join(OUTPUT_FOLDER, 'run_metadata.json')}")
    except Exception as _e:
        print(f"[WARN] Could not save run_metadata.json: {_e}")

    # Save core training curves immediately after training/metadata. These plots
    # should survive even if later robustness, ablation, or simulation steps fail.
    plot_training_curves(history, output_folder=OUTPUT_FOLDER)
    plot_mcc_curve(history, output_folder=OUTPUT_FOLDER)

    # ==========================================
    # 3.1 Validation PR Curve + Threshold Selection
    # ==========================================
    # This must happen before any post-training evaluation/ablation so every
    # class decision uses the same validation-committed threshold.
    CHOSEN_THRESHOLD, threshold_selection = compute_validation_pr_threshold(
        models, fwd_fn, data,
        output_folder=OUTPUT_FOLDER,
        criterion=HYPERPARAMS.get('threshold_criterion', 'max_mcc'),
        default_threshold=0.5,
    )

    # NOTE: The robustness/ablation suite (validation + test ablations) is the
    # most memory-intensive stage and has historically been the one to crash
    # (OOM) on large/LSTM runs. It has been moved to the very end of the run so
    # that every other artifact (training curves, threshold selection, decision
    # plots, test metrics, per-file metrics, channel importance, activation
    # stats, checkpoints, and simulations) is saved BEFORE the ablation suite
    # executes. See the "ROBUSTNESS ABLATION SUITE" block near the final summary.

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
    _act_buf = io.StringIO()
    _tee = TeeWriter(sys.stdout, _act_buf)
    sys.stdout = _tee
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
    sys.stdout = _tee.original
    save_text_report(OUTPUT_FOLDER, "activation_statistics", _act_buf.getvalue())
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
    if ei_ratios is not None:
        print_ei_consistency_check(ei_ratios, channel_names=CHANNEL_NAMES)
    
    # ==========================================
    # 5. EXISTING VISUALIZATIONS (kept for compatibility)
    # ==========================================
    if "conv1" in initial_weights:
        viz_layer(initial_weights, final_weights, "conv1", include_bias=False)
    viz_layer(initial_weights, final_weights, "linear", include_bias=True)
    #plot_conv1_input_channel_hists(models, initial_weights=initial_weights, final_weights=final_weights)

    #########################################################################
    # Save weights comparison to text file (in output folder)
    # Only runs for architectures that have a single conv1 layer.
    #########################################################################
    if "conv1" in initial_weights:
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
            layout_init = get_conv1_channel_layout(w_init, CHANNEL_NAMES)
            layout_final = get_conv1_channel_layout(w_final, CHANNEL_NAMES)
            print("conv1 logical layout:", layout_init["layout"])

            for in_ch, ch_name in enumerate(CHANNEL_NAMES):
                init_ch = layout_init["channel_blocks"][in_ch]
                final_ch = layout_final["channel_blocks"][in_ch]

                print("\n" + "="*80)
                print(
                    f"INPUT CHANNEL {in_ch}: {ch_name}  |  weights per channel = {init_ch.size} "
                    f"({init_ch.shape[0]} filters × {init_ch.shape[1]} kernel)"
                )
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
    # 6.5 Validation decision probabilities + decision-boundary plot
    # ==========================================
    for layer in models.values():
        layer.eval()
    with torch.inference_mode():
        _val_logits = fwd_fn(models, data['X_val'])
        _val_probs = torch.sigmoid(_val_logits).cpu().numpy().ravel()
    _val_y = data['y_val'].cpu().numpy().ravel().astype(int)
    save_decision_probabilities_csv(
        OUTPUT_FOLDER, "val_decision_probabilities.csv",
        _val_probs, _val_y, CHOSEN_THRESHOLD, section_name="VAL"
    )
    plot_threshold_decision_boundary(
        OUTPUT_FOLDER, "val_decision_boundary",
        _val_probs, _val_y, CHOSEN_THRESHOLD, title_prefix="Validation"
    )
    plot_relative_threshold_decision_boundary(
        OUTPUT_FOLDER, "val_relative_decision_boundary",
        _val_probs, _val_y, CHOSEN_THRESHOLD, title_prefix="Validation"
    )

    # ==========================================
    # 7. Evaluate on Test Set (at validation-committed threshold)
    # ==========================================
    test_metrics = evaluate_test_set(
        models, fwd_fn, data, pos_weight,
        output_folder=OUTPUT_FOLDER,
        threshold=CHOSEN_THRESHOLD,
    )

    if test_metrics:
        _test_probs = np.array(test_metrics.pop("_test_probabilities"))
        _test_preds = np.array(test_metrics.pop("_test_predictions"))
        _test_y = np.array(test_metrics.pop("_test_true_labels"))

        save_decision_probabilities_csv(
            OUTPUT_FOLDER, "test_decision_probabilities.csv",
            _test_probs, _test_y, CHOSEN_THRESHOLD, section_name="TEST"
        )
        plot_threshold_decision_boundary(
            OUTPUT_FOLDER, "test_decision_boundary",
            _test_probs, _test_y, CHOSEN_THRESHOLD, title_prefix="Test"
        )
        plot_relative_threshold_decision_boundary(
            OUTPUT_FOLDER, "test_relative_decision_boundary",
            _test_probs, _test_y, CHOSEN_THRESHOLD, title_prefix="Test"
        )

        # Per-test-file metrics + decision artifacts (one folder per source CSV).
        # Uses the same validation-committed t* used for the aggregate metrics
        # so per-file results are directly comparable. No tuning on test data.
        per_file_test_metrics = evaluate_test_set_by_file(
            models, fwd_fn, data,
            output_folder=OUTPUT_FOLDER,
            threshold=CHOSEN_THRESHOLD,
        )

        # Save consolidated test metrics report (JSON) with threshold context.
        _test_report = {
            "weight_init_seed": WEIGHT_INIT_SEED,
            "split_seed": SPLIT_SEED,
            "model_type": MODEL_TYPE,
            "hyperparameters": HYPERPARAMS,
            "threshold_selection": threshold_selection,
            "test_metrics": test_metrics,
            "per_file_test_metrics": per_file_test_metrics,
            "per_file_test_metrics_csv": (
                os.path.join(OUTPUT_FOLDER, "test_file_metrics.csv")
                if per_file_test_metrics else None
            ),
            "per_file_test_metrics_json": (
                os.path.join(OUTPUT_FOLDER, "test_file_metrics.json")
                if per_file_test_metrics else None
            ),
            "training_info": history.get("training_info", {}),
        }
        with open(os.path.join(OUTPUT_FOLDER, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_test_report, f, indent=2)
        print(f"Saved: {os.path.join(OUTPUT_FOLDER, 'test_metrics.json')}")

    # Save the CLEAN test per-sample probability report now (test_probabilities.txt).
    # The per-ablation probability sections are deferred to the end alongside the
    # robustness ablation suite, since they depend on the test ablations created there.
    save_test_probability_report(
        OUTPUT_FOLDER,
        models,
        fwd_fn,
        data,
        test_ablations=None,
        threshold=CHOSEN_THRESHOLD,
        filename="test_probabilities",
        include_clean=True,
    )

    _san_buf = io.StringIO()
    _tee_san = TeeWriter(sys.stdout, _san_buf)
    sys.stdout = _tee_san
    #run_activation_sanity_checks(models, seq_len=SEQ_LEN, channel_names=CHANNEL_NAMES, device="cpu")
    sys.stdout = _tee_san.original
    save_text_report(OUTPUT_FOLDER, "activation_sanity_checks", _san_buf.getvalue())
    
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
        output_folder=OUTPUT_FOLDER,
        missing_value=HYPERPARAMS['missing_value'],
        threshold=CHOSEN_THRESHOLD,
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

        # Trend plot (6-panel) with figure saving (uses validation-chosen t*)
        plot_test_file_trends(
            test_file,
            models,
            fwd_fn,
            scaler,
            sequence_length=SEQ_LEN,
            output_folder=OUTPUT_FOLDER,
            threshold=CHOSEN_THRESHOLD,
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
            missing_value=HYPERPARAMS['missing_value'],
            output_folder=OUTPUT_FOLDER,
            threshold=CHOSEN_THRESHOLD,
        )

        # Also simulate under the four requested multi-sensor ablation scenarios (cases 2–5)
        for abl_name in [
            'missing_pres_rad',
            'missing_cur_rad',
            'missing_cur_pres',
            'missing_all_sensors_prev_only',
        ]:
            test_single_file_simulation_with_ablation(
                test_file,
                models,
                fwd_fn,
                scaler,
                sequence_length=SEQ_LEN,
                step_size=AVG_STEP_SIZE,
                ablation_name=abl_name,
                missing_value=HYPERPARAMS['missing_value'],
                initial_A=0,
                max_limit_A=MAX_VOLTAGE_LIMIT,
                output_folder=OUTPUT_FOLDER,
                threshold=CHOSEN_THRESHOLD,
            )

        # Single-channel full/block and Prev_A1 ablation simulations
        for abl_name in [
            'current_full',
            'pressure_full',
            'radiation_full',
            'current_block',
            'pressure_block',
            'radiation_block',
            'prev_a1_shuffle',
            'prev_a1_const_0',
            'prev_a1_const_1',
            'prev_a1_flip',
        ]:
            test_single_file_simulation_with_single_ablation(
                test_file,
                models,
                fwd_fn,
                scaler,
                sequence_length=SEQ_LEN,
                step_size=AVG_STEP_SIZE,
                ablation_name=abl_name,
                missing_value=HYPERPARAMS['missing_value'],
                initial_A=0,
                max_limit_A=MAX_VOLTAGE_LIMIT,
                output_folder=OUTPUT_FOLDER,
                threshold=CHOSEN_THRESHOLD,
            )

    print("\nAll Simulations Complete.")

    # ==========================================
    # 3.5 ROBUSTNESS ABLATION SUITE (Validation + Test)  [RUNS LAST]
    # ==========================================
    # Moved to the end on purpose: this is the most memory-intensive stage and
    # the historical OOM crash point. Running it last guarantees all other
    # artifacts above are already written even if this step fails.
    print("\n>>> Running Robustness Ablation Suite...")

    # --- VALIDATION SET ABLATIONS ---
    print("\n--- Validation Set Ablations ---")
    val_ablations, val_missing_mask = create_validation_ablations(
        data['X_val'], data['y_val'], 
        missing_value=HYPERPARAMS['missing_value']
    )
    val_robustness = evaluate_robustness(
        models, fwd_fn, val_ablations,
        missing_mask=val_missing_mask,
        filter_missing=True,
        threshold=CHOSEN_THRESHOLD,
    )
    
    # --- TEST SET ABLATIONS ---
    print("\n--- Test Set Ablations ---")
    test_ablations, test_missing_mask = create_test_ablations(
        data['X_test'], data['y_test'], 
        missing_value=HYPERPARAMS['missing_value']
    )
    test_robustness = evaluate_robustness(
        models, fwd_fn, test_ablations,
        missing_mask=test_missing_mask,
        filter_missing=False,
        threshold=CHOSEN_THRESHOLD,
    )
    
    # Run validation checks
    validation_checks = run_validation_checks(
        data['X_train'], data['X_val'],
        models, fwd_fn, val_ablations, val_robustness,
        model_type=MODEL_TYPE,
        missing_value=HYPERPARAMS['missing_value'],
        corr_results=corr_results
    )
    
    # Plot robustness ablation results
    plot_robustness_ablation(val_robustness, title_suffix="Validation - Clean Subset",
                            output_folder=OUTPUT_FOLDER, filename="robustness_ablation_val")
    plot_robustness_ablation(test_robustness, title_suffix="Test - All Samples",
                            output_folder=OUTPUT_FOLDER, filename="robustness_ablation_test")
    # Focused plots for multi-sensor missingness cases (2–5)
    plot_multi_missing_ablations(val_robustness, title_suffix="Validation - Clean Subset",
                                 output_folder=OUTPUT_FOLDER, filename="robustness_multi_missing_val")
    plot_multi_missing_ablations(test_robustness, title_suffix="Test - All Samples",
                                 output_folder=OUTPUT_FOLDER, filename="robustness_multi_missing_test")
    
    # Save comprehensive robustness report (includes both val and test)
    save_robustness_report(val_robustness, validation_checks, OUTPUT_FOLDER, 
                          test_robustness=test_robustness)

    # Per-sample ABLATION probability sections (clean test already saved earlier
    # as test_probabilities.txt). Written to a separate file so the clean report
    # is preserved even if this stage runs out of memory.
    save_test_probability_report(
        OUTPUT_FOLDER,
        models,
        fwd_fn,
        data,
        test_ablations=test_ablations,
        threshold=CHOSEN_THRESHOLD,
        filename="test_ablation_probabilities",
        include_clean=False,
    )

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("RUN COMPLETE - OUTPUT SUMMARY")
    print("="*70)
    print(f"All outputs saved to: {OUTPUT_FOLDER}/")
    print(f"\nFigures ({OUTPUT_FOLDER}/figures/):")
    print(f"  - channel_importance_*.png        (L2 norm analysis)")
    print(f"  - excitation_inhibition_*.png     (E/I sign-aware analysis)")
    print(f"  - ei_ratio_over_epochs.png        (E/I dynamics)")
    print(f"  - ablation_study_results.png      (Channel importance validation)")
    print(f"  - temporal_sensitivity_test.png   (Spike position sensitivity)")
    print(f"  - training_curves.png, mcc_curve.png")
    print(f"  - val_pr_curve.png                (Val PR curve + F1/MCC vs threshold)")
    print(f"  - val_decision_boundary.png       (TN/FP/FN/TP histograms @ t*)")
    print(f"  - val_relative_decision_boundary.png  (TN/FP/FN/TP margins from t*)")
    print(f"  - test_decision_boundary.png      (TN/FP/FN/TP histograms @ t*)")
    print(f"  - test_relative_decision_boundary.png (TN/FP/FN/TP margins from t*)")
    print(f"  - precision_recall_curve.png      (Test PR curve)")
    print(f"  - trend_*.png, simulation_*.png   (Per test file, uses t*)")
    print(f"\nDecision artifacts ({OUTPUT_FOLDER}/):")
    print(f"  - val_pr_curve_points.csv         (Per-threshold P/R/F1/MCC)")
    print(f"  - val_threshold_selection.json    (chosen t*, argmax F1 and MCC rows)")
    print(f"  - val_decision_probabilities.csv  (Val probs, TN/TP/FN/FP, margin)")
    print(f"  - test_decision_probabilities.csv (Test probs, TN/TP/FN/FP, margin)")
    print(f"  - test_metrics.json               (Test metrics @ committed t*)")
    print(f"  - test_file_metrics.csv           (Per-test-file metrics summary)")
    print(f"  - test_file_metrics.json          (Per-test-file detailed report)")
    print(f"\nPer-test-file outputs ({OUTPUT_FOLDER}/test_files/<file>/):")
    print(f"  - test_metrics.json               (Metrics for that single CSV)")
    print(f"  - test_decision_probabilities.csv (Per-sample TN/TP/FN/FP, margin)")
    print(f"  - figures/test_decision_boundary.png")
    print(f"  - figures/test_relative_decision_boundary.png")
    print(f"\nCheckpoints:")
    print(f"  - {OUTPUT_FOLDER}/checkpoint.pth      (lightweight)")
    print(f"  - {OUTPUT_FOLDER}/full_checkpoint.pth (complete)")
    print(f"\nLogs:")
    print(f"  - {OUTPUT_FOLDER}/conv1_weights_comparison.txt")
    print("="*70)
    print("\nTo load this model later:")
    print(f'  models, fwd_fn, scaler, ckpt = load_checkpoint("{OUTPUT_FOLDER}/full_checkpoint.pth")')
    print("="*70)