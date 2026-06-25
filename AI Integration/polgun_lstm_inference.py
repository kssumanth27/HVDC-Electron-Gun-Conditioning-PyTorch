"""
polgun_lstm_inference.py
========================

Standalone LSTM inference adapter for the electron-gun conditioning model.

This file lets you REUSE an already-trained LSTM checkpoint (from one of the
multi-seed runs) on a separate test set or on live EPICS data, WITHOUT
retraining anything. It loads the model + scaler once and then exposes three
prediction modes (see ``MODE`` in the config block at the bottom):

    1. "offline_csv"  -> predict_csv(...)
       Read a complete, finished CSV file and produce a prediction (0/1) for
       every valid window. Good for testing a saved dataset.

    2. "live_csv"     -> predict_latest_from_live_csv(...)
       Watch a CSV that the EPICS / p4p client keeps appending to. Only the
       newest rows are read each cycle (the file is tailed, not re-parsed from
       scratch). Returns the newest 0/1 suggestion.

    3. "direct_row"   -> append_row_and_predict(...)
       Lowest latency. The EPICS / p4p client hands one new sample (a dict)
       directly to the already-loaded model object. An internal rolling buffer
       keeps the last ``sequence_length`` rows. Returns the newest 0/1
       suggestion once enough rows have arrived.

IMPORTANT DESIGN RULES
----------------------
* The EPICS-generated CSV is treated as READ-ONLY. This file never edits it.
  Suggestions are written to a SEPARATE csv (default: ``lstm_suggestions.csv``).
* Preprocessing (normalization with the trained scaler, missing-value handling,
  the Prev_A1 / previous-action channel, and the decision threshold) lives HERE,
  because all of that is part of the trained-model contract. The client only has
  to provide raw values.

WHAT FILES THE MODEL NEEDS TO KNOW THE TRAINED INFORMATION
----------------------------------------------------------
* REQUIRED:    full_checkpoint.pth        (weights + scaler + hyperparameters)
* RECOMMENDED: val_threshold_selection.json  (the validation-chosen threshold
               t*, expected in the SAME folder as full_checkpoint.pth). If it is
               missing, the code falls back to 0.5 and prints a warning.
* NOT needed at runtime: the big training script. The minimal model definition
  and preprocessing are reproduced inside this file.

The 4 input channels (exactly as in training) are:
    [GunCurrent.Avg, peg-BL-cc:pressureM, RadiationTotal, Prev_A1]
where Prev_A1 is the previous increase(1)/hold(0) action history.
"""

import os
import csv
import json
import time
import argparse
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ==========================================================================
# Column names (must match the training pipeline)
# ==========================================================================
FEATURE_COLS = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
VOLTAGE_COLS = ["glassmanDataXfer:hvPsVoltageMeasM", "hvps.lerec:voltageM"]
VOLTAGECHANGE_COL = "VoltageChange"
INPUT_DIM = 4  # 3 sensors + Prev_A1


# ==========================================================================
# Minimal LSTM architecture (copied from the training script so we do not
# need to import the whole 8k-line file at runtime).
# ==========================================================================
def create_lstm_model(input_dim, hidden_dim):
    lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                   num_layers=2, batch_first=True)
    linear = nn.Linear(hidden_dim, 1)
    return {"lstm": lstm, "linear": linear}


def forward_lstm(models, x):
    # LSTM expects (Batch, Seq, Features) -> no permutation needed.
    lstm_out, _ = models["lstm"](x)
    return models["linear"](lstm_out[:, -1, :])  # last time step


# ==========================================================================
# Prediction result container
# ==========================================================================
@dataclass
class PredictionResult:
    """One prediction. ``suggestion`` is the final 0/1 decision."""
    timestamp: str
    source_row_index: int
    prob_increase: float
    threshold: float
    suggestion: int

    def as_row(self):
        return asdict(self)


def _to_float(value):
    """Parse a possibly-missing/blank value into float or NaN."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class PolgunLSTMInference:
    """Loads a trained LSTM checkpoint ONCE and serves predictions.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``full_checkpoint.pth`` from a trained LSTM run.
    threshold : float, optional
        Override the decision threshold. If None (default), the threshold is
        read from ``val_threshold_selection.json`` next to the checkpoint, and
        falls back to 0.5 if that file is missing.
    prev_a1_source : str
        How to build the Prev_A1 (previous-action) channel:
          - "auto"            : VoltageChange column if present, else voltage
                                delta, else the model's own previous decision.
          - "voltagechange"   : require a VoltageChange value.
          - "voltage_delta"   : derive 1 when voltage increased vs the previous
                                sample, else 0.
          - "decision_history": use the model's previous 0/1 output.
    voltage_increase_tol : float
        Minimum voltage rise (vs previous sample) counted as an "increase" when
        deriving Prev_A1 from voltage. Default 0.0 (any rise counts).
    device : str
        Torch device, default "cpu".
    """

    def __init__(self, checkpoint_path, threshold=None,
                 prev_a1_source="auto", voltage_increase_tol=0.0,
                 device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.prev_a1_source = prev_a1_source
        self.voltage_increase_tol = float(voltage_increase_tol)
        self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}")

        print(f">>> Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # --- Guard: this adapter is LSTM-only ---
        model_type = ckpt.get("model_type", None)
        state = ckpt.get("model_state_dict", {})
        is_lstm = (model_type == "lstm") or (
            "lstm" in state and "linear" in state and "conv1" not in state)
        if not is_lstm:
            raise ValueError(
                f"This adapter only supports LSTM checkpoints, but the "
                f"checkpoint reports model_type={model_type!r} with layers "
                f"{list(state.keys())}. Point it at an LSTM run's "
                f"full_checkpoint.pth.")

        # --- Hyperparameters ---
        hp = ckpt.get("hyperparameters", {})
        self.sequence_length = int(hp.get("sequence_length", 30))
        self.hidden_dim = int(hp.get("hidden_dim", 64))
        self.missing_value = float(hp.get("missing_value", 0.0))
        self.hyperparameters = hp

        # --- Reconstruct the trained scaler (sensors only: 3 features) ---
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(ckpt["scaler_mean"], dtype=float)
        self.scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=float)
        self.scaler.var_ = np.array(ckpt["scaler_var"], dtype=float)
        self.scaler.n_features_in_ = int(ckpt.get("scaler_n_features", 3))

        # --- Reconstruct the model + load weights ---
        self.models = create_lstm_model(INPUT_DIM, hidden_dim=self.hidden_dim)
        for name, sd in state.items():
            if name in self.models:
                self.models[name].load_state_dict(sd)
        for layer in self.models.values():
            layer.to(self.device)
            layer.eval()
        self.forward_fn = forward_lstm

        # --- Decision threshold ---
        if threshold is not None:
            self.threshold = float(threshold)
            self._threshold_source = "user override"
        else:
            self.threshold, self._threshold_source = self._load_threshold(
                checkpoint_path)

        # --- Live/streaming state (used by live_csv & direct_row modes) ---
        self._buffer = deque(maxlen=self.sequence_length)
        self._prev_voltage = None
        self._last_suggestion = 0.0
        self._row_count = 0
        # Live-CSV tailing state:
        self._live_path = None
        self._live_fieldnames = None
        self._live_offset = 0

        print(f">>> LSTM ready | seq_len={self.sequence_length} | "
              f"hidden_dim={self.hidden_dim} | missing_value={self.missing_value}")
        print(f">>> Decision threshold t*={self.threshold:.6f} "
              f"({self._threshold_source})")

    # ------------------------------------------------------------------
    # Threshold loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_threshold(checkpoint_path):
        """Read chosen_threshold from val_threshold_selection.json beside the
        checkpoint; fall back to 0.5 if unavailable."""
        run_folder = os.path.dirname(os.path.abspath(checkpoint_path))
        json_path = os.path.join(run_folder, "val_threshold_selection.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                t = data.get("chosen_threshold")
                if t is not None:
                    return float(t), f"from {os.path.basename(json_path)}"
            except Exception as e:
                print(f"    WARNING: could not read {json_path}: {e}")
        print("    WARNING: val_threshold_selection.json not found; "
              "falling back to threshold = 0.5")
        return 0.5, "fallback default 0.5"

    # ------------------------------------------------------------------
    # Core math: scale a raw window and run the model
    # ------------------------------------------------------------------
    def _scale_window(self, raw_window):
        """raw_window: (seq_len, 4) unscaled. Returns scaled copy.

        Only the 3 sensor channels are standardized with the trained scaler;
        NaN sensor positions become ``missing_value``. Prev_A1 (channel 3) is
        left untouched (it is already 0/1)."""
        X = np.asarray(raw_window, dtype=float).copy()
        sensor = X[:, 0:3]
        nan_mask = np.isnan(sensor)
        sensor_filled = np.nan_to_num(sensor, nan=0.0)
        sensor_scaled = self.scaler.transform(sensor_filled)
        sensor_scaled[nan_mask] = self.missing_value
        X[:, 0:3] = sensor_scaled
        # Any NaN left in Prev_A1 -> 0 (hold) for safety.
        X[:, 3] = np.nan_to_num(X[:, 3], nan=0.0)
        return X

    def _forward_prob(self, scaled_window):
        """scaled_window: (seq_len, 4). Returns probability of INCREASE."""
        t = torch.as_tensor(scaled_window, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logit = self.forward_fn(self.models, t)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    def _forward_prob_batch(self, scaled_windows):
        """scaled_windows: (N, seq_len, 4). Returns 1D array of probabilities."""
        t = torch.as_tensor(scaled_windows, dtype=torch.float32,
                            device=self.device)
        with torch.inference_mode():
            logits = self.forward_fn(self.models, t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    # ------------------------------------------------------------------
    # Prev_A1 helpers
    # ------------------------------------------------------------------
    def _clean_voltagechange_array(self, series):
        """Replicate training cleaning: map -1 -> 0 and enforce a 5-step
        refractory window (current + next 5 forced to 0), NaN -> 0."""
        vc_raw = pd.to_numeric(series, errors="coerce").values.astype(float)
        n = len(vc_raw)
        force_zero = np.zeros(n, dtype=bool)
        for idx in np.where(vc_raw == -1)[0]:
            force_zero[idx:min(n, idx + 6)] = True
        vc = vc_raw.copy()
        vc[vc == -1] = 0
        vc[force_zero] = 0
        vc = np.nan_to_num(vc, nan=0.0)
        return vc

    @staticmethod
    def _voltage_series_from_df(df):
        for col in VOLTAGE_COLS:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return None

    def _voltage_to_change_array(self, volt_series):
        """1 where voltage rose above tolerance vs previous sample, else 0."""
        if volt_series is None:
            return None
        v = pd.to_numeric(volt_series, errors="coerce").values.astype(float)
        change = np.zeros(len(v), dtype=float)
        for i in range(1, len(v)):
            if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and \
                    (v[i] - v[i - 1]) > self.voltage_increase_tol:
                change[i] = 1.0
        return change

    def _prev_a1_from_df(self, df):
        """Build the Prev_A1 channel for a whole dataframe (offline path)."""
        src = self.prev_a1_source
        if src in ("auto", "voltagechange") and VOLTAGECHANGE_COL in df.columns:
            return self._clean_voltagechange_array(df[VOLTAGECHANGE_COL])
        if src == "voltagechange":
            raise ValueError(
                f"prev_a1_source='voltagechange' but column "
                f"'{VOLTAGECHANGE_COL}' is missing from the CSV.")
        volt = self._voltage_series_from_df(df)
        derived = self._voltage_to_change_array(volt)
        if derived is not None:
            return derived
        if src == "voltage_delta":
            raise ValueError(
                "prev_a1_source='voltage_delta' but no voltage column "
                f"({VOLTAGE_COLS}) found in the CSV.")
        # Last resort: all-zero history (treats past as 'hold').
        print("    WARNING: no VoltageChange or voltage column found; "
              "using all-zero Prev_A1 history.")
        return np.zeros(len(df), dtype=float)

    def _compute_prev_a1_for_row(self, row_dict):
        """Per-row Prev_A1 for the streaming paths. Updates internal voltage
        state as a side effect when deriving from voltage."""
        src = self.prev_a1_source

        if src in ("auto", "voltagechange") and VOLTAGECHANGE_COL in row_dict:
            v = _to_float(row_dict.get(VOLTAGECHANGE_COL))
            if not np.isnan(v):
                if v == -1:
                    return 0.0
                return 1.0 if v == 1 else 0.0
            if src == "voltagechange":
                return 0.0
            # else fall through to voltage / history

        # Voltage delta
        if src in ("auto", "voltage_delta"):
            volt = np.nan
            for col in VOLTAGE_COLS:
                if col in row_dict:
                    volt = _to_float(row_dict.get(col))
                    break
            if not np.isnan(volt):
                pa = 0.0
                if self._prev_voltage is not None and \
                        (volt - self._prev_voltage) > self.voltage_increase_tol:
                    pa = 1.0
                self._prev_voltage = volt
                return pa
            if src == "voltage_delta":
                return 0.0

        # Decision history fallback
        return float(self._last_suggestion)

    # ------------------------------------------------------------------
    # MODE 1: offline_csv  (predict for a whole finished file)
    # ------------------------------------------------------------------
    def predict_csv(self, csv_path, output_csv=None):
        """Predict 0/1 for every valid window in a finished CSV.

        Returns a pandas DataFrame with one row per prediction. If
        ``output_csv`` is given, the SAME predictions are written there. The
        input CSV is never modified.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input CSV is missing required feature columns: {missing_cols}")

        # Build full (N,4) raw matrix, then scale sensors.
        sensors = np.stack(
            [pd.to_numeric(df[c], errors="coerce").values.astype(float)
             for c in FEATURE_COLS], axis=1)
        prev_a1 = self._prev_a1_from_df(df).astype(float).reshape(-1, 1)
        X_raw = np.concatenate([sensors, prev_a1], axis=1)  # (N, 4)

        n = len(X_raw)
        seq = self.sequence_length
        if n <= seq:
            print(f"File too short: {n} rows <= sequence_length {seq}. "
                  "No predictions produced.")
            empty = pd.DataFrame(columns=list(PredictionResult.__annotations__))
            if output_csv:
                empty.to_csv(output_csv, index=False)
            return empty

        # Scale sensors once for the whole file.
        nan_mask = np.isnan(X_raw[:, 0:3])
        sensor_filled = np.nan_to_num(X_raw[:, 0:3], nan=0.0)
        sensor_scaled = self.scaler.transform(sensor_filled)
        sensor_scaled[nan_mask] = self.missing_value
        X_scaled = X_raw.copy()
        X_scaled[:, 0:3] = sensor_scaled
        X_scaled[:, 3] = np.nan_to_num(X_scaled[:, 3], nan=0.0)

        # Windows: window i = rows [i, i+seq); prediction is for row i+seq.
        windows = np.stack([X_scaled[i:i + seq] for i in range(n - seq)], axis=0)
        probs = self._forward_prob_batch(windows)
        preds = (probs >= self.threshold).astype(int)

        ts = datetime.now().isoformat(timespec="seconds")
        rows = []
        has_truth = VOLTAGECHANGE_COL in df.columns
        truth = (self._clean_voltagechange_array(df[VOLTAGECHANGE_COL])
                 if has_truth else None)
        for k in range(len(probs)):
            src_idx = k + seq
            rec = {
                "timestamp": ts,
                "source_row_index": int(src_idx),
                "prob_increase": float(probs[k]),
                "threshold": float(self.threshold),
                "suggestion": int(preds[k]),
            }
            if has_truth:
                rec["actual_voltagechange"] = int(truth[src_idx])
            rows.append(rec)

        out_df = pd.DataFrame(rows)
        if has_truth and len(out_df) > 0:
            acc = float((out_df["suggestion"] == out_df["actual_voltagechange"]).mean())
            print(f">>> predict_csv: {len(out_df)} predictions | "
                  f"accuracy vs actual = {acc:.4f}")
        else:
            print(f">>> predict_csv: {len(out_df)} predictions "
                  f"(no actual VoltageChange column to score against)")

        if output_csv:
            out_df.to_csv(output_csv, index=False)
            print(f">>> Saved suggestions to: {output_csv}")
        return out_df

    # ------------------------------------------------------------------
    # MODE 2: live_csv  (tail a growing CSV)
    # ------------------------------------------------------------------
    def _ingest_row(self, row_dict):
        """Add one raw sample to the rolling buffer."""
        cur = _to_float(row_dict.get(FEATURE_COLS[0]))
        pres = _to_float(row_dict.get(FEATURE_COLS[1]))
        rad = _to_float(row_dict.get(FEATURE_COLS[2]))
        prev_a1 = self._compute_prev_a1_for_row(row_dict)
        self._buffer.append([cur, pres, rad, prev_a1])
        self._row_count += 1

    def _predict_from_buffer(self, source_index):
        """Predict from the current rolling buffer, or None if not full yet."""
        if len(self._buffer) < self.sequence_length:
            return None
        raw_window = np.array(self._buffer, dtype=float)
        scaled = self._scale_window(raw_window)
        prob = self._forward_prob(scaled)
        sugg = int(prob >= self.threshold)
        self._last_suggestion = float(sugg)
        return PredictionResult(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source_row_index=int(source_index),
            prob_increase=prob,
            threshold=float(self.threshold),
            suggestion=sugg,
        )

    def _init_live_reader(self, csv_path):
        """Read the header + all existing rows once, priming the buffer, and
        remember the byte offset so later calls only read new lines."""
        self._live_path = csv_path
        self._buffer.clear()
        self._prev_voltage = None
        self._last_suggestion = 0.0
        self._row_count = 0
        with open(csv_path, "r", newline="") as f:
            header_line = f.readline()
            if not header_line:
                self._live_fieldnames = None
                self._live_offset = f.tell()
                return
            self._live_fieldnames = next(csv.reader([header_line]))
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._ingest_row(self._parse_live_line(line))
            self._live_offset = f.tell()

    def _parse_live_line(self, line):
        values = next(csv.reader([line]))
        return dict(zip(self._live_fieldnames, values))

    def _read_new_live_rows(self, csv_path):
        """Read only rows appended since the last call. Returns count read."""
        new_count = 0
        with open(csv_path, "r", newline="") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size < self._live_offset:
                # File was truncated/rotated -> re-initialize from scratch.
                print("    NOTE: live CSV shrank; re-initializing reader.")
                self._init_live_reader(csv_path)
                return 0
            f.seek(self._live_offset)
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._ingest_row(self._parse_live_line(line))
                    new_count += 1
            self._live_offset = f.tell()
        return new_count

    def predict_latest_from_live_csv(self, csv_path):
        """Tail ``csv_path`` and return the newest PredictionResult (or None if
        there are not yet ``sequence_length`` rows). The model and scaler are
        NOT reloaded, and the file is read incrementally."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Live CSV not found: {csv_path}")
        if self._live_path != csv_path or self._live_fieldnames is None:
            self._init_live_reader(csv_path)
        else:
            self._read_new_live_rows(csv_path)
        if len(self._buffer) < self.sequence_length:
            return None
        # source index = index of the most recent row consumed (0-based).
        return self._predict_from_buffer(self._row_count - 1)

    # ------------------------------------------------------------------
    # MODE 3: direct_row  (client pushes rows in)
    # ------------------------------------------------------------------
    def append_row_and_predict(self, row):
        """Push ONE new sample (dict) from the EPICS/p4p client and get the
        newest suggestion. Returns None until ``sequence_length`` rows have
        been collected (warm-up).

        ``row`` keys should include the sensor columns
        (GunCurrent.Avg, peg-BL-cc:pressureM, RadiationTotal) and, depending on
        ``prev_a1_source``, a VoltageChange and/or a voltage column.
        """
        self._ingest_row(row)
        return self._predict_from_buffer(self._row_count - 1)

    def reset_stream(self):
        """Clear the rolling buffer / streaming state (both live modes)."""
        self._buffer.clear()
        self._prev_voltage = None
        self._last_suggestion = 0.0
        self._row_count = 0
        self._live_path = None
        self._live_fieldnames = None
        self._live_offset = 0

    # ------------------------------------------------------------------
    # Output helper (writes to a SEPARATE suggestions CSV)
    # ------------------------------------------------------------------
    @staticmethod
    def append_suggestion_to_csv(result, output_csv):
        """Append one PredictionResult to a separate suggestions CSV, writing a
        header if the file does not exist yet. Never touches the input CSV."""
        if result is None:
            return
        row = result.as_row()
        write_header = not os.path.exists(output_csv) or \
            os.path.getsize(output_csv) == 0
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ==========================================================================
# Command-line / standalone usage
# ==========================================================================
# Pick the mode here when running this file directly, OR pass --mode on the CLI.
MODE = "offline_csv"          # "offline_csv" | "live_csv" | "direct_row"

# Path to a TRAINED LSTM run's checkpoint (the folder should also contain
# val_threshold_selection.json).
CHECKPOINT_PATH = r"path\to\run_folder\full_checkpoint.pth"

# The EPICS-generated CSV (read-only input).
INPUT_CSV = r"path\to\live_epics_rows.csv"

# Where suggestions are written (separate file; safe to delete/rotate).
OUTPUT_CSV = r"lstm_suggestions.csv"

# How often to poll the live CSV (seconds), used only in live_csv mode.
LIVE_POLL_SECONDS = 1.0


def _run_offline(model, input_csv, output_csv):
    print(f"\n=== MODE: offline_csv ===\nInput : {input_csv}\nOutput: {output_csv}")
    model.predict_csv(input_csv, output_csv=output_csv)


def _run_live(model, input_csv, output_csv, poll_seconds):
    print(f"\n=== MODE: live_csv (tailing) ===\nInput : {input_csv}\n"
          f"Output: {output_csv}\nPolling every {poll_seconds}s. Ctrl+C to stop.")
    last_emitted = None
    try:
        while True:
            result = model.predict_latest_from_live_csv(input_csv)
            if result is not None and result.source_row_index != last_emitted:
                last_emitted = result.source_row_index
                model.append_suggestion_to_csv(result, output_csv)
                print(f"[row {result.source_row_index}] "
                      f"prob={result.prob_increase:.4f} "
                      f"t*={result.threshold:.4f} -> suggestion={result.suggestion}")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped live tailing.")


def _run_direct_demo(model, input_csv, output_csv):
    """Demonstration of the direct_row API by replaying INPUT_CSV row by row.

    In real use, replace the loop body with values pulled from your EPICS / p4p
    client instead of reading a CSV.
    """
    print(f"\n=== MODE: direct_row (demo replay of {input_csv}) ===\n"
          f"Output: {output_csv}")
    df = pd.read_csv(input_csv)
    for i, (_, row) in enumerate(df.iterrows()):
        result = model.append_row_and_predict(row.to_dict())
        if result is not None:
            model.append_suggestion_to_csv(result, output_csv)
            print(f"[row {i}] prob={result.prob_increase:.4f} "
                  f"-> suggestion={result.suggestion}")


def main():
    parser = argparse.ArgumentParser(
        description="LSTM inference adapter for electron-gun conditioning.")
    parser.add_argument("--mode", choices=["offline_csv", "live_csv", "direct_row"],
                        default=MODE)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (default: load from "
                             "val_threshold_selection.json).")
    parser.add_argument("--prev-a1-source", default="auto",
                        choices=["auto", "voltagechange", "voltage_delta",
                                 "decision_history"])
    parser.add_argument("--poll-seconds", type=float, default=LIVE_POLL_SECONDS)
    args = parser.parse_args()

    model = PolgunLSTMInference(
        args.checkpoint,
        threshold=args.threshold,
        prev_a1_source=args.prev_a1_source,
    )

    if args.mode == "offline_csv":
        _run_offline(model, args.input, args.output)
    elif args.mode == "live_csv":
        _run_live(model, args.input, args.output, args.poll_seconds)
    elif args.mode == "direct_row":
        _run_direct_demo(model, args.input, args.output)


if __name__ == "__main__":
    main()
