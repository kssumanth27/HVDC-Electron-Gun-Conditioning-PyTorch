import os
# These must be set BEFORE importing torch or numpy
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
os.environ['KMP_WARNINGS'] = '0'
os.environ['MKL_CBWR'] = 'AUTO'
os.environ['MKL_WARN'] = '0'

import torch
import numpy as np
import pandas as pd

# Import components from the main script
# Ensure the filename matches your POL_CNN_or_LSTM_for_GIT_upload.py
from POL_CNN_or_LSTM_for_GIT_upload import (
    prepare_all_data, 
    train_model, 
    evaluate_test_set, 
    estimate_folder_noise_thresholds
)

def add_gaussian_noise(X, noise_level=0.1):
    """
    Adds Gaussian noise to the scaled feature values (indices 0, 2, 4).
    noise_level: fraction of the standard deviation to add as noise.
    """
    X_noisy = X.clone()
    # Features are [Current, Mask, Pressure, Mask, Rad, Mask, Prev_A1]
    # We only perturb the actual sensor values (0, 2, 4)
    for i in [0, 2, 4]:
        std = X_noisy[:, :, i].std()
        noise = torch.randn(X_noisy[:, :, i].shape) * noise_level * std
        X_noisy[:, :, i] += noise
    return X_noisy

def simulate_sensor_dropout(X, sensor_name):
    """
    Simulates total loss of a sensor signal by zeroing the value and its mask.
    """
    X_dropped = X.clone()
    mapping = {"GunCurrent.Avg": 0, "peg-BL-cc:pressureM": 2, "RadiationTotal": 4}
    if sensor_name not in mapping:
        return X_dropped
        
    idx = mapping[sensor_name]
    X_dropped[:, :, idx] = 0.0      # Zero out value
    X_dropped[:, :, idx + 1] = 0.0  # Zero out mask (telling the model data is missing)
    return X_dropped

def run_robustness_suite(models, fwd_fn, data, pos_weight):
    """
    Runs a series of evaluations to test model stability.
    """
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION SUITE")
    print("="*60)

    # 1. Baseline
    print("\n[TEST 1] Baseline Performance (Clean Test Set)")
    evaluate_test_set(models, fwd_fn, data, pos_weight)

    # 2. Noise Sensitivity
    # Testing how much sensor jitter affects the ramp-up detection
    for level in [0.10, 0.25, 0.50]:
        print(f"\n[TEST 2] Noise Sensitivity (Level: {level*100:.0f}% of StdDev)")
        noisy_data = {
            'X_test': add_gaussian_noise(data['X_test'], level),
            'y_test': data['y_test']
        }
        evaluate_test_set(models, fwd_fn, noisy_data, pos_weight)

    # 3. Sensor Dropout
    # Testing if the model can still predict using remaining sensors
    sensors = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
    for sensor in sensors:
        print(f"\n[TEST 3] Sensor Dropout: {sensor}")
        dropped_data = {
            'X_test': simulate_sensor_dropout(data['X_test'], sensor),
            'y_test': data['y_test']
        }
        evaluate_test_set(models, fwd_fn, dropped_data, pos_weight)

if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = "../data/polarized_gun/training"
    TEST_DIR = "../data/polarized_gun/testing"
    SEQ_LEN = 30
    FEATURE_COLS = ["GunCurrent.Avg", "peg-BL-cc:pressureM", "RadiationTotal"]
    
    # 1. Load and Prepare Data
    print("--- Preparing Data for Robustness Testing ---")
    noise_thresholds = estimate_folder_noise_thresholds(TRAIN_DIR, FEATURE_COLS)
    data, pos_weight, scaler = prepare_all_data(
        TRAIN_DIR, TEST_DIR, SEQ_LEN,
        noise_thresholds=noise_thresholds,
        filter_quiet_negatives=True
    )

    # 2. Train Model
    # We train a fresh model to ensure we are testing the current architecture's capacity
    print("\n--- Training Model ---")
    models, fwd_fn, history = train_model(
        'cnn', 
        data, 
        pos_weight, 
        epochs=40, 
        batch_size=128, 
        lr=0.00001
    )

    # 3. Run Robustness Tests
    run_robustness_suite(models, fwd_fn, data, pos_weight)