"""
Module for preprocessing the Aachen and MIT_Stanford battery degradation dataset for RUL classification or regression.
Handles data loading, EOL/RUL computation, binning, splitting, normalization, and encoding
for both classification (CNN) and regression (LSTM) models with a configurable sequence length.
"""

import os
import json
import pickle
import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
import mat4py

from config.defaults import Config  # Assuming Config is defined in config/defaults.py



def compute_eol_and_rul(row: pd.Series, fraction: float) -> pd.Series:
    """
    Computes End-of-Life (EOL) cycle and Remaining Useful Life (RUL) for a battery sequence
    based on a specified capacity threshold fraction of initial capacity.

    Args:
        row (pd.Series): Row containing 'History', 'History_Cycle', 'Target_expanded',
                        and 'Target_Cycle_Expanded' as lists or arrays.
        fraction (float): Fraction of initial capacity defining EOL (e.g., 0.65 for 65%).

    Returns:
        pd.Series: A series containing 'EOL' (cycle number of failure) and 'RUL' (cycles remaining).
    """
    history_cap = np.array(row["History"])
    history_cycles = np.array(row["History_Cycle"])
    target_cap = np.array(row["Target_expanded"])
    target_cycles = np.array(row["Target_Cycle_Expanded"])

    # Handle missing or empty data
    if not history_cap.size or not history_cycles.size:
        return pd.Series({"EOL": np.nan, "RUL": np.nan})

    # Define EOL as the first cycle where capacity drops below the threshold
    initial_capacity = history_cap[0]
    threshold = fraction * initial_capacity

    # If last known capacity is already below threshold, RUL is undefined
    if history_cap[-1] <= threshold:
        return pd.Series({"EOL": np.nan, "RUL": np.nan})

    if not target_cap.size or not target_cycles.size:
        return pd.Series({"EOL": np.nan, "RUL": np.nan})

    # Find EOL cycle (first occurrence below threshold)
    below_threshold_indices = np.where(target_cap < threshold)[0]
    eol_cycle = target_cycles[below_threshold_indices[0]] if below_threshold_indices.size else np.nan

    # Compute RUL as the difference between EOL cycle and the last recorded cycle
    rul = eol_cycle - history_cycles[-1] if not np.isnan(eol_cycle) else np.nan

    return pd.Series({"EOL": eol_cycle, "RUL": rul})


def process_sequences(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    Truncates and normalizes battery capacity sequences to a fixed length.

    Args:
        df (pd.DataFrame): DataFrame containing a 'History' column with capacity degradation sequences.
        seq_len (int): Length to truncate each sequence to.

    Returns:
        np.ndarray: Array of normalized and truncated sequences, shape (num_samples, seq_len).
    """
    # Filter rows with sufficient length and create a copy
    df_filtered = df[df["History"].apply(lambda x: len(x) >= seq_len)].copy()

    # Truncate sequences to the last seq_len steps
    df_filtered["History_truncated"] = df_filtered["History"].apply(lambda x: x[-seq_len:])

    # Stack sequences into a NumPy array
    sequences = np.stack(df_filtered["History_truncated"].values, axis=0)

    # Normalize sequences using MinMaxScaler fitted on all sequences
    scaler = MinMaxScaler()
    sequences_normalized = scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)

    return sequences_normalized


def prepare_data_classification(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares battery degradation data for classification, including normalization.

    Args:
        df (pd.DataFrame): Filtered DataFrame containing capacity history and RUL bin labels.
        seq_len (int): Sequence length for truncation and normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (shape: (samples, seq_len, 1)), y (shape: (samples,)),
                                    where X is normalized input sequences and y is one-hot encoded
                                    classification labels.
    """
    # Process and normalize sequences
    X = process_sequences(df, seq_len).reshape(-1, seq_len, 1)  # Reshape for CNN input
    y = df["RUL_binned_int"].values  # Classification labels
    return X, y


def prepare_data_regression(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares battery degradation data for regression, including normalization.

    Args:
        df (pd.DataFrame): Filtered DataFrame containing capacity history and RUL values.
        seq_len (int): Sequence length for truncation and normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (shape: (samples, seq_len, 1)), y (shape: (samples,)),
                                    where X is normalized input sequences and y is the corresponding
                                    RUL values.
    """
    # Process and normalize sequences
    X = process_sequences(df, seq_len).reshape(-1, seq_len, 1)  # Reshape for LSTM input
    y = df["RUL"].values  # Regression labels
    return X, y


def preprocess_aachen_dataset(config: Config) -> None:
    """
    Loads and preprocesses the Aachen dataset for RUL classification or regression,
    supporting both classification (CNN) and regression (LSTM) tasks with fixed-length sequences.
    Saves preprocessed data arrays and metadata to data/processed/ for reproducibility.

    Args:
        config (Config): Configuration object with preprocessing parameters:
            - data_path (str): Path to the '.mat' file containing the dataset.
            - test_cell_count (int): Number of unique cells to reserve for testing.
            - random_state (int): Seed for reproducible train/validation splitting.
            - log_transform (bool): If True, applies log transform to RUL values (regression only).
            - classification (bool): If True, prepares data for classification with binned RUL;
                                   if False, prepares data for regression with continuous RUL.
            - eol_capacity (float): Fraction of initial capacity defining EOL (e.g., 0.65).
            - seq_len (int): Length to which capacity sequences are truncated.

    Returns:
        None: Does not return data directly; instead, saves:
            - X_train, X_val, X_test (np.ndarray): Normalized sequences to .npy files.
            - y_train, y_val, y_test (np.ndarray): Targets (one-hot for classification, normalized RUL for regression) to .npy files.
            - Metadata (dict): Includes y_max, seq_len, eol_capacity, classification, and timestamp, saved as JSON.
    """
    classification = config.classification
    eol_capacity = config.eol_capacity
    seq_len = config.seq_len
    bins = config.bins
    labels = config.labels
    
    # Load dataset into a pandas DataFrame using mat4py. Path from Config
    data_loader = mat4py.loadmat(config.data_path)
    df = pd.DataFrame.from_dict(data_loader["TDS"])

    # Compute EOL and RUL for each sequence based on EOL capacity
    df[["EOL", "RUL"]] = df.apply(compute_eol_and_rul, axis=1, fraction=eol_capacity)
    df_filtered = df.dropna(subset=["RUL"])

    # Filter sequences based on seq_len
    df_filtered = df_filtered[df_filtered["History"].apply(lambda x: len(x) >= seq_len)]

    # Define RUL bins and labels for classification using config values
    label_mapping = None


    if classification:
        if len(bins) - 1 != len(labels):
            raise ValueError(f"Number of bins ({len(bins)}) must match number of labels + 1 ({len(labels)})")
        df_filtered["RUL_binned"] = pd.cut(df_filtered["RUL"], bins=bins, labels=labels, include_lowest=True)
        label_mapping = {label: i for i, label in enumerate(labels)}
        df_filtered["RUL_binned_int"] = df_filtered["RUL_binned"].map(label_mapping)

    # Split data by cell into train/val/test
    # Hold back specific cells for testing
    cells_to_hold_back = df_filtered["Cell"].unique()[:config.test_cell_count]
    df_test = df_filtered[df_filtered["Cell"].isin(cells_to_hold_back)]
    df_train_val = df_filtered[~df_filtered["Cell"].isin(cells_to_hold_back)]

    # Split the remaining data into training and validation sets
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=config.val_split_ratio,
        random_state=config.random_state,
        stratify=df_train_val["Cell"]
    )

    # Prepare data based on model type
    if classification:
        X_train, y_train = prepare_data_classification(df_train, seq_len)
        X_val, y_val = prepare_data_classification(df_val, seq_len)
        X_test, y_test = prepare_data_classification(df_test, seq_len)
    else:  # regression
        X_train, y_train = prepare_data_regression(df_train, seq_len)
        X_val, y_val = prepare_data_regression(df_val, seq_len)
        X_test, y_test = prepare_data_regression(df_test, seq_len)

    # Apply one-hot encoding only if classification
    if classification:
        num_classes = len(label_mapping)
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)
    else:
        # Apply log transform to RUL if specified
        if config.log_transform:
            y_train = np.log1p(y_train)  # Use log1p for robustness
            y_val = np.log1p(y_val)
            y_test = np.log1p(y_test)

    # Normalize target values (post-log if log_transform=True for regression)
    y_max = y_train.max() if len(y_train) > 0 and not classification else df_filtered["RUL"].max()
    if not classification:
        y_train_norm = y_train / y_max
        y_val_norm = y_val / y_max
        y_test_norm = y_test / y_max
    else:
        y_train_norm, y_val_norm, y_test_norm = y_train, y_val, y_test  # Already one-hot for classification

    
    # Store preprocessed data in data/processed/, overwriting if EOL and model type match
    output_dir = "data/Aachen/processed/"
    os.makedirs(output_dir, exist_ok=True)

    # Define base filename with EOL capacity and model type
    eol_str = f"eol{int(eol_capacity*100)}"
    model_type = "classification" if classification else "regression"

    # Check for existing files with the same EOL and model type, remove if present
    existing_data_files = [f for f in os.listdir(output_dir) if f.startswith(f"X_train_{model_type}_{eol_str}_") and f.endswith(".npy")]
    existing_metadata_files = [f for f in os.listdir(output_dir) if f.startswith(f"metadata_{model_type}_{eol_str}_") and f.endswith(".json")]

    # Remove existing data files if present
    if existing_data_files:
        for file in existing_data_files:
            for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
                old_file = os.path.join(output_dir, file.replace("X_train", key))
                if os.path.exists(old_file):
                    os.remove(old_file)

    # Remove existing metadata file if present
    if existing_metadata_files:
        for file in existing_metadata_files:
            metadata_file = os.path.join(output_dir, file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)

    # Store preprocessed data arrays, overwriting with the same parameters (no timestamp)
    # X_train, X_val, X_test: Normalized input sequences for model training
    np.save(os.path.join(output_dir, f"X_train_{model_type}_{eol_str}.npy"), X_train)
    np.save(os.path.join(output_dir, f"X_val_{model_type}_{eol_str}.npy"), X_val)
    np.save(os.path.join(output_dir, f"X_test_{model_type}_{eol_str}.npy"), X_test)

    # y_train, y_val, y_test: Normalized target RUL values for regression, or one-hot encoded for classification
    np.save(os.path.join(output_dir, f"y_train_{model_type}_{eol_str}.npy"), y_train_norm)
    np.save(os.path.join(output_dir, f"y_val_{model_type}_{eol_str}.npy"), y_val_norm)
    np.save(os.path.join(output_dir, f"y_test_{model_type}_{eol_str}.npy"), y_test_norm)

    # Store metadata in a separate JSON file for reproducibility, focusing on y_max as most relevant
    metadata = {
        "y_max": float(y_max),  # Convert to Python float for JSON serialization
        "seq_len": int(X_train.shape[1]),
        "eol_capacity": float(eol_capacity),
        "classification": classification,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(os.path.join(output_dir, f"metadata_{model_type}_{eol_str}.json"), "w") as f:
        json.dump(metadata, f, indent=4)



def clean_outliers_fixed_limits(seq, min_cap=0.5, max_cap=1.2):
    """
    Replaces extreme capacity values with NaN and imputes missing values.
    """
    seq = np.array(seq).squeeze()
    
    # Replace values outside limits with NaN
    seq[(seq < min_cap) | (seq > max_cap)] = np.nan

    # Impute missing values using linear interpolation + forward/backward fill
    seq = pd.Series(seq).interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill').values
    return seq


def preprocess_mit_stanford_dataset(config: Config) -> None:
    """
    Loads and preprocesses the MIT-Stanford dataset for RUL classification or regression,
    supporting both classification (CNN) and regression (LSTM) tasks with fixed-length sequences.
    Saves preprocessed data arrays and metadata to data/MIT_Stanford/processed/ for reproducibility.

    Args:
        config (Config): Configuration object with preprocessing parameters.
    """

    # -------- Define Paths --------
    raw_data_dir = "data/MIT_Stanford/raw/"
    output_dir = "data/MIT_Stanford/processed/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # -------- Load MIT-Stanford Dataset --------
    batch_files = ["batch1.pkl", "batch2.pkl", "batch3.pkl"]
    bat_dict = {}

    for batch_file in batch_files:
        batch_path = os.path.join(raw_data_dir, batch_file)
        if os.path.exists(batch_path):
            with open(batch_path, "rb") as f:
                batch_data = pickle.load(f)
                bat_dict.update(batch_data)
        else:
            print(f"Warning: {batch_path} not found!")

    # Remove non-usable cells
    for cell in ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22', 'b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']:
        bat_dict.pop(cell, None)

    # -------- Configuration --------
    classification = config.classification
    eol_capacity = config.eol_capacity
    seq_len = config.seq_len  # Should be 20
    threshold_fraction = eol_capacity
    step_size = 5
    min_cap, max_cap = 0.5, 1.2
    max_actual_cycles = 100  # Only keep last 100 real cycles before downsampling

    # -------- Step 1: Generate Sequences and Compute RUL --------
    exp_windows, exp_ruls, exp_cell_ids = [], [], []

    for cell_key, cell_data in bat_dict.items():
        cycles = np.array(cell_data['summary']['cycle'])
        capacities = np.array(cell_data['summary']['QD'])

        # Clean capacity values
        capacities = clean_outliers_fixed_limits(capacities, min_cap, max_cap)

        # Compute EOL threshold
        initial_capacity = capacities[0]
        threshold = threshold_fraction * initial_capacity
        eol_cycle = cycles[np.argmax(capacities < threshold)] if np.any(capacities < threshold) else cycles[-1]

        # Expand and sample windows
        for end_idx in range(step_size, len(cycles), step_size):
            window_cycles = cycles[:end_idx + 1]
            window_caps = capacities[:end_idx + 1]

            # Truncate to last 100 real cycles
            if window_cycles[-1] - window_cycles[0] > max_actual_cycles:
                start_idx = np.argmax(window_cycles > (window_cycles[-1] - max_actual_cycles))
                window_cycles = window_cycles[start_idx:]
                window_caps = window_caps[start_idx:]

            # Downsample: keep every 5th
            window_cycles = window_cycles[::5]
            window_caps = window_caps[::5]

            if len(window_caps) < seq_len:
                continue  # Skip if still too short

            rul = eol_cycle - window_cycles[-1]
            if rul < 0:
                continue

            exp_windows.append(window_caps[-seq_len:])
            exp_ruls.append(rul)
            exp_cell_ids.append(cell_key)

    # Convert to arrays
    X_expanding = np.array(exp_windows)
    y_expanding = np.array(exp_ruls)

    # -------- Step 2: Normalize Input --------
    scaler_X = RobustScaler()
    X_exp_norm = scaler_X.fit_transform(X_expanding.reshape(-1, 1)).reshape(X_expanding.shape[0], seq_len, 1)

    # -------- Step 3: Train/Test Split by Cell --------
    unique_cells = np.unique(exp_cell_ids)
    train_cells, test_cells = train_test_split(
        unique_cells,
        test_size=config.test_cell_count / len(unique_cells),
        random_state=config.random_state
    )

    train_idx = [i for i, cell in enumerate(exp_cell_ids) if cell in train_cells]
    test_idx = [i for i, cell in enumerate(exp_cell_ids) if cell in test_cells]

    X_train, y_train = X_exp_norm[train_idx], y_expanding[train_idx]
    X_test, y_test = X_exp_norm[test_idx], y_expanding[test_idx]

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config.val_split_ratio,
        random_state=config.random_state
    )

    # -------- Step 4: Label Processing --------
    if classification:
        bins = config.bins
        labels = config.labels

        if len(bins) - 1 != len(labels):
            raise ValueError("Number of bins must be one more than number of labels.")

        cat_y_train = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
        cat_y_val = pd.cut(y_val, bins=bins, labels=labels, include_lowest=True)
        cat_y_test = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

        label_mapping = {label: i for i, label in enumerate(labels)}

        y_train = to_categorical(cat_y_train.map(label_mapping).astype(int), num_classes=len(labels))
        y_val = to_categorical(cat_y_val.map(label_mapping).astype(int), num_classes=len(labels))
        y_test = to_categorical(cat_y_test.map(label_mapping).astype(int), num_classes=len(labels))

    else:
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # -------- Step 5: Save Data --------
    eol_str = f"eol{int(eol_capacity * 100)}"
    model_type = "classification" if classification else "regression"

    for key, data in zip(["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                         [X_train, X_val, X_test, y_train, y_val, y_test]):
        np.save(os.path.join(output_dir, f"{key}_{model_type}_{eol_str}.npy"), data)

    metadata = {
        "y_max": float(np.max(y_expanding)),
        "seq_len": int(seq_len),
        "eol_capacity": float(eol_capacity),
        "classification": classification,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    with open(os.path.join(output_dir, f"metadata_{model_type}_{eol_str}.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Preprocessing complete! Data saved in {output_dir}")
