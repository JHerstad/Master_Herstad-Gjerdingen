"""
Module for preprocessing the Aachen battery degradation dataset for RUL classification or regression.
Handles data loading, EOL/RUL computation, binning, splitting, normalization, and encoding
for both classification (CNN) and regression (LSTM) models with a configurable sequence length.
"""

import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mat4py
from typing import Dict, Optional, Tuple
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config.defaults import Config  # Assuming Config is defined in config/defaults.py
import os
import json


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


def preprocess_aachen_dataset(
    data_path: str,
    test_cell_count: int = Config().test_cell_count,
    random_state: int = 42,
    log_transform: bool = False,
    classification: bool = False
) -> Dict:
    """
    Loads and preprocesses the Aachen dataset for RUL classification or regression,
    supporting both classification (CNN, fixed sequence length) and regression (LSTM, fixed-length
    padded sequences). Automatically stores preprocessed data in data/processed/ for reproducibility.

    Args:
        data_path (str): Path to the '.mat' file containing the dataset.
        test_cell_count (int): Number of unique cells to hold out for testing.
        random_state (int): Seed for random operations.
        log_transform (bool): Whether to apply log transform to RUL values for regression.
        classification (bool): If True, prepare data for classification (CNN) with fixed sequence length
                             from config; if False, prepare for regression (LSTM) with fixed sequence length.

    Returns:
        Dict: Preprocessed data including X_train, X_val, X_test, y_train, y_val, y_test,
              y_max, label_mapping (for classification), df_filtered, and max_sequence_length (for both).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Load dataset into a pandas DataFrame
    data_loader = mat4py.loadmat(data_path)
    df = pd.DataFrame.from_dict(data_loader["TDS"])

    # Compute EOL and RUL, filter invalid entries
    config = Config()
    eol_capacity = config.eol_capacity
    seq_len = config.seq_len

    df[["EOL", "RUL"]] = df.apply(compute_eol_and_rul, axis=1, fraction=eol_capacity)
    df_filtered = df.dropna(subset=["RUL"])

    # Filter sequences based on seq_len
    df_filtered = df_filtered[df_filtered["History"].apply(lambda x: len(x) >= seq_len)]

    # Define RUL bins and labels for classification only
    label_mapping = None
    if classification:
        bins = [0, 200, 300, 400, 500, 600, 700, np.inf]
        labels = ["0-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700+"]
        df_filtered["RUL_binned"] = pd.cut(df_filtered["RUL"], bins=bins, labels=labels, include_lowest=True)

        # Map RUL bins to integers
        label_mapping = {label: i for i, label in enumerate(labels)}
        df_filtered["RUL_binned_int"] = df_filtered["RUL_binned"].map(label_mapping)

    # Split data by cell into train/val/test
    # Hold back specific cells for testing
    cells_to_hold_back = df_filtered["Cell"].unique()[:test_cell_count]
    df_test = df_filtered[df_filtered["Cell"].isin(cells_to_hold_back)]
    df_train_val = df_filtered[~df_filtered["Cell"].isin(cells_to_hold_back)]

    # Split the remaining data into training and validation sets
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.2,
        random_state=random_state,
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
        if log_transform:
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

    # Prepare preprocessed data dictionary
    preprocessed_data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_norm,
        "y_val": y_val_norm,
        "y_test": y_test_norm,
        "y_max": y_max,
        "label_mapping": label_mapping if classification else None,
        "df_filtered": df_filtered,
        "max_sequence_length": X_train.shape[1] if not classification else seq_len
    }

    # Store preprocessed data in data/processed/, overwriting if EOL and model type match
    output_dir = "data/processed/"
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
        "max_sequence_length": int(X_train.shape[1]),
        "eol_capacity": float(eol_capacity),
        "classification": classification,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(os.path.join(output_dir, f"metadata_{model_type}_{eol_str}.json"), "w") as f:
        json.dump(metadata, f, indent=4)