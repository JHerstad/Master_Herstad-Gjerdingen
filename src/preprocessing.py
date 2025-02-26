"""
Module for preprocessing the Aachen battery degradation dataset for RUL classification or regression.
Handles data loading, EOL/RUL computation, binning, splitting, normalization, encoding, and padding
for both classification (CNN) and regression (LSTM) models with a configurable sequence length.
"""

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from typing import Dict, Optional, Tuple
from config.defaults import Config  # Assuming Config is defined in config/defaults.py


def compute_eol_and_rul(row: pd.Series, fraction: float) -> pd.Series:
    """
    Computes End-of-Life (EOL) cycle and Remaining Useful Life (RUL) for a battery sequence
    based on a specified capacity threshold fraction of initial capacity.

    Args:
        row (pd.Series): DataFrame row containing 'History', 'History_Cycle',
                        'Target_expanded', and 'Target_Cycle_Expanded' columns as lists or arrays.
        fraction (float): Fraction of initial capacity defining EOL (e.g., 0.65 for 65%).

    Returns:
        pd.Series: Dictionary-like series with 'EOL' and 'RUL' values (NaN if invalid).
    """
    history_cap = np.array(row["History"])
    history_cycles = np.array(row["History_Cycle"])
    target_cap = np.array(row["Target_expanded"])
    target_cycles = np.array(row["Target_Cycle_Expanded"])

    # Handle edge cases: empty or missing data
    if len(history_cap) == 0 or len(history_cycles) == 0:
        return pd.Series({"EOL": np.nan, "RUL": np.nan})

    # Calculate EOL threshold as fraction of initial capacity
    initial_capacity = history_cap[0]
    threshold = fraction * initial_capacity

    # Check if last historical capacity is already below threshold (no future RUL possible)
    if history_cap[-1] <= threshold:
        return pd.Series({"EOL": np.nan, "RUL": np.nan})

    # Handle target data edge cases
    if len(target_cap) == 0 or len(target_cycles) == 0:
        return pd.Series({"EOL": np.nan, "RUL": np.nan})

    # Find the first index in target capacity where it drops below the threshold
    below_threshold_indices = np.where(target_cap < threshold)[0]
    eol_cycle = np.nan
    if len(below_threshold_indices) > 0:
        eol_index = below_threshold_indices[0]
        eol_cycle = target_cycles[eol_index]

    # Compute RUL as EOL cycle minus last historical cycle
    rul = np.nan
    if not pd.isna(eol_cycle):
        last_history_cycle = history_cycles[-1]
        rul = eol_cycle - last_history_cycle

    return pd.Series({"EOL": eol_cycle, "RUL": rul})


def truncate_sequence(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Truncates a sequence to the specified length, taking the last seq_len time steps.

    Args:
        seq (np.ndarray): Input sequence (e.g., capacity values).
        seq_len (int): Length of sequences to truncate to.

    Returns:
        np.ndarray: Sequence truncated to the last seq_len steps.
    """
    return seq[-seq_len:]


def prepare_data_classification(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for training a classification model (CNN) by filtering sequences, truncating
    to a fixed length, stacking into a NumPy array, and reshaping for input.

    Args:
        df (pd.DataFrame): DataFrame with 'History', 'RUL_binned_int' columns.
        seq_len (int): Length of sequences to truncate to.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (shape: (samples, seq_len, 1)), y (shape: (samples,)).
    """
    # Filter rows with sufficient length and create a copy
    df_filtered = df[df["History"].apply(lambda x: len(x) >= seq_len)].copy()

    # Truncate history to the last seq_len steps
    df_filtered[f"History_{seq_len}"] = df_filtered["History"].apply(lambda x: truncate_sequence(x, seq_len))

    # Stack sequences into a NumPy array and reshape for classification (add channel dimension)
    X = np.stack(df_filtered[f"History_{seq_len}"].values, axis=0)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Extract labels
    y = df_filtered["RUL_binned_int"].values

    return X, y


def prepare_data_regression(df: pd.DataFrame, seq_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for training a regression model (LSTM) by extracting sequences and labels,
    preserving variable sequence lengths for padding, optionally truncating to seq_len.

    Args:
        df (pd.DataFrame): DataFrame with 'History', 'RUL_binned_int' columns.
        seq_len (Optional[int]): Length of sequences to truncate to, or None for maximum length.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (shape: (samples, max_seq_len, 1)), y (shape: (samples,)).
    """
    # Extract sequences and labels
    histories = df["History"].tolist()
    y = df["RUL_binned_int"].values

    # Truncate or keep full sequences based on seq_len
    if seq_len is not None:
        histories = [truncate_sequence(h, seq_len) for h in histories]

    # Flatten all histories for scaling
    all_histories_flat = np.concatenate(histories)
    scaler = MinMaxScaler()
    scaler.fit(all_histories_flat.reshape(-1, 1))

    # Normalize sequences
    normalized_histories = [
        scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in histories
    ]

    # Pad sequences to the maximum length or specified seq_len
    max_sequence_length = max(len(h) for h in normalized_histories) if seq_len is None else seq_len
    X_padded = pad_sequences(
        normalized_histories,
        maxlen=max_sequence_length,
        padding="post",
        dtype="float32"
    )

    # Reshape for regression input (add channel dimension)
    X_lstm = X_padded[..., np.newaxis]

    return X_lstm, y


def preprocess_aachen_dataset(
    data_path: str,
    eol_capacity: float = 0.65,
    test_cell_count: int = 3,
    random_state: int = 42,
    log_transform: bool = False,
    classification: bool = False  # True for classification (CNN), False for regression (LSTM, default)
) -> Dict:
    """
    Loads and preprocesses the Aachen battery degradation dataset for RUL classification or regression,
    supporting both classification (CNN, fixed sequence length) and regression (LSTM, variable-length
    padded sequences).

    Args:
        data_path (str): Path to the '.mat' file containing the dataset.
        eol_capacity (float): Fraction of initial capacity defining EOL (e.g., 0.65 for 65%).
        test_cell_count (int): Number of unique cells to hold out for testing.
        random_state (int): Seed for random operations.
        log_transform (bool): Whether to apply log transform to RUL values for regression.
        classification (bool): If True, prepare data for classification (CNN) with fixed sequence length
                             from config; if False, prepare for regression (LSTM) with variable length.

    Returns:
        Dict: Preprocessed data including X_train, X_val, X_test, y_train, y_val, y_test,
              y_max, label_mapping (for classification), df_filtered, and max_sequence_length (for regression).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Load dataset into a pandas DataFrame
    data_loader = mpy.loadmat(data_path)
    df = pd.DataFrame.from_dict(data_loader["TDS"])

    # Compute EOL and RUL, filter invalid entries
    df[["EOL", "RUL"]] = df.apply(compute_eol_and_rul, axis=1)
    df_filtered = df.dropna(subset=["RUL"])

    # Filter sequences with sufficient length (minimum seq_len for classification, any for regression)
    config = Config()  # Access config for sequence length
    seq_len = config.seq_len  # Use configurable sequence length from config
    min_seq_len = seq_len if classification else 0
    df_filtered = df_filtered[df_filtered["History"].apply(lambda x: len(x) >= min_seq_len)]

    # Define RUL bins and labels for classification only
    if classification:
        bins = [0, 200, 300, 400, 500, 600, 700, np.inf]
        labels = ["0-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700+"]
        df_filtered["RUL_binned"] = pd.cut(df_filtered["RUL"], bins=bins, labels=labels, include_lowest=True)

        # Map RUL bins to integers
        label_mapping = {label: i for i, label in enumerate(labels)}
        df_filtered["RUL_binned_int"] = df_filtered["RUL_binned"].map(label_mapping)

    # Split data by cell into train/val/test
    unique_cells = df_filtered["Cell"].unique()
    np.random.shuffle(unique_cells)

    split_index = int(len(unique_cells) * config.train_split_ratio)
    train_cells = unique_cells[:split_index]
    test_cells = unique_cells[split_index:]

    df_train = df_filtered[df_filtered["Cell"].isin(train_cells)]
    df_test = df_filtered[df_filtered["Cell"].isin(test_cells)]

    # Further split train into train and validation
    val_index = int(len(train_cells) * (1 - config.val_split_ratio))
    val_cells = train_cells[val_index:]
    train_cells = train_cells[:val_index]

    df_val = df_filtered[df_filtered["Cell"].isin(val_cells)]

    # Prepare data based on model type
    if classification:
        X_train, y_train = prepare_data_classification(df_train, seq_len)
        X_val, y_val = prepare_data_classification(df_val, seq_len)
        X_test, y_test = prepare_data_classification(df_test, seq_len)
    else:  # regression
        X_train, y_train = prepare_data_regression(df_train, seq_len)
        X_val, y_val = prepare_data_regression(df_val, seq_len)
        X_test, y_test = prepare_data_regression(df_test, seq_len)

    # Normalize sequences using MinMaxScaler fitted on training data
    scaler = MinMaxScaler()
    if classification:
        X_train_2d = X_train.reshape(-1, 1)
    else:
        X_train_2d = X_train.reshape(-1, 1)  # Flatten for scaling, preserving variable length
    scaler.fit(X_train_2d)

    # Transform and reshape sequences
    if classification:
        X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    else:
        X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Handle labels based on model type (classification for classification, regression for regression)
    if classification:
        # One-hot encode labels for classification
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

    return {
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