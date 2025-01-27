import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mat4py as mpy

def preprocess_aachen_dataset(
    file_path,
    test_cell_count=3,
    random_state=42,
    phase=None,
    early_threshold=800,
    mid_threshold=400,
    log_transform=False
):
    """
    Preprocess the Aachen dataset for LSTM training, with optional phase filtering
    and optional log transform of the target values (RUL80).
    
    Parameters:
    ----------
    file_path : str
        Path to the .mat file containing the dataset.
    test_cell_count : int, optional
        Number of unique cells to hold out for testing, by default 3.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    phase : str or None, optional
        Life phase to filter the data. Can be one of {"early", "mid", "late"} or None.
        - "early": keep samples with RUL80 > early_threshold
        - "mid":   keep samples where mid_threshold < RUL80 <= early_threshold
        - "late":  keep samples with RUL80 <= mid_threshold
        If None, no phase-based filtering is applied.
    early_threshold : float, optional
        Upper RUL80 threshold used to define the "early" vs "mid" boundary, by default 800.
    mid_threshold : float, optional
        Lower RUL80 threshold used to define the "mid" vs "late" boundary, by default 400.
    log_transform : bool, optional
        Whether to apply a log transform to the RUL80 values before normalization,
        by default False.

    Returns:
    -------
    dict
        A dictionary containing preprocessed training, validation, and testing datasets:
        - X_train, X_val, X_test: Padded sequences for LSTM input
        - y_train, y_val, y_test: Target values after optional log transform and normalization
        - y_max: The (log-transformed) maximum value in the training set if log_transform=False,
                 otherwise it is the max of the log scale.
        - max_sequence_length: Maximum sequence length used for padding
        - log_transform: Boolean indicating if log transform was applied (for downstream usage)
    """

    # --- 1) Load and transform the raw data from .mat into a Pandas DataFrame ---
    data_loader = mpy.loadmat(file_path)
    df = pd.DataFrame.from_dict(data_loader["TDS"])
    
    def compute_eol_and_rul80(row):
        """
        Compute EOL80 (End of Life where capacity drops below 80% of initial)
        and RUL80 (Remaining Useful Life at 80% capacity).
        """
        history_cap = np.array(row["History"])
        history_cycles = np.array(row["History_Cycle"])
        target_cap = np.array(row["Target_expanded"])
        target_cycles = np.array(row["Target_Cycle_Expanded"])

        eol80_cycle = np.nan
        rul80 = np.nan

        # Handle cases with missing data in historical capacity or cycles
        if len(history_cap) == 0 or len(history_cycles) == 0:
            return pd.Series({"EOL80": eol80_cycle, "RUL80": rul80})

        # Determine the threshold for EOL80
        initial_capacity = history_cap[0]
        threshold = 0.8 * initial_capacity

        # Check if the historical capacity already falls below the threshold
        if history_cap[-1] <= threshold:
            return pd.Series({"EOL80": np.nan, "RUL80": np.nan})

        # Handle cases with missing target data
        if len(target_cap) == 0 or len(target_cycles) == 0:
            return pd.Series({"EOL80": eol80_cycle, "RUL80": rul80})

        # Find the first cycle where capacity drops below the threshold in the target portion
        below_threshold_indices = np.where(target_cap < threshold)[0]
        if len(below_threshold_indices) > 0:
            eol80_index = below_threshold_indices[0]
            eol80_cycle = target_cycles[eol80_index]

        # Calculate RUL80 as the difference between EOL80 and the last history cycle
        if not pd.isna(eol80_cycle):
            last_history_cycle = history_cycles[-1]
            rul80 = eol80_cycle - last_history_cycle

        return pd.Series({"EOL80": eol80_cycle, "RUL80": rul80})

    # Compute EOL80 and RUL80 for each row
    df[["EOL80", "RUL80"]] = df.apply(compute_eol_and_rul80, axis=1)

    # Filter out rows with invalid RUL80 values (NaN, etc.)
    df_valid = df[df["RUL80"].notna()]

    # --- 2) Hold back specific cells for testing ---
    cells_to_hold_back = df_valid["Cell"].unique()[:test_cell_count]
    df_test = df_valid[df_valid["Cell"].isin(cells_to_hold_back)]
    df_train_val = df_valid[~df_valid["Cell"].isin(cells_to_hold_back)]

    # --- 3) Split the remaining data into training and validation sets ---
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.2,
        random_state=random_state,
        stratify=df_train_val["Cell"]
    )

    # --- 4) (OPTIONAL) Phase-based filtering for train, val, test ---
    def filter_by_phase(df_in, phase_name, early_thr, mid_thr):
        if phase_name == "early":
            # Keep samples with RUL80 > early_threshold
            return df_in[df_in["RUL80"] > early_thr]
        elif phase_name == "mid":
            # Keep samples where mid_threshold < RUL80 <= early_threshold
            return df_in[(df_in["RUL80"] > mid_thr) & (df_in["RUL80"] <= early_thr)]
        elif phase_name == "late":
            # Keep samples with RUL80 <= mid_thr
            return df_in[df_in["RUL80"] <= mid_thr]
        else:
            # If no phase is specified or phase is None, do not filter
            return df_in
    
    if phase is not None:
        df_train = filter_by_phase(df_train, phase, early_threshold, mid_threshold)
        df_val = filter_by_phase(df_val, phase, early_threshold, mid_threshold)
        df_test = filter_by_phase(df_test, phase, early_threshold, mid_threshold)

    # --- 5) Extract sequences (X) and target RUL80 (y) ---
    history_train = df_train["History"].tolist()
    history_val = df_val["History"].tolist()
    history_test = df_test["History"].tolist()

    y_train = np.array(df_train["RUL80"])
    y_val = np.array(df_val["RUL80"])
    y_test = np.array(df_test["RUL80"])

    # --- 5.1) (OPTIONAL) Log transform the target ---
    # Make sure RUL is positive. If there are zeros or near-zero values, consider log1p.
    if log_transform:
        # You can use np.log1p if you're worried about zero or negative values:
        # y_train, y_val, y_test = np.log1p(y_train), np.log1p(y_val), np.log1p(y_test)
        y_train, y_val, y_test = np.log(y_train), np.log(y_val), np.log(y_test)

    # --- 6) Normalize historical capacity sequences ---
    # Flatten all histories to fit one MinMaxScaler
    all_histories = history_train + history_val + history_test
    if len(all_histories) == 0:
        raise ValueError(
            f"No data left after applying phase='{phase}'. "
            "Try a different threshold or remove the phase filter."
        )

    all_histories_flat = np.concatenate(all_histories)
    scaler = MinMaxScaler()
    scaler.fit(all_histories_flat.reshape(-1, 1))

    history_train_normalized = [
        scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in history_train
    ]
    history_val_normalized = [
        scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in history_val
    ]
    history_test_normalized = [
        scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in history_test
    ]

    # --- 7) Normalize target values (post-log if log_transform=True) ---
    # We'll still scale by dividing by max from the training set (but note it's now log-scale if log_transform=True).
    y_max = y_train.max() if len(y_train) > 0 else 1.0  # Avoid divide-by-zero

    y_train_norm = y_train / y_max
    y_val_norm = y_val / y_max
    y_test_norm = y_test / y_max

    # --- 8) Pad sequences to the maximum sequence length ---
    max_sequence_length = max(len(h) for h in all_histories)
    X_train_padded = pad_sequences(
        history_train_normalized,
        maxlen=max_sequence_length,
        padding='post',
        dtype='float32'
    )
    X_val_padded = pad_sequences(
        history_val_normalized,
        maxlen=max_sequence_length,
        padding='post',
        dtype='float32'
    )
    X_test_padded = pad_sequences(
        history_test_normalized,
        maxlen=max_sequence_length,
        padding='post',
        dtype='float32'
    )

    # --- 9) Reshape for LSTM input (samples, time steps, features) ---
    X_train_lstm = X_train_padded[..., np.newaxis]
    X_val_lstm = X_val_padded[..., np.newaxis]
    X_test_lstm = X_test_padded[..., np.newaxis]

    return {
        "X_train": X_train_lstm,
        "X_val": X_val_lstm,
        "X_test": X_test_lstm,
        "y_train": y_train_norm,
        "y_val": y_val_norm,
        "y_test": y_test_norm,
        "y_max": y_max,  # This is max of the (possibly log-transformed) training data
        "max_sequence_length": max_sequence_length,
        "log_transform": log_transform
    }

# --- Example usage ---
if __name__ == "__main__":
    # Full dataset, no log transform
    preprocessed_full = preprocess_aachen_dataset(
        "Aachen/Degradation_Prediction_Dataset_ISEA.mat",
        test_cell_count=3,
        random_state=42,
        phase=None,
        log_transform=False
    )
    print("Shapes with log_transform=False:")
    print("  X_train:", preprocessed_full["X_train"].shape)
    print("  X_val:  ", preprocessed_full["X_val"].shape)
    print("  X_test: ", preprocessed_full["X_test"].shape)

    # Full dataset, with log transform
    preprocessed_log = preprocess_aachen_dataset(
        "Aachen/Degradation_Prediction_Dataset_ISEA.mat",
        test_cell_count=3,
        random_state=42,
        phase=None,
        log_transform=True  # <--- apply log to the RUL values
    )
    print("Shapes with log_transform=True:")
    print("  X_train:", preprocessed_log["X_train"].shape)
    print("  X_val:  ", preprocessed_log["X_val"].shape)
    print("  X_test: ", preprocessed_log["X_test"].shape)
