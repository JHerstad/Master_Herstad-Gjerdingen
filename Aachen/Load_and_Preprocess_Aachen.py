import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mat4py as mpy

def preprocess_aachen_dataset(
    file_path,
    eol_capacity=0.80,
    test_cell_count=3,
    random_state=42,
    phase=None,
    early_threshold=800,
    mid_threshold=400,
    log_transform=False
):
    """
    Preprocess the Aachen dataset for LSTM training. Allows dynamic EOL threshold
    (e.g., 0.80 for 'EOL80' or 0.65 for 'EOL65').

    Parameters
    ----------
    file_path : str
        Path to the .mat file containing the dataset.
    eol_capacity : float, optional
        Fraction of the initial capacity at which the cell is considered End of Life (EOL).
        For example, 0.80 yields EOL at 80% capacity, 0.65 yields EOL at 65% capacity.
    test_cell_count : int, optional
        Number of unique cells to hold out for testing, by default 3.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    phase : str or None, optional
        Life phase to filter the data. Can be one of {"early", "mid", "late"} or None.
        - "early": keep samples with RUL > early_threshold
        - "mid":   keep samples where mid_threshold < RUL <= early_threshold
        - "late":  keep samples with RUL <= mid_threshold
        If None, no phase-based filtering is applied.
    early_threshold : float, optional
        Upper RUL threshold used to define the "early" vs "mid" boundary, by default 800.
        (Units of cycles remaining.)
    mid_threshold : float, optional
        Lower RUL threshold used to define the "mid" vs "late" boundary, by default 400.
        (Units of cycles remaining.)
    log_transform : bool, optional
        Whether to apply a log transform to the RUL values before normalization,
        by default False.

    Returns
    -------
    dict
        A dictionary containing preprocessed training, validation, and testing datasets:
        - X_train, X_val, X_test: Padded sequences for LSTM input
        - y_train, y_val, y_test: Target (RUL) values after optional log transform
                                  and normalization
        - y_max: The max RUL in the training set (post-log if log_transform=True),
                 used for re-scaling predictions.
        - max_sequence_length: Maximum sequence length used for padding
        - log_transform: Boolean indicating if log transform was applied.
    """

    # 1) Load and transform the raw data from .mat into a Pandas DataFrame
    data_loader = mpy.loadmat(file_path)
    df = pd.DataFrame.from_dict(data_loader["TDS"])
    
    def compute_eol_and_rul(row):
        """
        Compute EOL (End of Life) and RUL (Remaining Useful Life) at the chosen
        fraction of initial capacity (eol_capacity).
        """
        history_cap = np.array(row["History"])
        history_cycles = np.array(row["History_Cycle"])
        target_cap = np.array(row["Target_expanded"])
        target_cycles = np.array(row["Target_Cycle_Expanded"])

        eol_cycle = np.nan
        rul_value = np.nan

        # Handle cases with missing data in historical capacity or cycles
        if len(history_cap) == 0 or len(history_cycles) == 0:
            return pd.Series({"EOL": eol_cycle, "RUL": rul_value})

        # Determine the threshold for EOL
        initial_capacity = history_cap[0]
        threshold = eol_capacity * initial_capacity

        # Check if the historical capacity already falls below the threshold
        # If it does, we consider it invalid for our typical definition of "future" RUL
        if history_cap[-1] <= threshold:
            return pd.Series({"EOL": np.nan, "RUL": np.nan})

        # Handle cases with missing target data
        if len(target_cap) == 0 or len(target_cycles) == 0:
            return pd.Series({"EOL": eol_cycle, "RUL": rul_value})

        # Find the first cycle where capacity drops below the threshold in the target portion
        below_threshold_indices = np.where(target_cap < threshold)[0]
        if len(below_threshold_indices) > 0:
            eol_index = below_threshold_indices[0]
            eol_cycle = target_cycles[eol_index]

        # Calculate RUL as the difference between EOL and the last history cycle
        if not pd.isna(eol_cycle):
            last_history_cycle = history_cycles[-1]
            rul_value = eol_cycle - last_history_cycle

        return pd.Series({"EOL": eol_cycle, "RUL": rul_value})

    # Compute EOL and RUL for each row using the dynamic threshold
    df[["EOL", "RUL"]] = df.apply(compute_eol_and_rul, axis=1)

    # Filter out rows with invalid RUL (NaN, etc.)
    df_valid = df[df["RUL"].notna()]

    # 2) Hold back specific cells for testing
    cells_to_hold_back = df_valid["Cell"].unique()[:test_cell_count]
    df_test = df_valid[df_valid["Cell"].isin(cells_to_hold_back)]
    df_train_val = df_valid[~df_valid["Cell"].isin(cells_to_hold_back)]

    # 3) Split the remaining data into training and validation sets
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.2,
        random_state=random_state,
        stratify=df_train_val["Cell"]
    )

    # 4) (OPTIONAL) Phase-based filtering for train, val, test
    def filter_by_phase(df_in, phase_name, early_thr, mid_thr):
        if phase_name == "early":
            # Keep samples with RUL > early_threshold
            return df_in[df_in["RUL"] > early_thr]
        elif phase_name == "mid":
            # Keep samples where mid_threshold < RUL <= early_threshold
            return df_in[(df_in["RUL"] > mid_thr) & (df_in["RUL"] <= early_thr)]
        elif phase_name == "late":
            # Keep samples with RUL <= mid_thr
            return df_in[df_in["RUL"] <= mid_thr]
        else:
            # If no phase is specified or phase is None, do not filter
            return df_in
    
    if phase is not None:
        df_train = filter_by_phase(df_train, phase, early_threshold, mid_threshold)
        df_val = filter_by_phase(df_val, phase, early_threshold, mid_threshold)
        df_test = filter_by_phase(df_test, phase, early_threshold, mid_threshold)

    # 5) Extract sequences (X) and target RUL (y)
    history_train = df_train["History"].tolist()
    history_val = df_val["History"].tolist()
    history_test = df_test["History"].tolist()

    y_train = np.array(df_train["RUL"])
    y_val = np.array(df_val["RUL"])
    y_test = np.array(df_test["RUL"])

    # 5.1) (OPTIONAL) Log transform the target
    # If worried about zero or negative values, consider np.log1p
    if log_transform:
        y_train = np.log(y_train)
        y_val = np.log(y_val)
        y_test = np.log(y_test)

    # 6) Normalize historical capacity sequences
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

    # 7) Normalize target values (post-log if log_transform=True)
    y_max = y_train.max() if len(y_train) > 0 else 1.0  # Avoid divide-by-zero
    y_train_norm = y_train / y_max
    y_val_norm = y_val / y_max
    y_test_norm = y_test / y_max

    # 8) Pad sequences to the maximum sequence length
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

    # 9) Reshape for LSTM input (samples, time steps, features)
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
        "y_max": y_max,
        "max_sequence_length": max_sequence_length,
        "log_transform": log_transform
    }

# --- Example usage ---
if __name__ == "__main__":
    # EOL at 80% capacity (original behavior)
    preprocessed_80 = preprocess_aachen_dataset(
        "Aachen/Degradation_Prediction_Dataset_ISEA.mat",
        eol_capacity=0.80,
        test_cell_count=3,
        random_state=42,
        phase=None,
        log_transform=False
    )
    print("EOL80 => X_train:", preprocessed_80["X_train"].shape)
    print("EOL80 => X_val:  ", preprocessed_80["X_val"].shape)
    print("EOL80 => X_test: ", preprocessed_80["X_test"].shape)

    # EOL at 65% capacity
    preprocessed_65 = preprocess_aachen_dataset(
        "Aachen/Degradation_Prediction_Dataset_ISEA.mat",
        eol_capacity=0.65,
        test_cell_count=3,
        random_state=42,
        phase=None,
        log_transform=False
    )
    print("EOL65 => X_train:", preprocessed_65["X_train"].shape)
    print("EOL65 => X_val:  ", preprocessed_65["X_val"].shape)
    print("EOL65 => X_test: ", preprocessed_65["X_test"].shape)
