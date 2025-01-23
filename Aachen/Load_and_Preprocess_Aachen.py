# Script for loading and preprocessing the Aachen dataset

# Import necessary libraries
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mat4py as mpy

def preprocess_aachen_dataset(file_path, test_cell_count=3, random_state=42):
    """
    Preprocess the Aachen dataset for LSTM training.
    
    Parameters:
    ----------
    file_path : str
        Path to the .mat file containing the dataset.
    test_cell_count : int, optional
        Number of unique cells to hold out for testing, by default 3.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns:
    -------
    dict
        A dictionary containing preprocessed training, validation, and testing datasets:
        - X_train, X_val, X_test: Padded sequences for LSTM input
        - y_train, y_val, y_test: Normalized target values
        - max_sequence_length: Maximum sequence length used for padding
    """
    # Load the .mat file and convert it into a pandas DataFrame
    data_loader = mpy.loadmat(file_path)
    df = pd.DataFrame.from_dict(data_loader["TDS"])

    def compute_eol_and_rul80(row):
        """
        Compute EOL80 (End of Life where capacity drops below 80% of initial)
        and RUL80 (Remaining Useful Life at 80% capacity).

        Parameters:
        ----------
        row : pd.Series
            A row of the dataset containing historical and target capacity data.

        Returns:
        -------
        pd.Series
            Contains calculated EOL80 and RUL80 values.
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

        # Find the first cycle where capacity drops below the threshold
        below_threshold_indices = np.where(target_cap < threshold)[0]
        if len(below_threshold_indices) > 0:
            eol80_index = below_threshold_indices[0]
            eol80_cycle = target_cycles[eol80_index]

        # Calculate RUL80 as the difference between EOL80 and the last history cycle
        if not pd.isna(eol80_cycle):
            last_history_cycle = history_cycles[-1]
            rul80 = eol80_cycle - last_history_cycle

        return pd.Series({"EOL80": eol80_cycle, "RUL80": rul80})

    # Apply the function to compute EOL80 and RUL80 for each row
    df[["EOL80", "RUL80"]] = df.apply(compute_eol_and_rul80, axis=1)

    # Filter out rows with invalid RUL80 values
    df_valid = df[df["RUL80"].notna()]

    # Hold back specific cells for testing
    cells_to_hold_back = df_valid["Cell"].unique()[:test_cell_count]
    df_test = df_valid[df_valid["Cell"].isin(cells_to_hold_back)]
    df_train_val = df_valid[~df_valid["Cell"].isin(cells_to_hold_back)]

    # Split the remaining data into training and validation sets
    df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=random_state, stratify=df_train_val["Cell"])

    # Extract historical capacity sequences (inputs) and target RUL80 values (outputs)
    history_train = df_train["History"].tolist()
    history_val = df_val["History"].tolist()
    history_test = df_test["History"].tolist()

    y_train = np.array(df_train["RUL80"])
    y_val = np.array(df_val["RUL80"])
    y_test = np.array(df_test["RUL80"])

    # Normalize historical capacity sequences using MinMaxScaler
    all_histories = history_train + history_val + history_test
    all_histories_flat = np.concatenate(all_histories)
    scaler = MinMaxScaler()
    scaler.fit(all_histories_flat.reshape(-1, 1))

    # Apply the scaler to all historical capacity sequences
    history_train_normalized = [scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in history_train]
    history_val_normalized = [scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in history_val]
    history_test_normalized = [scaler.transform(np.array(h).reshape(-1, 1)).flatten() for h in history_test]

    # Normalize target RUL80 values using the maximum value in the training set
    y_max = y_train.max()
    y_train = y_train / y_max
    y_val = y_val / y_max
    y_test = y_test / y_max

    # Pad all normalized sequences to the maximum sequence length
    max_sequence_length = max(len(h) for h in all_histories)
    X_train_padded = pad_sequences(history_train_normalized, maxlen=max_sequence_length, padding='post', dtype='float32')
    X_val_padded = pad_sequences(history_val_normalized, maxlen=max_sequence_length, padding='post', dtype='float32')
    X_test_padded = pad_sequences(history_test_normalized, maxlen=max_sequence_length, padding='post', dtype='float32')

    # Add an extra dimension to sequences for LSTM input (samples, time steps, features)
    X_train_lstm = X_train_padded[..., np.newaxis]
    X_val_lstm = X_val_padded[..., np.newaxis]
    X_test_lstm = X_test_padded[..., np.newaxis]

    # Return the preprocessed datasets as a dictionary
    return {
        "X_train": X_train_lstm,
        "X_val": X_val_lstm,
        "X_test": X_test_lstm,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "max_sequence_length": max_sequence_length,
    }

# Example usage
if __name__ == "__main__":
    # Preprocess the dataset and print shapes of the preprocessed data
    preprocessed = preprocess_aachen_dataset("/Users/johannesherstad/Master_Herstad-Gjerdingen/Aachen/Degradation_Prediction_Dataset_ISEA.mat", test_cell_count=3, random_state=42)
    print("Training set:")
    print(f"X_train shape: {preprocessed['X_train'].shape}")
    print(f"y_train shape: {preprocessed['y_train'].shape}")
    print("\nValidation set:")
    print(f"X_val shape: {preprocessed['X_val'].shape}")
    print(f"y_val shape: {preprocessed['y_val'].shape}")
    print("\nTest set:")
    print(f"X_test shape: {preprocessed['X_test'].shape}")
    print(f"y_test shape: {preprocessed['y_test'].shape}")
    print("\nMax sequence length:", preprocessed["max_sequence_length"])
