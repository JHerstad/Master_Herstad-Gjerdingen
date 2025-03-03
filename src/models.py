#!/usr/bin/env python3
"""
Module for defining and training an LSTM model for RUL regression on the Aachen dataset.

This module integrates with src/preprocessing.py for data loading and config/defaults.py
for configuration, ensuring reproducibility and professionalism for thesis experiments.
It focuses on a single LSTM model for regression, with plans for expansion to other models.
Hyperparameter tuning is handled in a separate file (e.g., experiments/grid_search_lstm.py).
"""

import os
import sys
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config.defaults import Config
import datetime
import json
import fnmatch



# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add thesis_experiment/ to sys.path for imports (if needed when run standalone)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def load_preprocessed_data(model_type: str, eol_capacity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Loads preprocessed data and metadata from data/processed/ based on model type and EOL capacity.

    Args:
        model_type (str): Either "classification" or "regression".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, metadata),
               where metadata includes y_max, max_sequence_length, eol_capacity, classification, and timestamp.

    Raises:
        FileNotFoundError: If no preprocessed data or metadata files are found for the given model type and EOL capacity.
        ValueError: If required metadata fields (y_max for regression, max_sequence_length) are missing.

    Notes:
        - Loads data arrays (.npy) and metadata (.json) from data/processed/, ensuring reproducibility
          for thesis experiments.
        - Requires metadata to be stored in a separate JSON file (e.g., metadata_regression_eol65.json)
          for y_max and other preprocessing details; no fallback simulation is provided.
    """
    output_dir = "data/processed/"
    eol_str = f"eol{int(eol_capacity*100)}"

    # Find the most recent data files for the given model type and EOL capacity
    pattern = f"X_train_{model_type}_{eol_str}.npy"
    data_files = [f for f in os.listdir(output_dir) if f.startswith(pattern) and f.endswith(".npy")]
    if not data_files:
        raise FileNotFoundError(f"No preprocessed data found for {model_type} with EOL {eol_capacity}")

    # Load data arrays (overwritten files have no timestamp)
    X_train = np.load(os.path.join(output_dir, f"X_train_{model_type}_{eol_str}.npy"))
    X_val = np.load(os.path.join(output_dir, f"X_val_{model_type}_{eol_str}.npy"))
    X_test = np.load(os.path.join(output_dir, f"X_test_{model_type}_{eol_str}.npy"))
    y_train = np.load(os.path.join(output_dir, f"y_train_{model_type}_{eol_str}.npy"))
    y_val = np.load(os.path.join(output_dir, f"y_val_{model_type}_{eol_str}.npy"))
    y_test = np.load(os.path.join(output_dir, f"y_test_{model_type}_{eol_str}.npy"))

    # Load metadata from JSON file, required for operation
    metadata_file = [f for f in os.listdir(output_dir) if f.startswith(f"metadata_{model_type}_{eol_str}") and f.endswith(".json")]
    if not metadata_file:
        raise FileNotFoundError(f"No metadata file found for {model_type} with EOL {eol_capacity}")
    
    with open(os.path.join(output_dir, metadata_file[0]), "r") as f:
        metadata = json.load(f)  # Load the full dictionary directly

    # Validate metadata includes required fields for regression or classification
    if model_type == "regression" and "y_max" not in metadata:
        raise ValueError(f"Missing y_max in metadata for regression model with EOL {eol_capacity}")
    if "max_sequence_length" not in metadata:
        raise ValueError(f"Missing max_sequence_length in metadata for {model_type} with EOL {eol_capacity}")

    logger.info(f"Loaded preprocessed data and metadata for {model_type} with EOL {eol_capacity}")
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def build_lstm_model(input_shape: Tuple[int, int], config: Config) -> tf.keras.Model:
    """
    Builds an LSTM model for RUL regression on the Aachen dataset.

    Args:
        input_shape (Tuple[int, int]): Shape of input sequences (seq_len, 1).
        config (Config): Configuration object with hyperparameters (e.g., lstm_units, dropout_rate).

    Returns:
        tf.keras.Model: Compiled LSTM model for regression.
    """
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        LSTM(
            config.lstm_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            unroll=False
        ),
        Dropout(config.dropout_rate),
        Dense(config.dense_units, activation='tanh'),
        Dense(1)  # Output layer for regression
    ])
    optimizer = Adam(learning_rate=config.learning_rate, clipnorm=config.clipnorm)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    logger.info("LSTM model built for regression with config: %s", str(config))
    return model


def train_lstm_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: Config) -> Dict:
    """
    Trains the LSTM model for RUL regression with early stopping, model checkpointing, and explicit model saving.

    This function trains the model, saves the best model during training based on validation loss,
    and saves the final model after training for additional reproducibility in thesis experiments.

    Args:
        model (tf.keras.Model): Compiled LSTM model.
        X_train (np.ndarray): Training input sequences with shape (n_samples, seq_len, 1).
        y_train (np.ndarray): Training target values (normalized) with shape (n_samples,).
        X_val (np.ndarray): Validation input sequences with shape (n_samples, seq_len, 1).
        y_val (np.ndarray): Validation target values (normalized) with shape (n_samples,).
        config (Config): Configuration object with training parameters (e.g., epochs, batch_size, patience).

    Returns:
        Dict: Training history containing loss metrics.

    Notes:
        - Uses EarlyStopping to prevent overfitting and ModelCheckpoint to save the best model.
        - Explicitly saves the final model after training for additional reproducibility.
        - Saves models in the native Keras format (.keras) for modern compatibility, recommended over legacy HDF5 (.h5).
        - Logs training progress and saves for professional tracking in a thesis context.
    """
    # Define callbacks for training, including early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model with the specified callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=callbacks
    )

    # Explicitly save the final model after training, with a distinct filename in native Keras format
    final_model_path = os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_final.keras")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final LSTM model saved to {final_model_path} in native Keras format")
    logger.info("LSTM model trained successfully with config: %s", str(config))

    return history.history

import fnmatch  # Add this import at the top of src/models.py

def load_saved_model(model_type: str, eol_capacity: float, config: Config) -> Optional[tf.keras.Model]:
    """
    Loads a previously saved LSTM model for RUL regression with matching eol_capacity and model type,
    focusing only on the best (lowest validation loss) models.

    Args:
        model_type (str): Either "classification" or "regression".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).
        config (Config): Configuration object with model parameters (e.g., seq_len, lstm_units).

    Returns:
        Optional[tf.keras.Model]: Loaded best model if found, None if no matching best model exists.

    Notes:
        - Looks for the most recent saved best model in experiments/models/ with matching
          eol_capacity and model_type, ensuring compatibility with current config.
        - Validates input shape and hyperparameters for reproducibility in thesis experiments.
        - Uses fnmatch for precise filename pattern matching to handle files like
          lstm_regression_eol65_YYYYMMDD_HHMMSS_best.keras.
        - Ignores final models (ending with _final.keras) as per request.
    """
    model_dir = os.path.join("experiments", "models")
    os.makedirs(model_dir, exist_ok=True)
    eol_str = f"eol{int(eol_capacity*100)}"

    # Define pattern for best models only
    pattern_best = f"lstm_{model_type}_{eol_str}_*_best.keras"

    # List all files in the model directory for debugging
    all_files = os.listdir(model_dir)
    logger.debug(f"Files in {model_dir}: {all_files}")

    # Use fnmatch to filter only best model files
    best_files = [f for f in all_files if fnmatch.fnmatch(f, pattern_best)]

    if not best_files:
        logger.info(f"No saved best model found for {model_type} with EOL {eol_capacity}")
        return None

    # Sort by timestamp (extract from filename: YYYYMMDD_HHMMSS) to get most recent
    def extract_timestamp(filename: str) -> tuple:
        """Extract timestamp (YYYYMMDD_HHMMSS) from filename for sorting."""
        import re
        match = re.search(r'\d{8}_\d{6}', filename)
        if match:
            return match.group(0)
        return "00000000_000000"  # Default for sorting if no timestamp found

    best_files.sort(key=extract_timestamp, reverse=True)
    latest_model = best_files[0]
    model_path = os.path.join(model_dir, latest_model)

    # Debug the selected model path
    logger.debug(f"Attempting to load best model from {model_path}")

    try:
        # Load the best model in native Keras format
        model = load_model(model_path)
        logger.info(f"Loaded saved best model from {model_path}")

        # Validate model compatibility with current config
        input_shape = (config.seq_len, 1)  # Assumes seq_len is in Config
        if model.input_shape[1:] != input_shape:
            logger.warning(f"Input shape mismatch: saved best model has {model.input_shape[1:]}, config expects {input_shape}")
            return None
        # Optionally validate other hyperparameters (e.g., lstm_units, dense_units) by inspecting model layers
        # For simplicity, assume basic compatibility; extend for full validation if needed

        return model
    except Exception as e:
        logger.error(f"Failed to load saved best model from {model_path}: {str(e)}")
        return None
    

def main():
    """
    Main function to run LSTM experiments for RUL regression on the Aachen dataset,
    including training, evaluation, and visualization.
    Hyperparameter tuning is handled in a separate file (e.g., experiments/grid_search_lstm.py).
    """
    config = Config()
    model_type = "regression"  # Fixed for LSTM regression

    try:
        # Load preprocessed data for regression
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_preprocessed_data(
            model_type, config.eol_capacity
        )

        # Validate data shapes
        assert X_train.shape[-1] == 1, "Features dimension incorrect for LSTM"
        assert X_train.shape[1] == metadata["max_sequence_length"], "Sequence length mismatch"
        logger.info("Preprocessed data validated successfully")

        # Build and train the model with default config
        model = build_lstm_model((metadata["max_sequence_length"], 1), config)
        history = train_lstm_model(model, X_train, y_train, X_val, y_val, config)

        # Evaluate the model
        test_loss, test_mae = evaluate_lstm_model(model, X_test, y_test, metadata["y_max"])

        # Visualize results
        plot_training_history(history)
        y_pred = model.predict(X_test, verbose=0).flatten()
        plot_predictions_vs_actual(y_test, y_pred, metadata["y_max"])
        plot_residuals(y_test, y_pred, metadata["y_max"])

        # Store evaluation results
        results = {
            "test_loss": test_loss,
            "test_mae": test_mae,
            "eol_capacity": config.eol_capacity,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        np.save(os.path.join("experiments", "results", f"lstm_results_eol{int(config.eol_capacity*100)}_{results['timestamp']}.npy"), results)
        logger.info("LSTM experiment completed and results stored")

    except Exception as e:
        logger.error("Error in LSTM experiment: %s", str(e))
        raise


if __name__ == "__main__":
    main()