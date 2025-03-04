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

def load_preprocessed_data(task_type: str, eol_capacity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Loads preprocessed data and metadata from data/processed/ based on task type and EOL capacity.

    Args:
        task_type (str): Either "classification" or "regression".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, metadata),
               where metadata includes y_max, max_sequence_length, eol_capacity, classification, and timestamp.
    """
    output_dir = "data/processed/"
    eol_str = f"eol{int(eol_capacity*100)}"

    # Find the most recent data files for the given task type and EOL capacity
    pattern = f"X_train_{task_type}_{eol_str}.npy"
    data_files = [f for f in os.listdir(output_dir) if f.startswith(pattern) and f.endswith(".npy")]
    if not data_files:
        raise FileNotFoundError(f"No preprocessed data found for {task_type} with EOL {eol_capacity}")

    # Load data arrays
    X_train = np.load(os.path.join(output_dir, f"X_train_{task_type}_{eol_str}.npy"))
    X_val = np.load(os.path.join(output_dir, f"X_val_{task_type}_{eol_str}.npy"))
    X_test = np.load(os.path.join(output_dir, f"X_test_{task_type}_{eol_str}.npy"))
    y_train = np.load(os.path.join(output_dir, f"y_train_{task_type}_{eol_str}.npy"))
    y_val = np.load(os.path.join(output_dir, f"y_val_{task_type}_{eol_str}.npy"))
    y_test = np.load(os.path.join(output_dir, f"y_test_{task_type}_{eol_str}.npy"))

    # Load metadata from JSON file
    metadata_file = [f for f in os.listdir(output_dir) if f.startswith(f"metadata_{task_type}_{eol_str}") and f.endswith(".json")]
    if not metadata_file:
        raise FileNotFoundError(f"No metadata file found for {task_type} with EOL {eol_capacity}")
    
    with open(os.path.join(output_dir, metadata_file[0]), "r") as f:
        metadata = json.load(f)

    # Validate metadata
    if task_type == "regression" and "y_max" not in metadata:
        raise ValueError(f"Missing y_max in metadata for regression task with EOL {eol_capacity}")
    if "max_sequence_length" not in metadata:
        raise ValueError(f"Missing max_sequence_length in metadata for {task_type} with EOL {eol_capacity}")

    logger.info(f"Loaded preprocessed data and metadata for {task_type} with EOL {eol_capacity}")
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata

def train_lstm_model(config: Config, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[tf.keras.Model, Dict]:
    """
    Trains an LSTM model for RUL regression using tuned hyperparameters from config.

    Args:
        config (Config): Configuration object with tuned hyperparameters.
        X_train, y_train, X_val, y_val (np.ndarray): Training and validation data.
        input_shape (Tuple[int, int]): Shape of input sequences (seq_len, 1).

    Returns:
        Tuple: (trained model, training history).
    """
    # Build model using tuned parameters from config
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(
            units=config.lstm_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            unroll=False
        ),
        Dropout(config.lstm_dropout_rate),
        Dense(config.lstm_dense_units, activation='tanh'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=config.learning_rate, clipnorm=config.clipnorm)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    logger.info("LSTM model built with tuned config: %s", str(config))

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=callbacks
    )

    final_model_path = os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_final.keras")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final LSTM model saved to {final_model_path}")

    return model, history.history

def load_saved_model(task_type: str, eol_capacity: float, config: Config) -> Optional[tf.keras.Model]:
    """
    Loads a previously saved best LSTM model for the specified task type and EOL capacity.
    """
    model_dir = os.path.join("experiments", "models")
    os.makedirs(model_dir, exist_ok=True)
    eol_str = f"eol{int(eol_capacity*100)}"
    pattern_best = f"lstm_{task_type}_{eol_str}_*_best.keras"

    all_files = os.listdir(model_dir)
    logger.debug(f"Files in {model_dir}: {all_files}")

    best_files = [f for f in all_files if fnmatch.fnmatch(f, pattern_best)]
    if not best_files:
        logger.info(f"No saved best model found for {task_type} with EOL {eol_capacity}")
        return None

    def extract_timestamp(filename: str) -> str:
        import re
        match = re.search(r'\d{8}_\d{6}', filename)
        return match.group(0) if match else "00000000_000000"

    best_files.sort(key=extract_timestamp, reverse=True)
    latest_model = best_files[0]
    model_path = os.path.join(model_dir, latest_model)

    logger.debug(f"Attempting to load best model from {model_path}")
    try:
        model = load_model(model_path)
        logger.info(f"Loaded saved best model from {model_path}")
        input_shape = (config.seq_len, 1)
        if model.input_shape[1:] != input_shape:
            logger.warning(f"Input shape mismatch: saved best model has {model.input_shape[1:]}, config expects {input_shape}")
            return None
        return model
    except Exception as e:
        logger.error(f"Failed to load saved best model from {model_path}: {str(e)}")
        return None

def main():
    """
    Main function to run LSTM experiments for RUL regression on the Aachen dataset.
    """
    config = Config()
    task_type = "classification" if config.classification else "regression"  # Derive task_type from config

    try:
        # Load preprocessed data
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_preprocessed_data(
            task_type, config.eol_capacity
        )

        # Validate data shapes
        assert X_train.shape[-1] == 1, "Features dimension incorrect for LSTM"
        assert X_train.shape[1] == metadata["max_sequence_length"], "Sequence length mismatch"
        logger.info("Preprocessed data validated successfully")

        # Build and train the model
        model = build_lstm_model((metadata["max_sequence_length"], 1), config)
        history = train_lstm_model(model, X_train, y_train, X_val, y_val, config)

        # Note: evaluate_lstm_model, plot_training_history, etc., are missing; assuming they’re defined elsewhere
        # For completeness, here’s a placeholder evaluation
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

        results = {
            "test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "eol_capacity": config.eol_capacity,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        os.makedirs(os.path.join("experiments", "results"), exist_ok=True)
        np.save(os.path.join("experiments", "results", f"lstm_results_eol{int(config.eol_capacity*100)}_{results['timestamp']}.npy"), results)
        logger.info("LSTM experiment completed and results stored")

    except Exception as e:
        logger.error("Error in LSTM experiment: %s", str(e))
        raise

if __name__ == "__main__":
    main()