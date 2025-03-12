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
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Masking, Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten
)
from tensorflow.keras.regularizers import l2
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

def load_preprocessed_data(model_task: str, eol_capacity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Loads preprocessed data and metadata from data/processed/ based on model_task and EOL capacity.

    Args:
        model_task (str): Combined model and task identifier, e.g., "lstm_regression" or "cnn_classification".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, metadata),
               where metadata includes y_max, seq_len, eol_capacity, classification, and timestamp.
    """
    output_dir = "data/processed/"
    eol_str = f"eol{int(eol_capacity*100)}"

    # Extract task type from model_task for file naming consistency with preprocessing
    if "regression" in model_task:
        task_type = "regression"
    elif "classification" in model_task:
        task_type = "classification"
    else:
        raise ValueError(f"Invalid model_task: {model_task}. Must contain 'regression' or 'classification'.")

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
    if "regression" in model_task and "y_max" not in metadata:
        raise ValueError(f"Missing y_max in metadata for {model_task} with EOL {eol_capacity}")
    if "seq_len" not in metadata:
        raise ValueError(f"Missing seq_len in metadata for {model_task} with EOL {eol_capacity}")

    logger.info(f"Loaded preprocessed data and metadata for {model_task} with EOL {eol_capacity}")
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata

def train_lstm_model(config: Config, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[tf.keras.Model, Dict]:
    """
    Trains an LSTM model for RUL regression using tuned hyperparameters from config with Functional API.

    Args:
        config (Config): Configuration object with tuned hyperparameters.
        X_train, y_train, X_val, y_val (np.ndarray): Training and validation data.

    Returns:
        Tuple: (trained model, training history).
    """
    # Derive input shape from X_train
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features), e.g., (120, 1)
    logger.info(f"Input shape derived from X_train: {input_shape}")

    # Define the model using Functional API
    inputs = Input(shape=input_shape)
    x = LSTM(
        units=config.lstm_units,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=False,
        unroll=False,
        name="lstm"  # Explicitly named for stability analysis
    )(inputs)
    x = Dropout(config.lstm_dropout_rate)(x)
    x = Dense(config.lstm_dense_units, activation='tanh')(x)
    outputs = Dense(1)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = Adam(learning_rate=config.learning_rate, clipnorm=config.clipnorm)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    logger.info("LSTM model built with tuned config: %s", str(config))

    # Define callbacks
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{timestamp}_best.keras"),
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

    # Save the final model
    final_model_path = os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{timestamp}_final.keras")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final LSTM model saved to {final_model_path}")

    return model, history.history

def train_cnn_model(config: Config, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[tf.keras.Model, Dict]:
    """
    Trains a CNN model for RUL classification using tuned hyperparameters from config.

    Args:
        config (Config): Configuration object with tuned hyperparameters.
        X_train, y_train, X_val, y_val (np.ndarray): Training and validation data.

    Returns:
        Tuple: (trained model, training history).
    """
    input_shape = (X_train.shape[1], X_train.shape[2])  # Derive input_shape from X_train
    # Build model using tuned parameters from config
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(
            filters=config.conv1_filters,
            kernel_size=config.conv1_kernel_size,
            activation='relu',
            kernel_regularizer=l2(config.l2_reg)
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(
            filters=config.conv2_filters,
            kernel_size=config.conv2_kernel_size,
            activation='relu',
            kernel_regularizer=l2(config.l2_reg)
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(
            filters=config.conv3_filters,
            kernel_size=config.conv3_kernel_size,
            activation='relu',
            kernel_regularizer=l2(config.l2_reg)
        ),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(
            units=config.cnn_dense_units,
            activation='relu',
            kernel_regularizer=l2(config.l2_reg)
        ),
        Dropout(config.cnn_dropout_rate),
        Dense(7, activation='softmax')
    ])
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("CNN model built with tuned config: %s", str(config))

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join("experiments", "models", f"cnn_classification_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_best.keras"),
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

    final_model_path = os.path.join("experiments", "models", f"cnn_classification_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_final.keras")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final CNN model saved to {final_model_path}")

    return model, history.history


def load_saved_model(model_task: str, config: Config) -> Optional[tf.keras.Model]:
    """
    Loads a previously saved best model (LSTM or CNN) for the specified model_task and EOL capacity.

    Args:
        model_task (str): Combined model and task identifier, e.g., "lstm_regression" or "cnn_classification".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).
        config (Config): Configuration object with model parameters.

    Returns:
        Optional[tf.keras.Model]: Loaded model if successful, None otherwise.
    """
    model_dir = os.path.join("experiments", "models")
    os.makedirs(model_dir, exist_ok=True)

    eol_capacity = config.eol_capacity
    eol_str = f"eol{int(eol_capacity*100)}"

    # Extract model and task type from model_task
    if "lstm_regression" in model_task:
        pattern_best = f"lstm_regression_{eol_str}_*_best.keras"
        model_name = "LSTM"
    elif "cnn_classification" in model_task:
        pattern_best = f"cnn_classification_{eol_str}_*_best.keras"
        model_name = "CNN"
    else:
        raise ValueError(f"Unsupported model_task: {model_task}. Must be 'lstm_regression' or 'cnn_classification'.")

    all_files = os.listdir(model_dir)
    logger.debug(f"Files in {model_dir}: {all_files}")

    best_files = [f for f in all_files if fnmatch.fnmatch(f, pattern_best)]
    if not best_files:
        logger.info(f"No saved best {model_name} model found for {model_task} with EOL {eol_capacity}")
        return None

    def extract_timestamp(filename: str) -> str:
        import re
        match = re.search(r'\d{8}_\d{6}', filename)
        return match.group(0) if match else "00000000_000000"

    best_files.sort(key=extract_timestamp, reverse=True)
    latest_model = best_files[0]
    model_path = os.path.join(model_dir, latest_model)

    logger.debug(f"Attempting to load best {model_name} model from {model_path}")
    try:
        model = load_model(model_path)
        logger.info(f"Loaded saved best {model_name} model from {model_path}")
        input_shape = (config.seq_len, 1)
        if model.input_shape[1:] != input_shape:
            logger.warning(f"Input shape mismatch: saved best {model_name} model has {model.input_shape[1:]}, config expects {input_shape}")
            return None
        return model
    except Exception as e:
        logger.error(f"Failed to load saved best {model_name} model from {model_path}: {str(e)}")
        return None
