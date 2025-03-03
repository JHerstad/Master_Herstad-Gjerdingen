#!/usr/bin/env python3
"""
Script for performing hyperparameter tuning on the LSTM model for RUL regression using Keras Tuner.
This script integrates with existing scripts in src/models.py, src/evaluation.py, and src/visualization.py,
and saves the best hyperparameters automatically using Keras Tuner.

The script builds a model and runs a hyperparameter search, storing results in experiments/hyperparameter_tuning.
Training is left to src/models.py for consistency and modularity.
"""

import os
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
import logging
from src.models import load_preprocessed_data
from config.defaults import Config  # Import Config

# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_model(hp, input_shape):
    """
    Keras Tuner-compatible model function for hyperparameter tuning of the LSTM model.

    Args:
        hp: Hyperparameters object from Keras Tuner.
        input_shape: Tuple of (seq_len, 1) for input sequences.

    Returns:
        Compiled LSTM model with tunable hyperparameters for RUL regression.
    """
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(
            units=hp.Int("lstm_units", min_value=16, max_value=64, step=16),
            activation='tanh'
        ),
        Dropout(
            rate=hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
        ),
        Dense(
            units=hp.Int("dense_units", min_value=8, max_value=64, step=8),
            activation='tanh'
        ),
        Dense(1)  # Output layer for regression
    ])
    optimizer = Adam(
        learning_rate=hp.Choice("learning_rate", values=[0.001, 0.01, 0.1])
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def run_hyperparameter_search(config: Config):
    """
    Runs hyperparameter tuning for the LSTM model using Keras Tuner with Bayesian Optimization,
    saving the best hyperparameters automatically.

    Args:
        config (Config): Configuration object with model and tuning parameters.

    Returns:
        dict: Best hyperparameters found during the search.
    """
    # Determine task type based on config.classification
    task_type = "classification" if config.classification else "regression"
    logger.info(f"Task type set to: {task_type}")

    # Load preprocessed data
    logger.info("Loading preprocessed data for hyperparameter tuning...")
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_preprocessed_data(
        task_type, config.eol_capacity
    )
    input_shape = (metadata["max_sequence_length"], 1)

    # Set up the Keras Tuner
    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp, input_shape),
        objective="val_loss",
        max_trials=config.max_trials,
        executions_per_trial=1,
        directory=config.tuner_directory,
        project_name=f"{config.project_name}_lstm_tuning_eol{int(config.eol_capacity*100)}"
    )

    # Perform hyperparameter search
    logger.info("Starting hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        epochs=config.tuning_epochs,
        validation_data=(X_val, y_val),
        batch_size=config.batch_size,
        verbose=1
    )

    # Get and save the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = best_hps.values
    logger.info(f"Best hyperparameters found: {best_params}")

    return best_params

def main():
    """
    Main function to run the hyperparameter tuning for LSTM.
    """
    # Initialize configuration
    config = Config()

    try:
        # Run hyperparameter search
        best_params = run_hyperparameter_search(config)
        print(f"Best hyperparameters for tuning: {best_params}")
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()