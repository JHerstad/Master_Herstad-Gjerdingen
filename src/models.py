#!/usr/bin/env python3
"""
Module for defining and training LSTM models for RUL regression on the Aachen dataset,
including hyperparameter tuning with Keras Tuner, evaluation, and visualization.

This module integrates with src/preprocessing.py for data loading and config/defaults.py
for configuration, ensuring reproducibility and professionalism for thesis experiments.
"""

import os
import sys
import numpy as np
import logging
from typing import Tuple, Dict
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config.defaults import Config

# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add thesis_experiment/ to sys.path for imports (if needed when run standalone)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def load_preprocessed_data(model_type: str, eol_capacity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Loads preprocessed data from data/processed/ based on model type and EOL capacity.

    Args:
        model_type (str): Either "classification" or "regression".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, metadata)
    """
    output_dir = "data/processed/"
    eol_str = f"eol{int(eol_capacity*100)}"

    # Find the most recent files for the given model type and EOL capacity
    pattern = f"X_train_{model_type}_{eol_str}.npy"
    files = [f for f in os.listdir(output_dir) if f.startswith(f"X_train_{model_type}_{eol_str}_") and f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No preprocessed data found for {model_type} with EOL {eol_capacity}")

    # Load data arrays (overwritten files have no timestamp)
    X_train = np.load(os.path.join(output_dir, f"X_train_{model_type}_{eol_str}.npy"))
    X_val = np.load(os.path.join(output_dir, f"X_val_{model_type}_{eol_str}.npy"))
    X_test = np.load(os.path.join(output_dir, f"X_test_{model_type}_{eol_str}.npy"))
    y_train = np.load(os.path.join(output_dir, f"y_train_{model_type}_{eol_str}.npy"))
    y_val = np.load(os.path.join(output_dir, f"y_val_{model_type}_{eol_str}.npy"))
    y_test = np.load(os.path.join(output_dir, f"y_test_{model_type}_{eol_str}.npy"))

    # Load metadata (even though not stored, we'll simulate for regression)
    metadata_file = [f for f in os.listdir(output_dir) if f.startswith(f"metadata_{model_type}_{eol_str}_") and f.endswith(".json")]
    metadata = {}
    if metadata_file:
        with open(os.path.join(output_dir, metadata_file[0]), "r") as f:
            metadata = json.load(f)[model_type]
    else:
        metadata = {
            "max_sequence_length": X_train.shape[1],
            "eol_capacity": eol_capacity,
            "classification": model_type == "classification"
        }

    logger.info(f"Loaded preprocessed data for {model_type} with EOL {eol_capacity}")
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def build_lstm_model(input_shape: Tuple[int, int], config: Dict) -> tf.keras.Model:
    """
    Builds an LSTM model for RUL regression on the Aachen dataset.

    Args:
        input_shape (Tuple[int, int]): Shape of input sequences (seq_len, 1).
        config (Dict): Configuration dictionary with hyperparameters (e.g., lstm_units, dropout_rate).

    Returns:
        tf.keras.Model: Compiled LSTM model for regression.
    """
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        LSTM(
            config["lstm_units"],
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            unroll=False
        ),
        Dropout(config["dropout_rate"]),
        Dense(config["dense_units"], activation='tanh'),
        Dense(1)  # Output layer for regression
    ])
    optimizer = Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    logger.info("LSTM model built for regression with config: %s", str(config))
    return model


def tune_lstm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: Config) -> Dict:
    """
    Tunes LSTM hyperparameters using Bayesian Optimization from Keras Tuner.

    Args:
        X_train (np.ndarray): Training input sequences.
        y_train (np.ndarray): Training target values.
        X_val (np.ndarray): Validation input sequences.
        y_val (np.ndarray): Validation target values.
        config (Config): Configuration object with tuning parameters.

    Returns:
        Dict: Best hyperparameters from the tuning process.
    """
    def build_model_tuner(hp):
        """Keras Tuner-compatible model function for hyperparameter tuning."""
        model = Sequential([
            Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(
                hp.Int("lstm_units", min_value=16, max_value=64, step=16),
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False
            ),
            Dropout(hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)),
            Dense(hp.Int("dense_units", min_value=8, max_value=64, step=8), activation='tanh'),
            Dense(1)  # Output layer for regression
        ])
        optimizer = Adam(learning_rate=hp.Choice("learning_rate", values=[0.001, 0.01, 0.1]))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    tuner = kt.BayesianOptimization(
        build_model_tuner,
        objective="val_loss",
        max_trials=10,  # Number of models to test
        executions_per_trial=1,  # Number of times to train each model
        directory=os.path.join("experiments", "hyperparameter_tuning"),
        project_name=f"lstm_tuning_eol{int(config.eol_capacity*100)}"
    )

    # Run hyperparameter search
    logger.info("Starting hyperparameter tuning for LSTM with Bayesian Optimization")
    tuner.search(
        X_train, y_train,
        epochs=20,  # Shorter for tuning, increase for final training
        validation_data=(X_val, y_val),
        batch_size=config.batch_size,
        verbose=1
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = {
        "lstm_units": best_hps.get("lstm_units"),
        "dropout_rate": best_hps.get("dropout_rate"),
        "dense_units": best_hps.get("dense_units"),
        "learning_rate": best_hps.get("learning_rate"),
        "clipnorm": config.clipnorm  # Use default from Config if not tuned
    }
    logger.info("Best hyperparameters found: %s", str(best_params))
    return best_params


def train_lstm_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: Config) -> Dict:
    """
    Trains the LSTM model for RUL regression with early stopping and model checkpointing.

    Args:
        model (tf.keras.Model): Compiled LSTM model.
        X_train (np.ndarray): Training input sequences.
        y_train (np.ndarray): Training target values.
        X_val (np.ndarray): Validation input sequences.
        y_val (np.ndarray): Validation target values.
        config (Config): Configuration object with training parameters.

    Returns:
        Dict: Training history.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join("experiments", "models", f"lstm_regression_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=callbacks
    )
    logger.info("LSTM model trained successfully with config: %s", str(config))
    return history.history


def evaluate_lstm_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, y_max: float) -> Tuple[float, float]:
    """
    Evaluates the LSTM model on the test set and rescales predictions.

    Args:
        model (tf.keras.Model): Trained LSTM model.
        X_test (np.ndarray): Test input sequences.
        y_test (np.ndarray): Test target values (normalized).
        y_max (float): Maximum RUL for rescaling predictions.

    Returns:
        Tuple[float, float]: (Test Loss, Test MAE) after rescaling.
    """
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred_rescaled = y_pred * y_max
    y_test_rescaled = y_test * y_max
    mae_rescaled = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
    logger.info("LSTM model evaluated - Test Loss: %.4f, Test MAE (rescaled): %.4f", test_loss, mae_rescaled)
    return test_loss, mae_rescaled


def plot_training_history(history: Dict) -> None:
    """
    Plots the training and validation loss over epochs.

    Args:
        history (Dict): Training history containing 'loss' and 'val_loss'.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title("Training and Validation Loss for LSTM")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("experiments", "results", f"lstm_training_loss_eol{int(Config().eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.close()


def plot_predictions_vs_actual(y_test: np.ndarray, y_pred: np.ndarray, y_max: float) -> None:
    """
    Scatterplot comparing actual vs. predicted RUL, rescaled to original range.

    Args:
        y_test (np.ndarray): Test target values (normalized).
        y_pred (np.ndarray): Predicted values (normalized).
        y_max (float): Maximum RUL for rescaling.
    """
    import matplotlib.pyplot as plt
    y_test_rescaled = y_test * y_max
    y_pred_rescaled = y_pred.flatten() * y_max
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.7)
    plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--', label='Perfect Prediction')
    plt.title("Predicted vs Actual RUL for LSTM")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("experiments", "results", f"lstm_predictions_eol{int(Config().eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.close()


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, y_max: float) -> None:
    """
    Plots residuals (difference between actual and predicted values), rescaled to original range.

    Args:
        y_test (np.ndarray): Test target values (normalized).
        y_pred (np.ndarray): Predicted values (normalized).
        y_max (float): Maximum RUL for rescaling.
    """
    import matplotlib.pyplot as plt
    y_test_rescaled = y_test * y_max
    y_pred_rescaled = y_pred.flatten() * y_max
    residuals = y_test_rescaled - y_pred_rescaled
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Residuals Histogram for LSTM")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(os.path.join("experiments", "results", f"lstm_residuals_eol{int(Config().eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.close()


def main():
    """
    Main function to run LSTM experiments, including hyperparameter tuning, training, evaluation,
    and visualization for RUL regression on the Aachen dataset.
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

        # Hyperparameter tuning (optional, controlled by config)
        if config.hyperparameter_tuning:
            best_params = tune_lstm_hyperparameters(X_train, y_train, X_val, y_val, config)
            # Update config with best parameters
            for key, value in best_params.items():
                setattr(config, key, value)
        else:
            # Use default config for training
            best_params = {
                "lstm_units": config.lstm_units,
                "dropout_rate": config.dropout_rate,
                "dense_units": config.dense_units,
                "learning_rate": config.learning_rate,
                "clipnorm": config.clipnorm
            }

        # Build and train the model
        model = build_lstm_model((metadata["max_sequence_length"], 1), best_params)
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