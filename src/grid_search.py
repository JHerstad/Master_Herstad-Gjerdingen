#!/usr/bin/env python3
"""
Script for performing hyperparameter tuning on LSTM or CNN models using Keras Tuner.
Supports regression (LSTM) and classification (CNN) tasks, integrating with src/models.py,
src/evaluation.py, and src/visualization.py. Saves best hyperparameters in JSON.
"""

import os
import json
import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Masking, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import logging
from src.models import load_preprocessed_data
from config.defaults import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_lstm_model(hp, input_shape):
    """
    Keras Tuner-compatible function for hyperparameter tuning of the LSTM model (regression).

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
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            unroll=False
        ),
        Dropout(
            rate=hp.Float("lstm_dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
        ),
        Dense(
            units=hp.Int("lstm_dense_units", min_value=8, max_value=64, step=8),
            activation='tanh'
        ),
        Dense(1)  # Regression output
    ])
    optimizer = Adam(
        learning_rate=hp.Choice("learning_rate", values=[0.001, 0.01, 0.1]),
        clipnorm=hp.Float("clipnorm", min_value=0.5, max_value=1.5, step=0.5)
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def build_cnn_model(hp, input_shape):
    """
    Keras Tuner-compatible function for hyperparameter tuning of the CNN model (classification).

    Args:
        hp: Hyperparameters object from Keras Tuner.
        input_shape: Tuple of (seq_len, 1) for input sequences.

    Returns:
        Compiled CNN model with tunable hyperparameters for RUL classification.
    """
    input_layer = Input(shape=input_shape)
    x = Conv1D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('conv1_kernel_size', values=[9, 11, 13]),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.0005, 0.0001]))
    )(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(
        filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv2_kernel_size', values=[5, 7, 9]),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.0005, 0.0001]))
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(
        filters=hp.Int('conv3_filters', min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('conv3_kernel_size', values=[3, 5]),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.0005, 0.0001]))
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),  # CNN-specific, not prefixed
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.0005, 0.0001]))
    )(x)
    x = Dropout(rate=hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1))(x)  # CNN-specific, not prefixed
    output_layer = Dense(7, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-5])
    )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def run_hyperparameter_search(config: Config, model_type="lstm"):
    """
    Runs hyperparameter tuning using Keras Tuner with Bayesian Optimization.

    Args:
        config (Config): Configuration object with model and tuning parameters.
        model_type (str): "lstm" for regression or "cnn" for classification.

    Returns:
        dict: Best hyperparameters found during the search.
    """
    task_type = "classification" if model_type == "cnn" or config.classification else "regression"
    logger.info(f"Task type set to: {task_type}, Model type: {model_type}")

    # Load preprocessed data
    logger.info("Loading preprocessed data for hyperparameter tuning...")
    X_train, X_val, _, y_train, y_val, _, metadata = load_preprocessed_data(
        task_type, config.eol_capacity
    )
    input_shape = (metadata["max_sequence_length"], 1)

    # Select model-building function
    if model_type == "lstm":
        build_fn = lambda hp: build_lstm_model(hp, input_shape)
    elif model_type == "cnn":
        build_fn = lambda hp: build_cnn_model(hp, input_shape)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Set up the Keras Tuner
    tuner = kt.BayesianOptimization(
        build_fn,
        objective="val_accuracy" if model_type == "cnn" else "val_loss",
        max_trials=config.max_trials,
        executions_per_trial=1,
        directory=config.tuner_directory,
        project_name=f"{config.project_name}_{model_type}_tuning_eol{int(config.eol_capacity*100)}"
    )

    # Perform hyperparameter search
    logger.info("Starting hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        epochs=config.tuning_epochs,
        validation_data=(X_val, y_val),
        batch_size=config.batch_size,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.patience)]
    )

    # Get and save the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = best_hps.values
    logger.info(f"Best hyperparameters found: {best_params}")

    # Save to JSON
    output_file = os.path.join(
        config.tuner_directory,
        f"{config.project_name}_{model_type}_tuning_eol{int(config.eol_capacity*100)}_best_params.json"
    )
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Best hyperparameters saved to: {output_file}")

    return best_params

def main():
    config = Config()
    try:
        # Run for LSTM (regression)
        lstm_params = run_hyperparameter_search(config, model_type="lstm")
        print(f"Best LSTM hyperparameters: {lstm_params}")

        # Run for CNN (classification)
        cnn_params = run_hyperparameter_search(config, model_type="cnn")
        print(f"Best CNN hyperparameters: {cnn_params}")
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()