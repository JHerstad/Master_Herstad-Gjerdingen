#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Johannes Herstad and Sigurd Gjerdingen
# SPDX-License-Identifier: MIT

"""
Script for performing hyperparameter tuning on LSTM or CNN models using Keras Tuner.
Supports regression (LSTM) and classification (CNN) tasks, integrating with src/models.py,
src/evaluation.py, and src/visualization.py. Saves best hyperparameters in JSON.
"""

import os
import json
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import logging
from src.models import load_preprocessed_data
from config.defaults import Config
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_lstm_model(hp, input_shape):
    """
    Keras Tuner-compatible function for hyperparameter tuning of the LSTM model (regression) using Functional API.

    Args:
        hp: Hyperparameters object from Keras Tuner.
        input_shape: Tuple of (seq_len, 1) for input sequences.

    Returns:
        Compiled LSTM model with tunable hyperparameters for RUL regression.
    """
    # Define input
    inputs = Input(shape=input_shape)

    # LSTM layer
    x = LSTM(
        units=hp.Int("lstm_units", min_value=16, max_value=64, step=16),
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=False,
        unroll=False,
        name="lstm"  # Explicitly named for stability analysis
    )(inputs)

    # Dropout layer
    x = Dropout(
        rate=hp.Float("lstm_dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    )(x)

    # Dense layer
    x = Dense(
        units=hp.Int("lstm_dense_units", min_value=8, max_value=64, step=8),
        activation='tanh'
    )(x)

    # Output layer
    outputs = Dense(1)(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
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
    
    # First convolutional layer
    x = Conv1D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv1_kernel_size', values=[3, 5, 7]),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.0005, 0.0001])),
        padding='same'
    )(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Second convolutional layer
    x = Conv1D(
        filters=hp.Int('conv2_filters', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv2_kernel_size', values=[1, 3, 5]),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.001, 0.0005, 0.0001])),
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Flattening or using GlobalAveragePooling
    x = Flatten()(x)
    
    # Dense layer
    x = Dense(
        units=hp.Int('cnn_dense_units', min_value=32, max_value=128, step=32),
        activation='relu',
        kernel_regularizer=l2(hp.Choice('l2_reg', values=[0.01, 0.001, 0.0005]))
    )(x)
    x = Dropout(rate=hp.Float('cnn_dropout_rate', min_value=0.3, max_value=0.7, step=0.1))(x)
    
    # Output layer
    output_layer = Dense(Config.n_bins, activation='softmax')(x)

    # Build and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(
        learning_rate=hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3])
    )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_grid_search_models(config):
    """
    Runs grid search hyperparameter tuning for scikit-learn regression models
    (Decision Tree, Linear Regression, Lasso Regression) based on config.model_task.
    
    This function loads preprocessed data (using config.eol_capacity and config.use_aachen),
    flattens the training data, and then, based on the model_task specified in the config,
    performs a grid search over the defined parameter grid. The best hyperparameters
    are then saved to a JSON file in a subfolder based on the bottom map configuration.
    
    Args:
        config (Config): Configuration object with the following relevant attributes:
            - model_task: a string that should include 'dt' for Decision Tree,
                          'lr' for Linear Regression, or 'lasso' for Lasso Regression.
            - eol_capacity: float value for EOL capacity.
            - use_aachen: bool to determine the dataset to load.
            - tuner_directory: base directory for saving hyperparameter tuning results.
            - project_name: name used for naming the output file.
    
    Returns:
        dict: Best hyperparameters found for the specified model.
    """
    # Load preprocessed data using config parameters
    X_train, _, _, y_train, _, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )
    
    # Flatten the input for scikit-learn models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Get the model task in lowercase
    model_task = config.model_task.lower()
    
    best_params = None
    if "dt" in model_task:
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_param_grid = {
            'max_depth': [1, 2, 3, 4],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Decision Tree parameters: {best_params}")
    elif "lr" in model_task:
        lr_model = LinearRegression()
        lr_param_grid = {'fit_intercept': [True, False]}
        grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Linear Regression parameters: {best_params}")
    elif "lasso" in model_task:
        lasso_model = Lasso()
        lasso_param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_iter': [1000, 5000, 10000],
            'tol': [0.0001, 0.001],
            'selection': ['cyclic', 'random']
        }
        grid_search = GridSearchCV(lasso_model, lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Lasso Regression parameters: {best_params}")
    else:
        raise ValueError("Unsupported model_task for grid search. Must contain 'dt', 'lr', or 'lasso'.")

    # Save best hyperparameters to JSON in the appropriate folder
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    tuner_directory = os.path.join(config.tuner_directory, bottom_map_dir)
    os.makedirs(tuner_directory, exist_ok=True)
    output_file = os.path.join(
        tuner_directory,
        f"{config.project_name}_{config.model_task}_tuning_eol{int(config.eol_capacity*100)}_best_params.json"
    )
    try:
        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best hyperparameters saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save hyperparameters to {output_file}: {str(e)}")
    
    return best_params


def run_hyperparameter_search(config: Config):
    """
    Runs hyperparameter tuning using Keras Tuner with Bayesian Optimization.
    Saves best hyperparameters and the best model into a subfolder determined by config.use_aachen.

    Args:
        config (Config): Configuration object with model and tuning parameters.
        model_task (str): Combined model and task identifier, e.g., "lstm_regression" or "cnn_classification".
                         Default is "lstm_regression".

    Returns:
        dict: Best hyperparameters found during the search.
    """
    logger.info(f"Running hyperparameter search for: {config.model_task}")

    # Load preprocessed data
    logger.info(f"Loading preprocessed data for hyperparameter tuning with {config.model_task}...")
    X_train, X_val, _, y_train, y_val, _, metadata = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )
    
    input_shape = (metadata["seq_len"], 1)
    logger.info(f"Input shape: {input_shape}, X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Select model-building function and objective
    if "lstm" in config.model_task:
        if "regression" not in config.model_task:
            logger.warning(f"{config.model_task} specified, but LSTM typically used for regression.")
        build_fn = lambda hp: build_lstm_model(hp, input_shape)
        objective = "val_loss"  # Regression uses loss (MSE)
    elif "cnn" in config.model_task:
        if "classification" not in config.model_task:
            logger.warning(f"{config.model_task} specified, but CNN typically used for classification.")
        build_fn = lambda hp: build_cnn_model(hp, input_shape)
        objective = "val_loss"  # Use val_loss for consistency (or val_accuracy for classification)
    else:
        raise ValueError(f"Unsupported model_task: {config.model_task}. Must include 'lstm' or 'cnn'.")

    # Determine bottom map subfolder based on configuration flag
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    # Set tuner directory to include the bottom_map subfolder
    tuner_directory = os.path.join(config.tuner_directory, bottom_map_dir)
    os.makedirs(tuner_directory, exist_ok=True)

    # Set up the Keras Tuner
    tuner = kt.BayesianOptimization(
        build_fn,
        objective=objective,
        max_trials=config.max_trials,
        executions_per_trial=1,
        directory=tuner_directory,
        project_name=f"{config.project_name}_{config.model_task}_tuning_eol{int(config.eol_capacity*100)}"
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

    # Save best hyperparameters to JSON in the appropriate folder
    output_file = os.path.join(
        tuner_directory,
        f"{config.project_name}_{config.model_task}_tuning_eol{int(config.eol_capacity*100)}_best_params.json"
    )
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best hyperparameters saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save hyperparameters to {output_file}: {str(e)}")



    return best_params

def run_grid_search_models(config):
    """
    Runs grid search hyperparameter tuning for scikit-learn models based on config.model_task.
    Supports regression (Decision Tree, Linear Regression, Lasso Regression) and classification
    (Decision Tree Classifier, Logistic Regression) models.

    This function loads preprocessed data (using config.eol_capacity and config.use_aachen),
    flattens the training data, and performs a grid search over the defined parameter grid based
    on the model_task specified in the config. The best hyperparameters are saved to a JSON file
    in a subfolder based on the bottom map configuration.

    Args:
        config (Config): Configuration object with the following relevant attributes:
            - model_task: a string that should include 'dt' for Decision Tree Regressor,
                          'lr' for Linear Regression, 'lasso' for Lasso Regression,
                          'dt_classification' for Decision Tree Classifier,
                          or 'lr_classification' for Logistic Regression.
            - eol_capacity: float value for EOL capacity.
            - use_aachen: bool to determine the dataset to load.
            - tuner_directory: base directory for saving hyperparameter tuning results.
            - project_name: name used for naming the output file.

    Returns:
        dict: Best hyperparameters found for the specified model.
    """
    # Load preprocessed data using config parameters
    X_train, _, _, y_train, _, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )

    # Flatten the input for scikit-learn models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # Get the model task in lowercase
    model_task = config.model_task.lower()

    best_params = None
    if "dt_classification" in model_task:
        # Decision Tree Classifier
        y_train = np.argmax(y_train, axis=1)  # Convert one-hot encoded labels to class indices
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_param_grid = {
            'max_depth': [2, 3, 5],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Decision Tree Classifier parameters: {best_params}")
        logger.info(f"Best cross-validation accuracy: {grid_search.best_score_}")
    elif "lr_classification" in model_task:
        # Logistic Regression
        y_train = np.argmax(y_train, axis=1)  # Convert one-hot encoded labels to class indices
        lr_clf = LogisticRegression(random_state=42)
        lr_param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'max_iter': [1000]
        }
        grid_search = GridSearchCV(lr_clf, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Logistic Regression parameters: {best_params}")
        logger.info(f"Best cross-validation accuracy: {grid_search.best_score_}")
    elif "dt" in model_task:
        # Decision Tree Regressor
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_param_grid = {
            'max_depth': [1, 2, 3, 4],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Decision Tree parameters: {best_params}")
    elif "lr" in model_task:
        # Linear Regression
        lr_model = LinearRegression()
        lr_param_grid = {'fit_intercept': [True, False]}
        grid_search = GridSearchCV(lr_model, lr_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Linear Regression parameters: {best_params}")
    elif "lasso" in model_task:
        # Lasso Regression
        lasso_model = Lasso()
        lasso_param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_iter': [1000, 5000, 10000],
            'tol': [0.0001, 0.001],
            'selection': ['cyclic', 'random']
        }
        grid_search = GridSearchCV(lasso_model, lasso_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_flat, y_train)
        best_params = grid_search.best_params_
        logger.info(f"Best Lasso Regression parameters: {best_params}")
    else:
        raise ValueError(
            "Unsupported model_task. Must contain 'dt', 'lr', 'lasso', 'dt_classification', or 'lr_classification'."
        )

    # Save best hyperparameters to JSON in the appropriate folder
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    tuner_directory = os.path.join(config.tuner_directory, bottom_map_dir)
    os.makedirs(tuner_directory, exist_ok=True)
    output_file = os.path.join(
        tuner_directory,
        f"{config.project_name}_{config.model_task}_tuning_eol{int(config.eol_capacity*100)}_best_params.json"
    )
    try:
        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best hyperparameters saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save hyperparameters to {output_file}: {str(e)}")

    return best_params
