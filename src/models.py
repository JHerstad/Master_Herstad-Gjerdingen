#!/usr/bin/env python3
"""
Module for defining, training, and loading machine learning models (LSTM, CNN, Decision Tree, Linear Regression, Lasso) 
for RUL prediction on Aachen or MIT_Stanford datasets. Integrates with preprocessing and configuration modules for 
thesis experiments, focusing on reproducibility and hyperparameter tuning.
"""

# Standard library imports
import os
import sys
import logging
import datetime
import json

# Third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
import joblib
import fnmatch

# Local imports
from config.defaults import Config


# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add thesis_experiment/ to sys.path for imports (if needed when run standalone)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def load_preprocessed_data(model_task: str, eol_capacity: float, use_aachen: bool):
    """
    Loads preprocessed data and metadata from data/processed/ based on model_task and EOL capacity.

    Args:
        model_task (str): Combined model and task identifier, e.g., "lstm_regression" or "cnn_classification".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test, metadata),
               where metadata includes y_max, seq_len, eol_capacity, classification, and timestamp.
    """
    # Choose dataset based on the use_aachen flag.
    dataset = "Aachen" if use_aachen else "MIT_Stanford"
    output_dir = f"data/{dataset}/processed/"

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
        raise FileNotFoundError(f"No preprocessed data found for {task_type} with EOL {eol_capacity}.")

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

def train_lstm_model(config: Config):
    """
    Trains an LSTM model for RUL regression using tuned hyperparameters from config with Functional API.
    Preprocessed data is loaded internally based on config.model_task, config.eol_capacity, and config.use_aachen.
    
    Args:
        config (Config): Configuration object with tuned hyperparameters and data parameters including
                         model_task, eol_capacity, use_aachen, lstm_units, lstm_dropout_rate, lstm_dense_units,
                         learning_rate, clipnorm, patience, epochs, and batch_size.
    
    Returns:
        Tuple: (trained model, training history).
    """
    # Load preprocessed data using config parameters.
    X_train, X_val, _, y_train, y_val, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )

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

    # Determine the subfolder based on configuration (Aachen or MIT_Stanford)
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    base_model_dir = os.path.join("experiments", "models", bottom_map_dir)
    os.makedirs(base_model_dir, exist_ok=True)

    # Define callbacks and file paths using the folder structure from above
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_model_path = os.path.join(
        base_model_dir,
        f"lstm_regression_eol{int(config.eol_capacity * 100)}_{timestamp}_best.keras"
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            best_model_path,
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
        callbacks=callbacks
    )

    # Save the final model
    final_model_path = os.path.join(
        base_model_dir,
        f"lstm_regression_eol{int(config.eol_capacity * 100)}_{timestamp}_final.keras"
    )
    model.save(final_model_path)
    logger.info(f"Final LSTM model saved to {final_model_path}")

    return model, history.history

def train_cnn_model(config: Config):
    """
    Trains a CNN model for RUL classification using tuned hyperparameters from config, with Functional API.

    Args:
        config (Config): Configuration object with tuned hyperparameters.
        X_train, y_train, X_val, y_val (np.ndarray): Training and validation data.

    Returns:
        Tuple: (trained model, training history).
    """
    # Load preprocessed data using config parameters.
    X_train, X_val, _, y_train, y_val, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )

    # Derive input shape from X_train
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Define the model using Functional API
    input_layer = Input(shape=input_shape)
    
    # First Conv1D block
    x = Conv1D(
        filters=config.conv1_filters,
        kernel_size=config.conv1_kernel_size,
        activation='relu',
        kernel_regularizer=l2(config.l2_reg)
    )(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Second Conv1D block
    x = Conv1D(
        filters=config.conv2_filters,
        kernel_size=config.conv2_kernel_size,
        activation='relu',
        kernel_regularizer=l2(config.l2_reg)
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    """    # Third Conv1D block
        x = Conv1D(
            filters=config.conv3_filters,
            kernel_size=config.conv3_kernel_size,
            activation='relu',
            kernel_regularizer=l2(config.l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)"""
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(
        units=config.cnn_dense_units,
        activation='relu',
        kernel_regularizer=l2(config.l2_reg)
    )(x)
    x = Dropout(rate=config.cnn_dropout_rate)(x)
    output_layer = Dense(config.n_bins, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("CNN model built with tuned config: %s", str(config))
    
    # Determine subfolder based on bottom map configuration
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    base_model_dir = os.path.join("experiments", "models", bottom_map_dir)
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Define callbacks with a timestamp for consistent naming
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(
        base_model_dir,
        f"cnn_classification_eol{int(config.eol_capacity*100)}_{timestamp}_best.keras"
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config.patience, restore_best_weights=True),
        ModelCheckpoint(
            checkpoint_path,
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
        callbacks=callbacks
    )
    
    # Save the final model in the appropriate subfolder
    final_model_path = os.path.join(
        base_model_dir,
        f"cnn_classification_eol{int(config.eol_capacity*100)}_{timestamp}_final.keras"
    )
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    logger.info(f"Final CNN model saved to {final_model_path}")
    
    return model, history.history


def train_dt_model(config: Config):
    """
    Trains the Decision Tree Regressor on the preprocessed data.
    Preprocessed data is loaded internally based on config.model_task, config.eol_capacity, 
    and config.use_aachen. The model is trained using the default tuning parameters provided in the config,
    the trained model is saved using joblib, and then returned.
    
    Args:
        config (Config): Configuration object with data and model parameters. It should include:
                         - max_depth, min_samples_split, and min_samples_leaf for the Decision Tree.
    
    Returns:
        DecisionTreeRegressor: The trained Decision Tree Regressor model.
    """
    # Load preprocessed data
    X_train, _, _, y_train, _, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )
    
    # Flatten training data (scikit-learn expects 2D arrays)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Create and train the Decision Tree Regressor using default parameters from config
    dt_regressor = DecisionTreeRegressor(
        random_state=42,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf
    )
    dt_regressor.fit(X_train_flat, y_train)
    
    # Determine subfolder based on bottom map configuration
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    base_model_dir = os.path.join("experiments", "models", bottom_map_dir)
    os.makedirs(base_model_dir, exist_ok=True)

    # Generate a timestamp for consistent naming
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Construct the filename with a .pkl extension and save the model using joblib
    model_file = os.path.join(
        base_model_dir,
        f"{config.project_name}_{config.model_task}_eol{int(config.eol_capacity * 100)}_{timestamp}_best.pkl"
    )
    joblib.dump(dt_regressor, model_file)
    
    return dt_regressor


def train_lr_model(config: Config):
    """
    Trains the Linear Regression model on the preprocessed data.
    Preprocessed data is loaded internally based on config.model_task, config.eol_capacity, 
    and config.use_aachen. The model is trained using the default tuning parameter (fit_intercept)
    provided in config, the trained model is saved using joblib, and then returned.
    
    Args:
        config (Config): Configuration object with data and model parameters. It should include:
                         - fit_intercept for the Linear Regression.
    
    Returns:
        LinearRegression: The trained Linear Regression model.
    """
    # Load preprocessed data
    X_train, _, _, y_train, _, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )
    
    # Flatten training data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Create and train the Linear Regression model using default parameters from config
    lr_model = LinearRegression(fit_intercept=config.fit_intercept)
    lr_model.fit(X_train_flat, y_train)
    
    # Determine subfolder based on bottom map configuration
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    base_model_dir = os.path.join("experiments", "models", bottom_map_dir)
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Generate a timestamp for consistent naming
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Construct the filename with a .pkl extension and save the model using joblib
    model_file = os.path.join(
        base_model_dir,
        f"{config.project_name}_{config.model_task}_eol{int(config.eol_capacity * 100)}_{timestamp}_best.pkl"
    )
    joblib.dump(lr_model, model_file)
    
    return lr_model


def train_lasso_model(config: Config):
    """
    Trains the Lasso Regression model on the preprocessed data.
    Preprocessed data is loaded internally based on config.model_task, config.eol_capacity, 
    and config.use_aachen. The model is trained using the default tuning parameters (alpha, max_iter, tol, selection)
    provided in config, the trained model is saved using joblib, and then returned.
    
    Args:
        config (Config): Configuration object with data and model parameters. It should include:
                         - alpha, max_iter, tol, selection for the Lasso Regression.
    
    Returns:
        Lasso: The trained Lasso Regression model.
    """
    # Load preprocessed data
    X_train, _, _, y_train, _, _, _ = load_preprocessed_data(
        config.model_task, config.eol_capacity, config.use_aachen
    )
    
    # Flatten training data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Create and train the Lasso Regression model using default parameters from config
    lasso_model = Lasso(
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        selection=config.selection
    )
    lasso_model.fit(X_train_flat, y_train)
    
    # Determine subfolder based on bottom map configuration
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    base_model_dir = os.path.join("experiments", "models", bottom_map_dir)
    os.makedirs(base_model_dir, exist_ok=True)
    
    # Generate a timestamp for consistent naming
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Construct the filename with a .pkl extension and save the model using joblib
    model_file = os.path.join(
        base_model_dir,
        f"{config.project_name}_{config.model_task}_eol{int(config.eol_capacity * 100)}_{timestamp}_best.pkl"
    )
    joblib.dump(lasso_model, model_file)
    
    return lasso_model



def load_saved_model(config: Config):
    """
    Loads a previously saved best model for the specified model task and EOL capacity.
    
    The folder used for the search is determined by the config.use_aachen flag: if True, the model is 
    looked for in the 'aachen' folder; otherwise, in the 'mit_stanford' folder.
    
    For Keras models (e.g., "lstm_regression" or "cnn_classification"), models are loaded using TensorFlow's load_model,
    and files are expected to have the .keras extension.
    
    For scikit-learn models (e.g., "dt_regression", "lr_regression", "lasso_regression"), models are loaded
    using joblib, and files are expected to have the .pkl extension.
    
    Args:
        config (Config): Configuration object with model parameters including model_task, eol_capacity, seq_len,
                         and use_aachen.
    
    Returns:
        Loaded model if successful; the type may be a tf.keras.Model for Keras models or a scikit-learn 
                       estimator if a pkl file is loaded. Returns None if no matching model file is found.
    """
    
    # Determine the appropriate folder based on config.use_aachen
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    model_dir = os.path.join("experiments", "models", bottom_map_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create an EOL string from the config
    eol_str = f"eol{int(config.eol_capacity * 100)}"
    
    # Determine the file pattern and loader based on model_task
    model_task_lower = config.model_task.lower()
    if model_task_lower in ["lstm_regression", "cnn_classification"]:
        pattern = f"{config.model_task}_{eol_str}_*_best.keras"
        loader = tf.keras.models.load_model
    else:
        pattern = f"{config.project_name}_{config.model_task}_eol{int(config.eol_capacity * 100)}_*_best.pkl"
        loader = joblib.load
    
    # List matching files and sort them (assuming filenames contain timestamps)
    files = os.listdir(model_dir)
    matching_files = [f for f in files if fnmatch.fnmatch(f, pattern)]
    if not matching_files:
        return None
    matching_files.sort()  # Sort alphabetically; the latest timestamp should appear last.
    latest_file = matching_files[-1]
    model_path = os.path.join(model_dir, latest_file) # Select the latest file
    
    try:
        model = loader(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None