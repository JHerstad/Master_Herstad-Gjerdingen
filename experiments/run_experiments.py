"""
Script to preprocess the Aachen battery degradation dataset for RUL classification and regression,
and load the preprocessed data from data/processed/ for further experiments.

This script ensures professional reproducibility by leveraging stored preprocessed data,
aligning with thesis standards for data management and experimentation.
"""

import os
import sys
import numpy as np
import logging
import json
from src.preprocessing import preprocess_aachen_dataset
from config.defaults import Config

# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_preprocessed_data(model_type: str, eol_capacity: float) -> tuple:
    """
    Loads preprocessed data from data/processed/ based on model type and EOL capacity.

    Args:
        model_type (str): Either "classification" or "regression".
        eol_capacity (float): EOL capacity fraction (e.g., 0.65 for EOL65).

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, metadata)
    """
    output_dir = "data/processed/"
    eol_str = f"eol{int(eol_capacity*100)}"

    # Find the most recent files for the given model type and EOL capacity
    pattern = f"X_train_{model_type}_{eol_str}_*.npy"
    files = [f for f in os.listdir(output_dir) if f.startswith(f"X_train_{model_type}_{eol_str}_") and f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No preprocessed data found for {model_type} with EOL {eol_capacity}")

    latest_file = max(files, key=lambda x: x.split("_")[-1].split(".")[0])
    timestamp = latest_file.split("_")[-1].split(".")[0]

    # Load data arrays
    X_train = np.load(os.path.join(output_dir, f"X_train_{model_type}_{eol_str}_{timestamp}.npy"))
    X_val = np.load(os.path.join(output_dir, f"X_val_{model_type}_{eol_str}_{timestamp}.npy"))
    X_test = np.load(os.path.join(output_dir, f"X_test_{model_type}_{eol_str}_{timestamp}.npy"))
    y_train = np.load(os.path.join(output_dir, f"y_train_{model_type}_{eol_str}_{timestamp}.npy"))
    y_val = np.load(os.path.join(output_dir, f"y_val_{model_type}_{eol_str}_{timestamp}.npy"))
    y_test = np.load(os.path.join(output_dir, f"y_test_{model_type}_{eol_str}_{timestamp}.npy"))

    # Load metadata
    metadata_file = [f for f in os.listdir(output_dir) if f.startswith(f"metadata_{model_type}_{eol_str}_{timestamp}") and f.endswith(".json")][0]
    with open(os.path.join(output_dir, metadata_file), "r") as f:
        metadata = json.load(f)[model_type]

    logger.info(f"Loaded preprocessed data for {model_type} with EOL {eol_capacity} and timestamp {timestamp}")
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata

def run_preprocessing_and_load():
    """
    Preprocesses the Aachen dataset for both classification and regression, stores the data,
    and loads it for further use, ensuring reproducibility and professionalism.
    """
    config = Config()
    output_dir = "data/processed/"
    results_dir = "experiments/results/"
    os.makedirs(results_dir, exist_ok=True)

    # Preprocess and store data for classification (CNN)
    logger.info("Starting preprocessing for classification (CNN) with EOL %f", config.eol_capacity)
    try:
        preprocessed_classification = preprocess_aachen_dataset(
            config.data_path,
            eol_capacity=config.eol_capacity,
            test_cell_count=config.test_cell_count,
            random_state=config.random_state,
            log_transform=config.log_transform,
            classification=True
        )

        # Load preprocessed classification data
        X_train_cls, X_val_cls, X_test_cls, y_train_cls, y_val_cls, y_test_cls, metadata_cls = load_preprocessed_data(
            "classification", config.eol_capacity
        )

        # Log loaded data shapes for verification
        logger.info("Classification data loaded - X_train shape: %s, y_train shape: %s", str(X_train_cls.shape), str(y_train_cls.shape))
        logger.info("Classification metadata: %s", str(metadata_cls))

        # Optionally save loaded data shapes to results for documentation
        np.save(os.path.join(results_dir, f"classification_shapes_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"), {
            "X_train_shape": X_train_cls.shape,
            "X_val_shape": X_val_cls.shape,
            "X_test_shape": X_test_cls.shape,
            "y_train_shape": y_train_cls.shape,
            "y_val_shape": y_val_cls.shape,
            "y_test_shape": y_test_cls.shape
        })

    except Exception as e:
        logger.error("Error in classification preprocessing or loading: %s", str(e))
        raise

    # Preprocess and store data for regression (LSTM)
    logger.info("Starting preprocessing for regression (LSTM) with EOL %f", config.eol_capacity)
    try:
        preprocessed_regression = preprocess_aachen_dataset(
            config.data_path,
            eol_capacity=config.eol_capacity,
            test_cell_count=config.test_cell_count,
            random_state=config.random_state,
            log_transform=config.log_transform,
            classification=False
        )

        # Load preprocessed regression data
        X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg, metadata_reg = load_preprocessed_data(
            "regression", config.eol_capacity
        )

        # Log loaded data shapes for verification
        logger.info("Regression data loaded - X_train shape: %s, y_train shape: %s", str(X_train_reg.shape), str(y_train_reg.shape))
        logger.info("Regression metadata: %s", str(metadata_reg))

        # Optionally save loaded data shapes to results for documentation
        np.save(os.path.join(results_dir, f"regression_shapes_eol{int(config.eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"), {
            "X_train_shape": X_train_reg.shape,
            "X_val_shape": X_val_reg.shape,
            "X_test_shape": X_test_reg.shape,
            "y_train_shape": y_train_reg.shape,
            "y_val_shape": y_val_reg.shape,
            "y_test_shape": y_test_reg.shape
        })

    except Exception as e:
        logger.error("Error in regression preprocessing or loading: %s", str(e))
        raise

    # Return loaded data for further use (optional, for testing or scripts)
    return {
        "classification": (X_train_cls, X_val_cls, X_test_cls, y_train_cls, y_val_cls, y_test_cls, metadata_cls),
        "regression": (X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg, metadata_reg)
    }

if __name__ == "__main__":
    # Add thesis_experiment/ to sys.path for imports
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    
    # Run preprocessing and load data
    loaded_data = run_preprocessing_and_load()
    logger.info("Preprocessing and loading completed successfully")