"""
Module defining default configuration settings for preprocessing and modeling the Aachen
battery degradation dataset. This configuration is used across preprocessing, modeling, and
explainability scripts to ensure consistency and reproducibility.
"""

from dataclasses import dataclass
import os


@dataclass
class Config:
    """
    Configuration class for the Aachen dataset preprocessing and modeling pipeline.

    Attributes:
        project_name (str): Name of the project for organizing experiments and tuner results.
        data_path (str): Path to the '.mat' file containing the Aachen dataset.
        eol_capacity (float): Fraction of initial capacity defining End-of-Life (EOL), e.g., 0.65 for 65%.
        test_cell_count (int): Number of unique cells to hold out for testing.
        random_state (int): Random seed for reproducibility of splits and sampling.
        log_transform (bool): Whether to apply a log transform to RUL values for regression (LSTM) models.
        classification (bool): If True, prepare data for classification (CNN) with fixed sequence length;
                             if False, prepare for regression (LSTM) with variable length.
        seq_len (int): Configurable sequence length for both classification (fixed) and regression (optional truncation).
        train_split_ratio (float): Ratio of data to use for training (e.g., 0.8 for 80% training).
        val_split_ratio (float): Ratio of training data to use for validation (e.g., 0.2 for 20% validation).
        batch_size (int): Batch size for training and hyperparameter tuning.
        max_trials (int): Maximum number of hyperparameter combinations to try in tuning.
        tuning_epochs (int): Number of epochs for each trial during hyperparameter tuning.
        tuner_directory (str): Directory to store hyperparameter tuning results.
    """
    project_name: str = "Experiment1"
    
    data_path: str = os.path.join("data", "raw", "Degradation_Prediction_Dataset_ISEA.mat")
    
    # Preprocessing
    eol_capacity: float = 0.65  # Default EOL at 65% capacity
    test_cell_count: int = 3    # Default number of test cells
    random_state: int = 42      # Default random seed for reproducibility
    log_transform: bool = False # Default no log transform for RUL in regression
    classification: bool = False # Default to regression (LSTM), False; True for classification (CNN)
    seq_len: int = 120          # Default sequence length for fixed-length classification or optional truncation in regression
    train_split_ratio: float = 0.8  # Default 80% of cells for training
    val_split_ratio: float = 0.2   # Default 20% of training cells for validation
    
    # LSTM model
    lstm_units = 32  # Number of LSTM units (default, can be overridden by hyperparameter tuning)
    dropout_rate = 0.2  # Dropout rate for regularization
    dense_units = 16  # Number of units in the fully connected layer
    learning_rate = 0.001  # Learning rate for Adam optimizer
    clipnorm = 1.0  # Gradient clipping norm for stable training
    batch_size = 32  # Mini-batch size
    epochs = 50  # Maximum number of training epochs
    patience = 5  # Early stopping patience


    # LSTM Grid Search
    batch_size: int = 32        # Default batch size for training/tuning
    max_trials: int = 3        # Default number of trials for hyperparameter tuning
    tuning_epochs: int = 3      # Default epochs for tuning (kept low for speed)
    tuner_directory: str = os.path.join("experiments", "hyperparameter_tuning")  # Default directory for tuning results


if __name__ == "__main__":
    # Example usage for testing or validation
    config = Config()
    print("Default Configuration:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")