"""
Module defining default configuration settings for preprocessing and modeling the Aachen
battery degradation dataset. This configuration is used across preprocessing, modeling, and
explainability scripts to ensure consistency and reproducibility.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    """
    Configuration class for the Aachen dataset preprocessing and modeling pipeline.

    Attributes:
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
    """
    data_path: str = os.path.join("data", "raw", "Degradation_Prediction_Dataset_ISEA.mat")
    eol_capacity: float = 0.65  # Default EOL at 65% capacity
    test_cell_count: int = 3    # Default number of test cells
    random_state: int = 42      # Default random seed for reproducibility
    log_transform: bool = False  # Default no log transform for RUL in regression
    classification: bool = False # Default to regression (LSTM), False; True for classification (CNN)
    seq_len: int = 120          # Default sequence length for fixed-length classification or optional truncation in regression
    train_split_ratio: float = 0.8  # Default 80% of cells for training
    val_split_ratio: float = 0.2   # Default 20% of training cells for validation
    lstm_units: int = 32
    dropout_rate: float = 0.2
    dense_units: int = 16
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    patience: int = 15
    batch_size: int = 32
    epochs: int = 50


if __name__ == "__main__":
    # Example usage for testing or validation
    config = Config()
    print("Default Configuration:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")