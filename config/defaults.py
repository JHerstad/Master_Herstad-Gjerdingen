"""
Module defining default configuration settings for preprocessing and modeling the Aachen
battery degradation dataset. This configuration is used across preprocessing, modeling, and
explainability scripts to ensure consistency and reproducibility.
"""

from dataclasses import dataclass, field
import os
import json


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
        classification (bool): If True, prepare data for classification (CNN); if False, for regression (LSTM).
        seq_len (int): Configurable sequence length for both classification and regression.
        train_split_ratio (float): Ratio of data to use for training (e.g., 0.8 for 80% training).
        val_split_ratio (float): Ratio of training data to use for validation (e.g., 0.2 for 20% validation).
        batch_size (int): Batch size for training and hyperparameter tuning.
        max_trials (int): Maximum number of hyperparameter combinations to try in tuning.
        tuning_epochs (int): Number of epochs for each trial during hyperparameter tuning.
        tuner_directory (str): Directory to store hyperparameter tuning results.
        bins (list): List of bin edges for RUL classification (e.g., [0, 200, 300, ...]).
        labels (list): List of labels for RUL bins (e.g., ["0-200", "200-300", ...]).
    """
    project_name: str = "Experiment1"
    data_path: str = os.path.join("data", "Aachen", "raw", "Degradation_Prediction_Dataset_ISEA.mat")
    
    # Preprocessing
    eol_capacity: float = 0.80  # Default EOL at 65% capacity
    test_cell_count: int = 3    # Default number of test cells
    random_state: int = 42      # Default random seed for reproducibility
    log_transform: bool = False # Default no log transform for RUL in regression
    classification: bool = False # Default to regression (LSTM); True for classification (CNN)
    seq_len: int = 20          # Default sequence length
    train_split_ratio: float = 0.8  # Default 80% of cells for training
    val_split_ratio: float = 0.2   # Default 20% of training cells for validation

    # Aachen
    bins: list = field(default_factory=lambda: [0, 200, 300, 400, 500, 600, 700, float("inf")])
    labels: list = field(default_factory=lambda: ["0-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700+"])

    # MIT_Stanford (Comment out for Aachen)
    bins: list = field(default_factory=lambda: [0, 200, 400, 600, float("inf")])
    labels: list = field(default_factory=lambda: ["0-200", "200-400", "400-600", "600+"])

    n_bins = 4

    # Grid Search
    batch_size: int = 32        # Default batch size for training/tuning
    max_trials: int = 20        # Default number of trials for hyperparameter tuning
    tuning_epochs: int = 50     # Default epochs for tuning
    tuner_directory: str = os.path.join("experiments", "hyperparameter_tuning")

    # To be deleted - Start
    # LSTM model
    lstm_units: int = 32
    lstm_dropout_rate: float = 0.2
    lstm_dense_units: int = 16
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    epochs: int = 50
    patience: int = 20

    # CNN model
    conv1_filters: int = 32
    conv1_kernel_size: int = 11
    conv2_filters: int = 64
    conv2_kernel_size: int = 7
    conv3_filters: int = 64
    conv3_kernel_size: int = 5
    l2_reg: float = 0.001
    cnn_dense_units: int = 64
    cnn_dropout_rate: float = 0.2
    # To be deleted - End 

    def load_best_params(self, model_task: str = "lstm_regression", eol_capacity: float = None) -> None:
        """
        Load best hyperparameters from a tuning file and override defaults if available.

        Args:
            model_task (str): Type of model (e.g., "lstm_regression", "cnn_classification").
            eol_capacity (float, optional): EOL capacity to match the tuning file; defaults to self.eol_capacity.
        """
        if eol_capacity is None:
            eol_capacity = self.eol_capacity

        tuning_file = os.path.join(
            self.tuner_directory,
            f"{self.project_name}_{model_task}_tuning_eol{int(eol_capacity*100)}_best_params.json"
        )
        
        if os.path.exists(tuning_file):
            try:
                with open(tuning_file, 'r') as f:
                    best_params = json.load(f)
                for key, value in best_params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        print(f"Warning: Parameter '{key}' from {tuning_file} not found in Config; ignoring.")
                print(f"Loaded best hyperparameters from {tuning_file}: {best_params}")
            except Exception as e:
                print(f"Error loading best params from {tuning_file}: {str(e)}")
        else:
            print(f"No tuning file found at {tuning_file}; using defaults.")


if __name__ == "__main__":
    config = Config()
    print("Default Configuration:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")