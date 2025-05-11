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

    # Mode
    use_aachen: bool = True  # Default to Aachen dataset; set to False for MIT_Stanford
    model_task: str = "lstm_regression"  # Default model task; can be "lstm_regression" or "cnn_classification"


    # Preprocessing
    eol_capacity: float = 0.80  # Default EOL at 80% capacity 
    test_cell_count: int = 3    # Default number of test cells
    random_state: int = 42      # Default random seed for reproducibility
    log_transform: bool = False # Default no log transform for RUL in regression
    classification: bool = False # Default to regression (LSTM); True for classification (CNN)
    seq_len: int = 20          # Default sequence length
    train_split_ratio: float = 0.8  # Default 80% of cells for training
    val_split_ratio: float = 0.2   # Default 20% of training cells for validation

    
    # MIT_Stanford (Comment out for Aachen)
    bins: list = field(default_factory=lambda: [0, 200, 400, 600, float("inf")])
    labels: list = field(default_factory=lambda: ["0-200", "200-400", "400-600", "600+"])

    n_bins = 4

    # Aachen (Comment out for MIT_Stanford)
    #bins: list = field(default_factory=lambda: [0, 350, 700, 1050, float("inf")])
    #labels: list = field(default_factory=lambda: ["0-350", "350-700", "700-1050", "1050+"])


    # Grid Search
    batch_size: int = 32        # Default batch size for training/tuning
    max_trials: int = 20        # Default number of trials for hyperparameter tuning
    tuning_epochs: int = 50     # Default epochs for tuning
    tuner_directory: str = os.path.join("experiments", "hyperparameter_tuning")


    epochs: int = 100 # Default epochs for training used on LSTM and CNN models


    # To be deleted - Start
    # LSTM model
    lstm_units: int = 32
    lstm_dropout_rate: float = 0.2
    lstm_dense_units: int = 16
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    
    patience: int = 20

    # CNN model
    conv1_filters: int = 32
    conv1_kernel_size: int = 5
    conv2_filters: int = 64
    conv2_kernel_size: int = 3
    conv3_filters: int = 64
    conv3_kernel_size: int = 5
    l2_reg: float = 0.001
    cnn_dense_units: int = 64
    cnn_dropout_rate: float = 0.5
    # To be deleted - End
    
    # --- Grid Search Tuning Parameters for scikit-learn models ---
    # Decision Tree Regressor Tuning Parameters
    max_depth: any = None        # Candidate values: None, 3, 5, 10
    min_samples_split: int = 2   # Candidate values: 2, 5, 10
    min_samples_leaf: int = 1    # Candidate values: 1, 2, 4
    
    # Linear Regression Tuning Parameter
    fit_intercept: bool = True   # Candidate values: True, False
    
    # Lasso Regression Tuning Parameters
    alpha: float = 0.001         # Candidate values: 0.0001, 0.001, 0.01, 0.1, 1.0
    max_iter: int = 1000         # Candidate values: 1000, 5000, 10000
    tol: float = 0.001           # Candidate values: 0.0001, 0.001
    selection: str = "cyclic"    # Candidate values: 'cyclic', 'random' 

    # Decision Tree Classifier Tuning Parameters
    criterion: str = "entropy"      # Candidate values: 'gini', 'entropy'

    # Logistic Regression Tuning Parameters
    C: float = 1.0               # Candidate values: 0.01, 0.1, 1, 10, 100
    penalty: str = "l2"          # Candidate values: 'l2'
    max_iter: int = 1000         # Candidate values: 1000

    def load_best_params(self) -> None:
        """
        Load best hyperparameters from a tuning file and override defaults if available.

        Args:
            model_task (str): Type of model (e.g., "lstm_regression", "cnn_classification").
            eol_capacity (float, optional): EOL capacity to match the tuning file; defaults to self.eol_capacity.
        """
    

        # Determine the subfolder based on the configuration
        bottom_map_dir = "aachen" if self.use_aachen else "mit_stanford"

        # Build the full path to the tuning file using the new folder structure.
        tuning_file = os.path.join(
            self.tuner_directory,
            bottom_map_dir,
            f"{self.project_name}_{self.model_task}_tuning_eol{int(self.eol_capacity * 100)}_best_params.json"
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