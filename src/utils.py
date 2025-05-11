#!/usr/bin/env python3
"""
Module for visualization utilities to evaluate RUL regression and classification models on the
Aachen or MIT_Stanford datasets. Provides plotting functions (scatterplots, histograms, training history)
for model performance assessment, ensuring reusability across models (e.g., LSTM, CNN) for thesis experiments.
"""

# Standard library imports
import datetime
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from config.defaults import Config

def plot_predictions_vs_actual(config:Config, y_test: np.ndarray, y_pred: np.ndarray, y_max: float, title: "str" = "Predicted vs Actual RUL"):
    """
    Creates a scatterplot comparing actual vs. predicted Remaining Useful Life (RUL) values,
    rescaled to their original range for better interpretability.

    This visualization is useful for assessing the performance of regression models (e.g., LSTM)
    on the Aachen dataset, highlighting the agreement between predicted and actual RUL values.

    Args:
        config (Config): Configuration object with attributes including model_task, eol_capacity, and use_aachen.
        y_test (np.ndarray): Test target values (normalized) with shape (n_samples,).
        y_pred (np.ndarray): Predicted values (normalized) with shape (n_samples,).
        y_max (float): The maximum RUL value used for normalizing the targets, used for rescaling
                    predictions and actual values to their original range.
        title (str): Title for the scatterplot (default: "Predicted vs Actual RUL").

    Notes:
        - Saves the plot to experiments/results/ with a filename including the EOL capacity
        and timestamp for versioning and reproducibility.
        - Uses matplotlib for plotting, with a red dashed line representing perfect predictions.
        - Displays the plot using plt.show(), suitable for interactive use in thesis experiments.

    Returns:
        None
    """
    # Set the font to Times New Roman globally for this plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Rescale the normalized values to their original RUL range using y_max
    y_test_rescaled = y_test * y_max
    y_pred_rescaled = y_pred.flatten() * y_max
    
    # Create a scatter plot comparing actual vs. predicted RUL
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.7)
    plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 
             'r--', label='Perfect Prediction')
    plt.title(title, fontsize=16)
    plt.xlabel("Actual RUL", fontsize=14)
    plt.ylabel("Predicted RUL", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid()

    # Determine the subfolder based on config.use_aachen
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    output_dir = os.path.join("experiments", "results", bottom_map_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build filename using config.model_task and config.eol_capacity along with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.model_task}_predictions_eol{int(config.eol_capacity * 100)}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, y_max: float):
    """
    Generates a histogram of residuals (differences between actual and predicted RUL values),
    rescaled to their original range for better interpretability.

    This visualization helps assess the distribution of prediction errors for regression models
    (e.g., LSTM) on the Aachen dataset, aiding in model evaluation for thesis experiments.

    Args:
        y_test (np.ndarray): Test target values (normalized) with shape (n_samples,).
        y_pred (np.ndarray): Predicted values (normalized) with shape (n_samples,).
        y_max (float): The maximum RUL value used for normalizing the targets, used for rescaling
                    residuals to their original range.

    Notes:
        - Saves the histogram to experiments/results/ with a filename including the EOL capacity
        (from Config) and timestamp for versioning and reproducibility.
        - Uses 30 bins and blue bars with black edges for clarity, suitable for thesis visualizations.
        - Closes the plot after saving to manage memory, ideal for batch processing in thesis experiments.

    Returns:
        None
    """
    # Rescale the normalized values to their original RUL range using y_max
    y_test_rescaled = y_test * y_max
    y_pred_rescaled = y_pred.flatten() * y_max
    
    # Calculate residuals as the difference between actual and predicted values
    residuals = y_test_rescaled - y_pred_rescaled
    
    # Create a histogram of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Residuals Histogram")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid()
    
    # Save the histogram with versioning based on EOL capacity and timestamp
    output_dir = os.path.join("experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"lstm_residuals_eol{int(Config().eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_training_history(history: dict, model_task: str):
    """
    Plots the training and validation metrics (loss and task-specific metric) over epochs dynamically
    based on model_task.

    Args:
        history (dict): Training history containing 'loss', 'val_loss', and task-specific metrics
                    ('mae' for regression, 'accuracy' for classification).
        model_task (str): Combined model and task identifier, e.g., "lstm_regression" or "cnn_classification".

    Notes:
        - Plots both loss and a task-specific metric (MAE for regression, accuracy for classification).
        - Displays the plot using plt.show(), suitable for interactive use in thesis experiments.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # Plot loss (common to both regression and classification)
    plt.plot(history['loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')

    # Determine task type and additional metric
    if "regression" in model_task:
        metric_key = 'mae'
        metric_label = 'MAE'
        model_name = "LSTM" if "lstm" in model_task else "CNN"
        plt.title(f"Training and Validation Loss/MAE for {model_name} Regression")
    elif "classification" in model_task:
        metric_key = 'accuracy'
        metric_label = 'Accuracy'
        model_name = "CNN" if "cnn" in model_task else "LSTM"
        plt.title(f"Training and Validation Loss/Accuracy for {model_name} Classification")
    else:
        raise ValueError(f"Unsupported model_task: {model_task}. Must contain 'regression' or 'classification'.")

    # Plot the task-specific metric if available
    if metric_key in history:
        plt.plot(history[metric_key], label=f'Training {metric_label}', marker='x')
    if f'val_{metric_key}' in history:
        plt.plot(history[f'val_{metric_key}'], label=f'Validation {metric_label}', marker='x')

    plt.xlabel("Epochs")
    plt.ylabel(f"Loss and {metric_label}")
    plt.legend()
    plt.grid()
    plt.show()