#!/usr/bin/env python3
"""
Module for utility functions related to visualization and data analysis for the Aachen dataset
models, specifically for RUL regression and classification experiments.

This module provides plotting functions for model evaluation, ensuring reusability across
different models (e.g., LSTM, CNN) and integration with src/models.py and src/preprocessing.py
for thesis reproducibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from config.defaults import Config
import os
import datetime
from typing import NoReturn

def plot_predictions_vs_actual(y_test: np.ndarray, y_pred: np.ndarray, y_max: float) -> NoReturn:
    """
    Creates a scatterplot comparing actual vs. predicted Remaining Useful Life (RUL) values,
    rescaled to their original range for better interpretability.

    This visualization is useful for assessing the performance of regression models (e.g., LSTM)
    on the Aachen dataset, highlighting the agreement between predicted and actual RUL values.

    Args:
        y_test (np.ndarray): Test target values (normalized) with shape (n_samples,).
        y_pred (np.ndarray): Predicted values (normalized) with shape (n_samples,).
        y_max (float): The maximum RUL value used for normalizing the targets, used for rescaling
                      predictions and actual values to their original range.

    Notes:
        - Saves the plot to experiments/results/ with a filename including the EOL capacity
          and timestamp for versioning and reproducibility.
        - Uses matplotlib for plotting, with a red dashed line representing perfect predictions.
        - Closes the plot after saving to free memory, suitable for batch processing in thesis experiments.
    """
    # Rescale the normalized values to their original RUL range using y_max
    y_test_rescaled = y_test * y_max
    y_pred_rescaled = y_pred.flatten() * y_max
    
    # Create a scatter plot comparing actual vs. predicted RUL
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_rescaled, y_pred_rescaled, alpha=0.7)
    plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 
             'r--', label='Perfect Prediction')
    plt.title("Predicted vs Actual RUL")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.legend()
    plt.grid()
    
    # Save the plot with versioning based on EOL capacity and timestamp
    output_dir = os.path.join("experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"lstm_predictions_eol{int(Config().eol_capacity*100)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, y_max: float) -> NoReturn:
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
          and timestamp for versioning and reproducibility.
        - Uses 30 bins and blue bars with black edges for clarity, suitable for thesis visualizations.
        - Closes the plot after saving to manage memory, ideal for batch processing in thesis experiments.
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