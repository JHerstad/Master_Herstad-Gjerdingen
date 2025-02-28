#!/usr/bin/env python3
"""
Module for evaluating machine learning models, specifically LSTM models for RUL regression on the Aachen dataset.

This module provides functions to evaluate model performance, including loss and metrics like MAE,
with rescaling for RUL regression. It integrates with src/models.py and src/preprocessing.py,
ensuring reproducibility and professionalism for thesis experiments.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple



def evaluate_lstm_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, y_max: float) -> Tuple[float, float]:
    """
    Evaluates an LSTM model on the test set for RUL regression, rescaling predictions to the original range.

    This function computes the test loss and mean absolute error (MAE) for the LSTM model,
    rescales the predictions and actual values using y_max, and logs the results for reproducibility.

    Args:
        model (tf.keras.Model): The trained LSTM model to evaluate.
        X_test (np.ndarray): Test input sequences with shape (n_samples, seq_len, 1).
        y_test (np.ndarray): Test target values (normalized) with shape (n_samples,).
        y_max (float): The maximum RUL value used for normalizing the targets, for rescaling predictions.

    Returns:
        Tuple[float, float]: A pair containing (test_loss, test_mae_rescaled), where:
            - test_loss is the model's loss (e.g., MSE) on the test set.
            - test_mae_rescaled is the mean absolute error after rescaling to the original RUL range.

    Notes:
        - Assumes the model was compiled with 'mse' loss and 'mae' metric.
        - Uses verbose=1 in model.evaluate() for progress reporting during evaluation.
        - Logs results using the logging module for professional tracking in a thesis context.
    """
    # Evaluate the model on the test set, reporting progress
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    
    # Generate predictions for the test set silently
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Rescale predictions and actual values to the original RUL range using y_max
    y_pred_rescaled = y_pred * y_max
    y_test_rescaled = y_test * y_max
    
    # Calculate the rescaled MAE for better interpretability
    mae_rescaled = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
    
    # Log the evaluation results for reproducibility and thesis documentation
    logger.info("LSTM model evaluated - Test Loss: %.4f, Test MAE (rescaled): %.4f", test_loss, mae_rescaled)
    
    return test_loss, mae_rescaled