#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2025 Johannes Herstad and Sigurd Gjerdingen
# SPDX-License-Identifier: MIT

"""
Module for evaluating regression and classification models on test data. Supports Keras and scikit-learn models, 
computing metrics like RMSE, MAE, R^2, accuracy, and F1 score, and visualizing results with plots such as true vs. 
predicted scatter plots and confusion matrices.
"""

# Standard library imports
import datetime
import logging
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Local imports
from config.defaults import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_regression_model(model, X_test: np.ndarray, y_test: np.ndarray, y_max: float = None):
    """
    Evaluates a regression model on the test set. Works for:
    - Keras models (i.e., tf.keras.Model) by calling model.evaluate() and model.predict().
    - Non-Keras models (e.g., scikit-learn) by manually computing MSE and MAE.

    Args:
        model: The regression model to evaluate.
            If it's a Keras model, it must have `evaluate()` and `predict()` methods.
            Otherwise, it must have a `predict()` method (scikit-learn style).
        X_test (np.ndarray): Test features, shape depends on the model.
        y_test (np.ndarray): Test targets (normalized or unnormalized).
        y_max (float, optional): Maximum RUL (or similar scaling factor) used for normalization.
                                If provided, predictions and targets are rescaled by y_max.

    Returns:
        tuple: A tuple containing:
            - rmse (float): Root Mean Squared Error on normalized data,
            - mae (float): Mean Absolute Error on normalized data,
            - r2 (float): Coefficient of determination (R^2) on normalized data,
            - original_rmse (float): Root Mean Squared Error on rescaled data (if y_max provided, else np.nan),
            - original_mae (float): Mean Absolute Error on rescaled data (if y_max provided, else np.nan).
    """

    # Detect if it's a Keras model by checking for an 'evaluate' method.
    is_keras_model = hasattr(model, "evaluate") and callable(model.evaluate)

    if is_keras_model:
        # Evaluate using Keras built-in methods.
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
        y_pred = model.predict(X_test, verbose=0).flatten()
    else:
        # Assume a scikit-learn-like model with a predict() method.
        y_pred = model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(test_loss)
    mae = test_mae
    r2 = r2_score(y_test, y_pred)
    
    if y_max is not None:
        y_test_rescaled = y_test * y_max
        y_pred_rescaled = y_pred * y_max
        mse_rescaled = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        original_rmse = np.sqrt(mse_rescaled)
        original_mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    else:
        original_rmse, original_mae = np.nan, np.nan
    
    return rmse, mae, r2, original_rmse, original_mae


def evaluate_classification_model(config: Config, model, X_test: np.ndarray, y_test: np.ndarray, labels: list = None, title: str = "Confusion Matrix"):
    """
    Evaluates a classification model on the test set.

    Args:
        config (Config): Configuration object with attributes for saving plots (e.g., model_task, eol_capacity, use_aachen).
        model: The classification model (Keras or scikit-learn-like).
            Must have `evaluate()` and `predict()` (Keras) or just `predict()` (scikit-learn).
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets (one-hot for Keras, integers for scikit-learn).
        labels (list, optional): List of class names for confusion matrix and report.
        title (str): Title used for the confusion matrix plot.

    Returns:
        tuple: (test_accuracy, f1_macro) where test_accuracy is the accuracy score and 
            f1_macro is the macro-average F1 score.
    """

    # Detect if it's a Keras model by checking for an 'evaluate' method.
    is_keras_model = hasattr(model, "evaluate") and callable(model.evaluate)

    if is_keras_model:
        # Evaluate using Keras methods.
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
    else:
        y_pred = model.predict(X_test)
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_test_int = np.argmax(y_test, axis=1)
        else:
            y_test_int = y_test
        test_accuracy = accuracy_score(y_test_int, y_pred)

    # Compute macro-average F1 score: unweighted mean of the F1 scores per class.
    f1_macro = f1_score(y_test_int, y_pred, average='macro')

    if labels is not None:
        plot_confusion_matrix(config, y_test_int, y_pred, labels, title)
        print(classification_report(y_test_int, y_pred, target_names=labels))
    else:
        logger.warning("No labels provided; skipping confusion matrix and classification report.")

    return test_accuracy, f1_macro

def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, y_max: float = None, title: str = "True vs. Predicted Values"):
    """
    Plots a 'True vs. Predicted' scatter plot for regression outputs.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Model-predicted target values.
        y_max (float, optional): If normalized, rescale by multiplying with y_max.
        title (str): Title for the scatter plot (default: "True vs. Predicted Values").

    Returns:
        None
    """
    if y_max is not None:
        y_true = y_true * y_max
        y_pred = y_pred * y_max

    plt.figure()
    plt.scatter(y_true, y_pred)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(title)
    plt.show()

def plot_confusion_matrix(config: Config, y_true: np.ndarray, y_pred: np.ndarray, labels: list, title: str = "Confusion Matrix"):
    """
    Plots a confusion matrix for classification outputs and saves the plot using the provided configuration.

    Args:
        config (Config): Configuration object with attributes including model_task, eol_capacity, and use_aachen.
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        labels (list): List of class names.
        title (str): Title for the plot (default: "Confusion Matrix").

    Returns:
        None
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    
    # Determine subfolder based on config.use_aachen
    bottom_map_dir = "aachen" if config.use_aachen else "mit_stanford"
    output_dir = os.path.join("experiments", "results", bottom_map_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build filename using config.model_task and config.eol_capacity along with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.model_task}_confusion_eol{int(config.eol_capacity * 100)}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()