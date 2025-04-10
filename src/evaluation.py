#!/usr/bin/env python3
"""
General module for evaluating regression models on a test set.
Supports both Keras-based models and generic scikit-learn-like models.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report, r2_score

# Configure logging for professional tracking
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
        (float, float, float): A tuple containing:
            - RMSE: Root Mean Squared Error,
            - MAE: Mean Absolute Error (after optional rescaling),
            - R^2: Coefficient of determination.
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

    normal_rmse = np.sqrt(test_loss)
    normal_mae = test_mae
    normal_r2 = r2_score(y_test, y_pred)
    
    if y_max is not None:
        y_test_rescaled = y_test * y_max
        y_pred_rescaled = y_pred * y_max
        mse_rescaled = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        rescaled_rmse = np.sqrt(mse_rescaled)
        rescaled_mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    else:
        rescaled_rmse, rescaled_mae = np.nan, np.nan
    
    return normal_rmse, normal_mae, normal_r2, rescaled_rmse, rescaled_mae


def evaluate_classification_model(model, X_test: np.ndarray, y_test: np.ndarray, labels: list = None, title: str = "Confusion Matrix"):
    """
    Evaluates a classification model on the test set.

    Args:
        model: The classification model (Keras or scikit-learn-like).
               Must have `evaluate()` and `predict()` (Keras) or just `predict()` (scikit-learn).
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets (one-hot for Keras, integers for scikit-learn).
        labels (list, optional): List of class names for confusion matrix and report.

    Returns:
        tuple: (test_loss, test_accuracy, y_pred) where test_loss is cross-entropy (Keras only),
               test_accuracy is the accuracy score, and y_pred are predicted class indices.
    """
    is_keras_model = hasattr(model, "evaluate") and callable(model.evaluate)

    if is_keras_model:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
    else:
        y_pred = model.predict(X_test)
        test_loss = None  # No loss for non-Keras models unless computed separately
        test_accuracy = accuracy_score(y_test, y_pred)
        y_test_int = y_test  # Assume integer labels for non-Keras

    if labels is not None:
        plot_confusion_matrix(y_test_int, y_pred, labels, title)
        print(classification_report(y_test_int, y_pred, target_names=labels))
    else:
        logger.warning("No labels provided; skipping confusion matrix and classification report.")

def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, y_max: float = None, title: str = "True vs. Predicted Values") -> None:
    """
    Plots a 'True vs. Predicted' scatter plot for regression outputs.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Model-predicted target values.
        y_max (float, optional): If normalized, rescale by multiplying with y_max.
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

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list, title: str = "Confusion Matrix") -> None:
    """
    Plots a confusion matrix for classification outputs.

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        labels (list): List of class names.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.show()