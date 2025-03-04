#!/usr/bin/env python3
"""
General module for evaluating regression models on a test set.
Supports both Keras-based models and generic scikit-learn-like models.
"""

import numpy as np
import logging

# Optional, if you want to compute metrics for non-Keras models:
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging for professional tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_regression_model(model, X_test: np.ndarray, y_test: np.ndarray, y_max: float = None):
    """
    Evaluates a regression model on the test set. Works for:
      - Keras models (i.e., tf.keras.Model) by calling model.evaluate().
      - Non-Keras models (e.g., scikit-learn) by manually computing MSE and MAE.

    Args:
        model: The regression model to evaluate.
               If it's a Keras model, must have `evaluate()` and `predict()` methods.
               Otherwise, must have a `predict()` method (scikit-learn style).
        X_test (np.ndarray): Test features, shape depends on the model.
        y_test (np.ndarray): Test targets (normalized or unnormalized).
        y_max (float, optional): Max RUL (or similar scaling factor) used for normalization.                 
                If provided, predictions and targets are rescaled by multiplying with y_max.

    Returns:
        (float, float): (test_loss, test_mae_rescaled) to mirror your LSTM approach,
                        where test_loss is MSE (or the Keras loss) and test_mae_rescaled
                        is MAE after optional rescaling.

    Notes:
        - For Keras models, we assume the loss is MSE and one metric is MAE.
        - For other models, we compute MSE and MAE manually via scikit-learn.
        - If y_max is provided, we rescale y_pred and y_test by y_max before computing final MAE.
        - If y_max is None, no rescaling is done.
    """

    # Detect if it's a Keras model by checking for an 'evaluate' method
    is_keras_model = hasattr(model, "evaluate") and callable(model.evaluate)

    if is_keras_model:
        # 1. Evaluate with Keras's built-in method
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
        # 2. Predict
        y_pred = model.predict(X_test, verbose=0).flatten()  # flatten in case it returns shape (n,1)

        # In your LSTM code, 'test_loss' should be MSE if compiled with `loss='mse'`
        # and 'test_mae' is the MAE metric. We'll keep those to stay consistent.

    else:
        # Assume a scikit-learn-like model with only `.predict()`
        y_pred = model.predict(X_test)

        # Compute test_loss as MSE
        test_loss = mean_squared_error(y_test, y_pred)
        # Compute test_mae as raw MAE
        test_mae = mean_absolute_error(y_test, y_pred)

    # Optionally rescale predictions and targets if y_max is given
    if y_max is not None:
        y_test_rescaled = y_test * y_max
        y_pred_rescaled = y_pred * y_max
        mae_rescaled = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
    else:
        # If no rescaling, the "rescaled" MAE is just the raw MAE
        mae_rescaled = test_mae

    # Log the results
    logger.info("Test Loss (MSE): %.4f", test_loss)
    logger.info("Test MAE (rescaled): %.4f", mae_rescaled)

    return test_loss, mae_rescaled





# src/evaluate_model.py
import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_pred(y_true, y_pred, y_max=None):
    """
    Plots a simple 'True vs. Predicted' scatter plot for regression outputs.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Model-predicted target values.
        y_max (float, optional): If the data was normalized by dividing by y_max,
                                 multiply values by y_max to rescale before plotting.
    """
    if y_max is not None:
        y_true = y_true * y_max
        y_pred = y_pred * y_max

    plt.figure()
    plt.scatter(y_true, y_pred)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("True vs. Predicted Values")
    plt.show()
