#!/usr/bin/env python3
"""
Simple script to run an LSTM experiment for RUL regression on the Aachen dataset.

This script loads preprocessed data, trains an LSTM model, evaluates its performance,
and generates visualizations, ensuring reproducibility for thesis experiments.
"""

import os
import sys
from src.models import build_lstm_model, train_lstm_model, load_preprocessed_data
from src.evaluation import evaluate_lstm_model
from src.utils import plot_training_history, plot_predictions_vs_actual, plot_residuals
from config.defaults import Config

# Add thesis_experiment/ to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def main():
    """
    Runs a simple LSTM experiment for RUL regression, training, evaluating, and visualizing results.
    """
    config = Config()
    model_type = "regression"  # Fixed for LSTM regression

    # Load preprocessed data using the function from src/models.py
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_preprocessed_data(
        model_type, config.eol_capacity
    )

    # Build and train the LSTM model
    model = build_lstm_model((metadata["max_sequence_length"], 1), config)
    history = train_lstm_model(model, X_train, y_train, X_val, y_val, config)

    # Evaluate the model
    test_loss, test_mae = evaluate_lstm_model(model, X_test, y_test, metadata["y_max"])

    # Generate predictions and visualize results
    y_pred = model.predict(X_test, verbose=0).flatten()
    plot_training_history(history)
    plot_predictions_vs_actual(y_test, y_pred, metadata["y_max"])
    plot_residuals(y_test, y_pred, metadata["y_max"])

if __name__ == "__main__":
    main()