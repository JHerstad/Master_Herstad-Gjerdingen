# Standard library imports
import os
import time

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
#import shap

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Import the dataset preprocessing function
from Load_and_Preprocess_Aachen import preprocess_aachen_dataset



# Configurable parameters
def get_config():
    """Return a dictionary of configurable parameters."""
    return {
        "learning_rate": 0.001,
        "clipnorm": 1.0,
        "lstm_units": 32,
        "dense_units": 16,
        "dropout_rate": 0.2,
        "epochs": 50,
        "batch_size": 32,
        "patience": 5  # For EarlyStopping
    }


def build_model(input_shape, config):
    """Build and compile the LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        LSTM(config["lstm_units"], activation='tanh', recurrent_activation='sigmoid', return_sequences=False, unroll=False),
        Dropout(config["dropout_rate"]),
        Dense(config["dense_units"], activation='tanh'),
        Dense(1)  # Output layer for regression
    ])
    optimizer = Adam(learning_rate=config["learning_rate"], clipnorm=config["clipnorm"])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def plot_training_history(history):
    """Plot the training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


def plot_predictions_vs_actual(y_test, y_pred):
    """Scatterplot comparing actual vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.title("Predicted vs Actual RUL80")
    plt.xlabel("Actual RUL80")
    plt.ylabel("Predicted RUL80")
    plt.legend()
    plt.grid()
    plt.show()


def plot_residuals(y_test, y_pred):
    """Plot residuals (difference between actual and predicted values)."""
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Residuals Histogram")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()





def explain_with_shap(model, X_train, X_test):
    """Explain the model predictions using SHAP."""
    # Create a SHAP explainer
    explainer = shap.DeepExplainer(model, X_train[:100])  # Use a representative subset of X_train
    
    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    # Summarize feature importance
    shap.summary_plot(shap_values[0], X_test, plot_type="bar")
    
    # Detailed feature contributions for a specific sample
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test[0])



def get_unique_model_name(base_name="model", directory="Aachen/Models"):
    """Generate a unique model name based on the current timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"{base_name}_{timestamp}")


def save_model_structure_and_weights(model, model_name):
    """Save the model structure and weights separately."""
    os.makedirs(os.path.dirname(model_name), exist_ok=True)

    # Save model structure
    structure_file = f"{model_name}.structure.json"
    with open(structure_file, "w") as json_file:
        json_file.write(model.to_json())
    print(f"Model structure saved to {structure_file}")

    # Save model weights (conform to the .weights.h5 requirement)
    weights_file = f"{model_name}.weights.h5"
    model.save_weights(weights_file)
    print(f"Model weights saved to {weights_file}")


def load_model_structure_and_weights(model_name, directory="Aachen/Models"):
    """Load a model structure and weights."""
    # Ensure model_name includes the directory
    if not os.path.dirname(model_name):  # If no directory in model_name
        model_name = os.path.join(directory, model_name)

    # Load model structure and weights paths
    structure_file = f"{model_name}.structure.json"
    weights_file = f"{model_name}.weights.h5"

    # Print the paths being checked
    print("Checking for model files in the following paths:")
    print(f"Structure file: {structure_file}")
    print(f"Weights file: {weights_file}")

    # Print the contents of the directory
    directory = os.path.dirname(structure_file)
    print(f"Contents of directory '{directory}':")
    print(os.listdir(directory) if os.path.exists(directory) else "Directory does not exist.")

    # Check if files exist
    if not os.path.exists(structure_file) or not os.path.exists(weights_file):
        raise FileNotFoundError("Model structure or weights file not found.")

    # Load model structure
    with open(structure_file, "r") as json_file:
        model = model_from_json(json_file.read())

    # Load model weights
    model.load_weights(weights_file)
    print(f"Model loaded from {structure_file} and {weights_file}")
    return model

def main():
    # Load configuration
    config = get_config()

    # Load and preprocess the Aachen dataset
    aachen_data = preprocess_aachen_dataset(
        "/Users/johannesherstad/Master_Herstad-Gjerdingen/Aachen/Degradation_Prediction_Dataset_ISEA.mat",
        test_cell_count=3,
        random_state=42,
        phase="Mid",
        log_transform=False,
    )

    # Extract the preprocessed data
    X_train_lstm = aachen_data["X_train"]
    X_val_lstm = aachen_data["X_val"]
    X_test_lstm = aachen_data["X_test"]
    y_train = aachen_data["y_train"]
    y_val = aachen_data["y_val"]
    y_test = aachen_data["y_test"]
    y_max = aachen_data["y_max"]

    # Define model name for loading
    model_name = None
    model_name = "Aachen/Models/model_20250127_135003"

    if model_name is None:
        model_name = get_unique_model_name()

    # Check if a trained model exists
    try:
        print("Attempting to load pre-trained model...")
        model = load_model_structure_and_weights(model_name)
        print("Pre-trained model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found. Training a new model...")

        # Build the model
        model = build_model(X_train_lstm.shape[1:], config)

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config["patience"], restore_best_weights=True)
        ]

        # Train the model
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            verbose=1,
            callbacks=callbacks
        )

        # Save the best model after training
        save_model_structure_and_weights(model, model_name)
        print(f"New model saved as {model_name}")

        # Plot training history
        plot_training_history(history)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test_lstm, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")

    # Make predictions on the test set
    y_pred = model.predict(X_test_lstm)

    # Rescale predictions and test data back to the original range
    y_pred_rescaled = y_pred.flatten() * y_max
    y_test_rescaled = y_test * y_max

    # Compare actual and predicted values
    results = pd.DataFrame({
        "Actual RUL80": y_test_rescaled,
        "Predicted RUL80": y_pred_rescaled
    })
    print(results.head())

    # Plot predictions vs actual
    plot_predictions_vs_actual(y_test_rescaled, y_pred_rescaled)

    # Plot residuals
    plot_residuals(y_test_rescaled, y_pred_rescaled)


    print(model.summary())


    # Disable eager execution for SHAP
    tf.compat.v1.disable_eager_execution()

    # Explain the model predictions using SHAP
    #explain_with_shap(model, X_train_lstm, X_test_lstm)




if __name__ == "__main__":
    main()
