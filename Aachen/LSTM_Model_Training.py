# Import the required libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam

# Import the dataset preprocessing function
from Load_and_Preprocess_Aachen import preprocess_aachen_dataset

# Load and preprocess the Aachen dataset
aachen_data = preprocess_aachen_dataset("/Users/johannesherstad/Master_Herstad-Gjerdingen/Aachen/Degradation_Prediction_Dataset_ISEA.mat", test_cell_count=3, random_state=42)


# Extract the preprocessed data
X_train_lstm = aachen_data["X_train"]
X_val_lstm = aachen_data["X_val"]
X_test_lstm = aachen_data["X_test"]
y_train = aachen_data["y_train"]
y_val = aachen_data["y_val"]
y_test = aachen_data["y_test"]

# Define the model architecture
model = Sequential([
    # Masking layer to ignore padding values (0.0) in the input data
    Masking(mask_value=0.0, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    
    # LSTM layer with 32 units and ReLU activation function
    LSTM(32, activation='relu', return_sequences=False),  # Fewer units
    
    # Dropout layer to prevent overfitting by randomly setting 20% of input units to 0
    Dropout(0.2),
    
    # Dense layer with 16 units and ReLU activation function
    Dense(16, activation='relu'),  # Simpler Dense layer
    
    # Output layer with a single unit for regression (RUL80 prediction)
    Dense(1)  # Output layer for RUL80
])

# Define the optimizer with a lower learning rate and gradient clipping
optimizer = Adam(learning_rate=0.01, clipnorm=1.0)  # Lower learning rate with gradient clipping

# Compile the model with mean squared error loss and mean absolute error metric
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Display the model summary
print(model.summary())

# Train the model
history = model.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)


# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_lstm, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")


# Make predictions on the test set
y_pred = model.predict(X_test_lstm)

# Rescale predictions back to the original range
y_pred_rescaled = y_pred.flatten() * y_max
y_test_rescaled = y_test * y_max


import pandas as pd

# Create a DataFrame to compare actual and predicted RUL80
results = pd.DataFrame({
    "Actual RUL80": y_test_rescaled,
    "Predicted RUL80": y_pred_rescaled
})
print(results.head())
