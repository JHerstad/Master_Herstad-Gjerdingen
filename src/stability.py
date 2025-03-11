# shap_stability.py
import numpy as np
from sklearn.metrics import pairwise_distances
import shap
from tensorflow.keras.models import Model

def perturbation_function(x, noise_scale=0.1):
    """
    Perturb the input x with Gaussian noise across all timesteps.
    
    Parameters:
    - x: Input array of shape (1, timesteps).
    - noise_scale: Standard deviation of Gaussian noise (default: 0.1).
    
    Returns:
    - Perturbed input array with the same shape as x.
    """
    return x + np.random.normal(0, noise_scale, size=x.shape)

def calculate_stability_ratio(values_orig, values_perturbed, input_orig, input_perturbed):
    """
    Calculate the stability ratio between two sets of values relative to input differences.
    
    Parameters:
    - values_orig: Original values (e.g., SHAP, representations, or outputs).
    - values_perturbed: Perturbed values (array of shape (n_perturbations, ...)).
    - input_orig: Original input array (shape: (1, timesteps)).
    - input_perturbed: Perturbed input array (shape: (n_perturbations, timesteps)).
    
    Returns:
    - ratios: Array of stability ratios (n_perturbations,).
    """
    input_diffs = pairwise_distances(input_perturbed, input_orig, metric="euclidean").flatten()
    n_perturbations = values_perturbed.shape[0]
    value_diffs = np.zeros(n_perturbations)
    for i in range(n_perturbations):
        value_diffs[i] = np.linalg.norm(values_orig - values_perturbed[i])
    
    input_diffs[input_diffs == 0] = 1e-10
    return value_diffs / input_diffs

def get_representation_model(model, layer_name=None):
    """
    Create a model that outputs the hidden states from a specified layer.
    
    Parameters:
    - model: Trained Keras model (e.g., LSTM).
    - layer_name: Name of the layer to extract representations from (default: last LSTM layer).
    
    Returns:
    - A Keras Model object that outputs representations.
    """
    if layer_name is None:
        for layer in model.layers[::-1]:
            if "lstm" in layer.name.lower():
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("No LSTM layer found in the model.")
    
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def calculate_relative_input_stability(model, X_background, test, 
                                      n_perturbations=20, noise_scale=0.1, nsamples=100, runs=1):
    """
    Calculate Relative Input Stability (RIS) using SHAP explanations.
    
    Parameters:
    - model: Trained model expecting 3D input (n_samples, timesteps, 1).
    - X_background: 2D background dataset (n_samples, timesteps) for KernelExplainer.
    - test: 2D test instance (shape: (1, timesteps)).
    - n_perturbations: Number of perturbed inputs per run (default: 20).
    - noise_scale: Standard deviation of Gaussian noise (default: 0.1).
    - nsamples: Number of samples for KernelExplainer (default: 100).
    - runs: Number of perturbation runs (default: 1).
    
    Returns:
    - max_ratio: Maximum RIS across all runs.
    - mean_ratios: Mean RIS across all runs.
    """
    def predict_wrapper(X):
        return model.predict(X.reshape(-1, X.shape[1], 1), verbose=0)
    
    explainer = shap.KernelExplainer(predict_wrapper, X_background)
    shap_values_orig = explainer.shap_values(test, nsamples=nsamples, silent=True)
    if isinstance(shap_values_orig, list):
        shap_values_orig = shap_values_orig[0]
    
    all_ratios = []
    for _ in range(runs):
        X_perturbed = np.array([perturbation_function(test, noise_scale) for _ in range(n_perturbations)])
        X_perturbed_2d = X_perturbed.reshape(n_perturbations, -1)
        
        # Fix: Reshape each perturbed sample to (1, timesteps) before passing to SHAP
        shap_values_perturbed = np.array([
            explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True)[0] 
            if isinstance(explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True), list)
            else explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True)
            for i in range(n_perturbations)
        ])
        
        ratios = calculate_stability_ratio(shap_values_orig, shap_values_perturbed, test, X_perturbed_2d)
        all_ratios.extend(ratios)
    
    all_ratios = np.array(all_ratios)
    return np.max(all_ratios), np.mean(all_ratios)

def calculate_relative_representation_stability(model, test, 
                                               n_perturbations=20, noise_scale=0.1, runs=1, 
                                               layer_name=None):
    """
    Calculate Relative Representation Stability (RRS) using hidden states.
    
    Parameters:
    - model: Trained model expecting 3D input (n_samples, timesteps, 1).
    - test: 2D test instance (shape: (1, timesteps)).
    - n_perturbations: Number of perturbed inputs per run (default: 20).
    - noise_scale: Standard deviation of Gaussian noise (default: 0.1).
    - runs: Number of perturbation runs (default: 1).
    - layer_name: Name of the layer to extract representations from (default: last LSTM layer).
    
    Returns:
    - max_ratio: Maximum RRS across all runs.
    - mean_ratios: Mean RRS across all runs.
    """
    test_3d = test.reshape(-1, test.shape[1], 1)  # Shape: (1, timesteps, 1)
    
    repr_model = get_representation_model(model, layer_name)
    repr_orig = repr_model.predict(test_3d, verbose=0)
    
    all_ratios = []
    for _ in range(runs):
        X_perturbed = np.array([perturbation_function(test, noise_scale) for _ in range(n_perturbations)])
        X_perturbed_2d = X_perturbed.reshape(n_perturbations, -1)
        X_perturbed_3d = X_perturbed.reshape(n_perturbations, -1, 1)
        
        repr_perturbed = np.array([repr_model.predict(X_perturbed_3d[i:i+1], verbose=0) 
                                 for i in range(n_perturbations)])
        
        ratios = calculate_stability_ratio(repr_orig, repr_perturbed, test, X_perturbed_2d)
        all_ratios.extend(ratios)
    
    all_ratios = np.array(all_ratios)
    return np.max(all_ratios), np.mean(all_ratios)

def calculate_relative_output_stability(model, test, 
                                       n_perturbations=20, noise_scale=0.1, runs=1):
    """
    Calculate Relative Output Stability (ROS) using model predictions.
    
    Parameters:
    - model: Trained model expecting 3D input (n_samples, timesteps, 1).
    - test: 2D test instance (shape: (1, timesteps)).
    - n_perturbations: Number of perturbed inputs per run (default: 20).
    - noise_scale: Standard deviation of Gaussian noise (default: 0.1).
    - runs: Number of perturbation runs (default: 1).
    
    Returns:
    - max_ratio: Maximum ROS across all runs.
    - mean_ratios: Mean ROS across all runs.
    """
    test_3d = test.reshape(-1, test.shape[1], 1)  # Shape: (1, timesteps, 1)
    
    output_orig = model.predict(test_3d, verbose=0)
    
    all_ratios = []
    for _ in range(runs):
        X_perturbed = np.array([perturbation_function(test, noise_scale) for _ in range(n_perturbations)])
        X_perturbed_2d = X_perturbed.reshape(n_perturbations, -1)
        X_perturbed_3d = X_perturbed.reshape(n_perturbations, -1, 1)
        
        output_perturbed = np.array([model.predict(X_perturbed_3d[i:i+1], verbose=0) 
                                   for i in range(n_perturbations)])
        
        ratios = calculate_stability_ratio(output_orig, output_perturbed, test, X_perturbed_2d)
        all_ratios.extend(ratios)
    
    all_ratios = np.array(all_ratios)
    return np.max(all_ratios), np.mean(all_ratios)

if __name__ == "__main__":
    # Optional test code
    pass