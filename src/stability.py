import numpy as np
from sklearn.metrics import pairwise_distances
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def perturbation_function(x, noise_scale=0.1):
    return x + np.random.normal(0, noise_scale, size=x.shape)

def calculate_stability_ratio(x, x_prime, ex, ex_prime, p=2, epsilon_min=1e-8):
    # Compute the percent change in explanation (numerator)
    percent_change_ex = np.linalg.norm(x=((ex_prime - ex) / (ex + epsilon_min)), ord=p)

    # Compute the normalized input difference (denominator)
    input_difference = np.linalg.norm(x=((x_prime - x) / (x + epsilon_min)), ord=p)

    # Ensure denominator is not too small
    denominator = max(input_difference, epsilon_min)

    # Compute RIS
    ris = percent_change_ex / denominator

    return ris

def get_representation_model(model, layer_name=None):
    if layer_name is None:
        for layer in model.layers[::-1]:
            if "conv1d" in layer.name.lower() or "lstm" in layer.name.lower():
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("No Conv1D or LSTM layer found in the model.")
    
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {[layer.name for layer in model.layers]}")
    
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def calculate_relative_input_stability(model, X_background, test, 
                                      n_perturbations=20, noise_scale=0.1, nsamples=200, runs=1):
    def predict_wrapper(X):
        return model.predict(X.reshape(-1, X.shape[1], 1), verbose=0)  # Works for both CNN and LSTM
    
    explainer = shap.KernelExplainer(predict_wrapper, X_background)
    shap_values_orig = explainer.shap_values(test, nsamples=nsamples, silent=True)
    if isinstance(shap_values_orig, list):  # Multi-output case (e.g., classification)
        shap_values_orig = np.array(shap_values_orig).transpose(1, 2, 0)  # (1, n_features, n_outputs)
    else:
        shap_values_orig = shap_values_orig  # Single output (regression)
    
    all_ratios = []
    test_3d = test.reshape(-1, test.shape[1], 1)
    y_orig = predict_wrapper(test)  # Original predictions
    
    for _ in range(runs):
        X_perturbed = np.array([perturbation_function(test, noise_scale) for _ in range(n_perturbations)])
        X_perturbed_2d = X_perturbed.reshape(n_perturbations, -1)
        X_perturbed_3d = X_perturbed.reshape(n_perturbations, -1, 1)
        y_perturbed = predict_wrapper(X_perturbed_2d)  # Predictions for perturbed inputs
        
        # Label consistency check, ensuring 1D mask
        if y_orig.shape[-1] > 1:  # Classification
            mask = np.argmax(y_perturbed, axis=-1) == np.argmax(y_orig, axis=-1)
        else:  # Regression
            mask = np.abs(y_perturbed - y_orig).flatten() < 0.05  # Flatten to 1D
        
        if not np.any(mask):  # Skip if no perturbations maintain the same label
            continue
        
        # Compute SHAP values for valid perturbed inputs
        shap_values_perturbed = np.array([
            np.array(explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True)).transpose(1, 2, 0)
            if isinstance(explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True), list)
            else explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True)
            for i in range(n_perturbations)
        ])
        
        # Calculate stability ratios only for consistent predictions
        ratios = calculate_stability_ratio(shap_values_orig, shap_values_perturbed[mask], test, X_perturbed_2d[mask])
        all_ratios.extend(ratios)
    
    all_ratios = np.array(all_ratios)
    if len(all_ratios) == 0:  # Handle case where no valid perturbations exist
        return np.nan, np.nan
    return np.max(all_ratios), np.mean(all_ratios)

def calculate_relative_representation_stability(model, test, 
                                               n_perturbations=20, noise_scale=0.1, runs=1, 
                                               layer_name=None):
    test_3d = test.reshape(-1, test.shape[1], 1)
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

def calculate_relative_output_stability(model, X_background, test, n_perturbations=20, noise_scale=0.1, nsamples=200, runs=1):
    predict_wrapper = lambda X: model.predict(X.reshape(-1, X.shape[1], 1), verbose=0)
    explainer = shap.KernelExplainer(predict_wrapper, X_background)
    shap_values_orig = explainer.shap_values(test, nsamples=nsamples, silent=True)
    if isinstance(shap_values_orig, list):  # Multi-output case
        shap_values_orig = np.array(shap_values_orig).transpose(1, 2, 0)
    
    test_3d = test.reshape(-1, test.shape[1], 1)
    logits_orig = predict_wrapper(test)  # h(x)
    
    all_ratios = []
    for _ in range(runs):
        X_perturbed = np.array([perturbation_function(test, noise_scale) for _ in range(n_perturbations)])
        X_perturbed_2d = X_perturbed.reshape(n_perturbations, -1)
        logits_perturbed = predict_wrapper(X_perturbed_2d)  # h(x')
        
        # Label consistency check
        mask = np.argmax(logits_perturbed, axis=-1) == np.argmax(logits_orig, axis=-1) if logits_orig.shape[-1] > 1 else np.abs(logits_perturbed - logits_orig) < 0.05
        if not np.any(mask):
            continue
        
        # Compute SHAP values for perturbed inputs
        shap_values_perturbed = np.array([
            np.array(explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True)).transpose(1, 2, 0)
            if isinstance(explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True), list)
            else explainer.shap_values(X_perturbed[i:i+1].reshape(1, -1), nsamples=nsamples, silent=True)
            for i in range(n_perturbations)
        ])
        
        # Calculate ROS using modified stability ratio
        ratios = calculate_stability_ratio(shap_values_orig, shap_values_perturbed[mask], 
                                         logits_orig, logits_perturbed[mask])
        all_ratios.extend(ratios)
    
    all_ratios = np.array(all_ratios)
    return np.max(all_ratios) if len(all_ratios) > 0 else np.nan, np.mean(all_ratios) if len(all_ratios) > 0 else np.nan