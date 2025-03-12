import numpy as np
from sklearn.metrics import pairwise_distances
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def perturbation_function(x, noise_scale=0.1):
    return x + np.random.normal(0, noise_scale, size=x.shape)

def calculate_stability_ratio(values_orig, values_perturbed, input_orig, input_perturbed):
    input_diffs = pairwise_distances(input_perturbed, input_orig, metric="euclidean").flatten()
    n_perturbations = values_perturbed.shape[0]
    value_diffs = np.zeros(n_perturbations)
    for i in range(n_perturbations):
        value_diffs[i] = np.linalg.norm(values_orig - values_perturbed[i])
    
    print("Input diffs:", input_diffs[:5])
    print("Value diffs:", value_diffs[:5])
    input_diffs[input_diffs == 0] = 1e-10
    ratios = value_diffs / input_diffs
    print("Ratios:", ratios[:5])
    return ratios

def get_representation_model(model, layer_name=None):
    """
    Create a model that outputs the hidden states from a specified layer using the original model's structure.
    
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
    
    # Verify layer exists
    if layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {[layer.name for layer in model.layers]}")
    
    # Get the layer's output directly from the original model
    layer_output = model.get_layer(layer_name).output
    
    # Create a new model from the original input to the desired layer's output
    return Model(inputs=model.input, outputs=layer_output)

def calculate_relative_input_stability(model, X_background, test, 
                                      n_perturbations=20, noise_scale=0.1, nsamples=1000, runs=1):
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

def calculate_relative_output_stability(model, test, 
                                       n_perturbations=20, noise_scale=0.1, runs=1):
    test_3d = test.reshape(-1, test.shape[1], 1)
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