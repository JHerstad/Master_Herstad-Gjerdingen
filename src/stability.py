import numpy as np
import shap


class StabilityMetrics:
    def __init__(self, model: Callable, background_data: np.ndarray, n_perturbations: int = 5, 
                 noise_scale: float = 0.01, p: int = 2, eps_min: float = 1e-3, 
                 mode: str = "regression"):
        self.model = model
        self.background_data = background_data
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.p = p
        self.eps_min = eps_min
        self.mode = mode  # "regression" or "classification"
        # Initialize LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.background_data,
            feature_names=[f"timestep_{i}" for i in range(20)], 
            mode=self.mode,
            discretize_continuous=False,
            class_names=[f"class_{i}" for i in range(7)] if mode == "classification" else None
        )
        
    def perturbation_function(self, x: np.ndarray) -> np.ndarray:
        return x + np.random.normal(0, self.noise_scale, size=x.shape)

    def predict_fn(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.reshape(-1, X.shape[1], 1), verbose=0)

    def compute_shap_values(self, x: np.ndarray, nsamples: int = 1000) -> np.ndarray:
        """Compute SHAP values with debug info."""
        explainer = shap.KernelExplainer(self.predict_fn, self.background_data)
        shap_values = explainer.shap_values(x, nsamples=nsamples, silent=True)
        shap_values = np.array(shap_values)
        return shap_values
    
    def compute_lime_values(self, x: np.ndarray, h_x: np.ndarray = None, num_features: int = 20) -> np.ndarray:
        """Compute LIME explanation and convert to (1, 20) array for predicted class."""
        x_flat = x.flatten()  # Shape (20,)
        if self.mode == "regression":
            explanation = self.lime_explainer.explain_instance(
                x_flat,
                self.predict_fn,
                num_features=num_features
            )
            lime_list = explanation.as_list()  # List of (feature_name, weight)
        else:  # Classification
            if h_x is None:
                h_x = self.predict_fn(x[np.newaxis, :])  # Predict for classification
            predicted_class = np.argmax(h_x, axis=1)[0]  # Get predicted class index
            explanation = self.lime_explainer.explain_instance(
                x_flat,
                self.predict_fn,
                num_features=num_features,
                labels=(predicted_class,)  # Explain only the predicted class
            )
            lime_list = explanation.as_list(label=predicted_class)  # Get explanation for predicted class

        lime_values = np.zeros((1, 20), dtype=np.float64)
        for feature_name, weight in lime_list:
            timestep_idx = int(feature_name.split('_')[1])  # Extract index from "timestep_X"
            lime_values[0, timestep_idx] = weight
        return lime_values
    
    def process_explanation_values(self, explain_values: np.ndarray, h_x: np.ndarray) -> np.ndarray:
        """Process explanation values for the predicted class if multi-output, handle 2D arrays."""
        if explain_values.ndim == 3 and explain_values.shape[2] > 1:  # Multi-class SHAP (1, 20, 7)
            y_x = np.argmax(h_x, axis=1)[0]
            return explain_values[:, :, y_x]  # Shape (1, 20)
        elif explain_values.ndim == 3:  # Regression SHAP (1, 20, 1)
            return np.squeeze(explain_values, axis=2)  # Shape (1, 20)
        else:  # LIME or already processed (1, 20)
            return explain_values  # Shape (1, 20)

    def compute_ris(self, x: np.ndarray, x_prime: np.ndarray, e_x: np.ndarray, 
                    e_x_prime: np.ndarray, h_x: np.ndarray, h_x_prime: np.ndarray) -> Optional[float]:
        x = np.array(x, dtype=np.float64)
        x_prime = np.array(x_prime, dtype=np.float64)
        h_x = np.array(h_x, dtype=np.float64)
        h_x_prime = np.array(h_x_prime, dtype=np.float64)

        if h_x.shape[1] > 1:  # Classification
            y_x = np.argmax(h_x)
            y_x_prime = np.argmax(h_x_prime)
            if y_x != y_x_prime:
                print(f"Class mismatch: y_x={y_x}, y_x_prime={y_x_prime}")
                return None
        
        e_x = self.process_explanation_values(e_x, h_x)
        e_x_prime = self.process_explanation_values(e_x_prime, h_x_prime)
        
        if e_x.shape != (1, 20) or e_x_prime.shape != (1, 20):
            raise ValueError(f"e_x shape {e_x.shape} or e_x_prime shape {e_x_prime.shape} must be (1, 20).")

        e_diff = e_x - e_x_prime
        normalization_factor = np.max(np.abs(e_x)) or self.eps_min
        relative_change = e_diff / normalization_factor
        e_diff_norm = np.linalg.norm(relative_change, ord=self.p)

        x_diff = (x - x_prime) / (x)
        x_diff_norm = np.linalg.norm(x_diff, ord=self.p)

        denominator = max(x_diff_norm, self.eps_min)
        print(f"RIS: e_diff_norm: {e_diff_norm:.6f}, x_diff_norm: {x_diff_norm:.6f}, RIS: {e_diff_norm / denominator:.6f}")
        return e_diff_norm / denominator

    def compute_ros(self, e_x: np.ndarray, e_x_prime: np.ndarray, h_x: np.ndarray, 
                    h_x_prime: np.ndarray) -> Optional[float]:
        e_x = np.array(e_x, dtype=np.float64)
        e_x_prime = np.array(e_x_prime, dtype=np.float64)
        h_x = np.array(h_x, dtype=np.float64)
        h_x_prime = np.array(h_x_prime, dtype=np.float64)

        if h_x.shape[1] > 1:  # Classification
            y_x = np.argmax(h_x)
            y_x_prime = np.argmax(h_x_prime)
            if y_x != y_x_prime:
                print(f"Class mismatch: y_x={y_x}, y_x_prime={y_x_prime}")
                return None
        
        e_x = self.process_explanation_values(e_x, h_x)
        e_x_prime = self.process_explanation_values(e_x_prime, h_x_prime)

        if e_x.shape != (1, 20) or e_x_prime.shape != (1, 20):
            raise ValueError(f"e_x shape {e_x.shape} or e_x_prime shape {e_x_prime.shape} must be (1, 20).")
        if h_x.shape != h_x_prime.shape:
            raise ValueError("h_x and h_x_prime must have the same shape.")

        e_diff = e_x - e_x_prime
        normalization_factor = np.max(np.abs(e_x)) or self.eps_min
        relative_change = e_diff / normalization_factor
        e_diff_norm = np.linalg.norm(relative_change, ord=self.p)

        h_diff = h_x - h_x_prime
        h_diff_norm = np.linalg.norm(h_diff, ord=self.p)
        denominator = max(h_diff_norm, self.eps_min)

        print(f"ROS: e_diff_norm: {e_diff_norm:.6f}, h_diff_norm: {h_diff_norm:.6f}, ROS: {e_diff_norm / denominator:.6f}")
        return e_diff_norm / denominator

    def calculate_stability(self, x: np.ndarray, metric: str = "ris", explainer_type: str = "shap") -> Tuple[float, float]:
        h_x = self.predict_fn(x)
        if explainer_type == "shap":
            e_x = self.compute_shap_values(x)
        elif explainer_type == "lime":
            e_x = self.compute_lime_values(x, h_x) 
        else:
            raise ValueError("explainer_type must be 'shap' or 'lime'")
        
        stability_values = []
        for _ in range(self.n_perturbations):
            x_prime = self.perturbation_function(x)
            h_x_prime = self.predict_fn(x_prime)
            if explainer_type == "shap":
                e_x_prime = self.compute_shap_values(x_prime)
            elif explainer_type == "lime":
                e_x_prime = self.compute_lime_values(x_prime, h_x_prime)
            
            if metric == "ris":
                stability = self.compute_ris(x, x_prime, e_x, e_x_prime, h_x, h_x_prime)
            elif metric == "ros":
                stability = self.compute_ros(e_x, e_x_prime, h_x, h_x_prime)
            else:
                raise ValueError("Metric must be 'ris' or 'ros'.")
            
            if stability is not None:
                stability_values.append(stability)
        
        return (np.max(stability_values), np.mean(stability_values)) if stability_values else (None, None)

def calculate_relative_input_stability(model: Callable, test_instance: np.ndarray, 
                                       background_data: np.ndarray, noise_scale: float = 0.1, 
                                       n_perturbations: int = 5, explainer_type: str = "shap",
                                       mode: str = "regression") -> Tuple[float, float]:
    stability = StabilityMetrics(model, background_data, n_perturbations, noise_scale, mode=mode)
    return stability.calculate_stability(test_instance, "ris", explainer_type)

def calculate_relative_output_stability(model: Callable, test_instance: np.ndarray, 
                                        background_data: np.ndarray, noise_scale: float = 0.1, 
                                        n_perturbations: int = 5, explainer_type: str = "shap",
                                        mode: str = "regression") -> Tuple[float, float]:
    stability = StabilityMetrics(model, background_data, n_perturbations, noise_scale, mode=mode)
    return stability.calculate_stability(test_instance, "ros", explainer_type)

if __name__ == "__main__":
    pass