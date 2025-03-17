import numpy as np
import shap
from typing import Callable, Tuple, Optional

class StabilityMetrics:
    def __init__(self, model: Callable, background_data: np.ndarray, n_perturbations: int = 5, 
                 noise_scale: float = 0.01, p: int = 2, eps_min: float = 1e-2, eps_perturb: float = 0.2):
        self.model = model
        self.background_data = background_data
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.p = p
        self.eps_min = eps_min
        self.eps_perturb = eps_perturb
        
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

    def get_predicted_class_shap(self, shap_values: np.ndarray, h_x: np.ndarray) -> np.ndarray:
        """Extract SHAP values for the predicted class if multi-output."""
        if shap_values.shape[2] > 1:  # Multi-class case (1, 120, 7)
            y_x = np.argmax(h_x, axis=1)[0]
            return shap_values[:, :, y_x]  # Shape (1, 120)
        else:  # Regression case (1, 120, 1)
            return np.squeeze(shap_values, axis=2)  # Shape (1, 120)

    def compute_ris(self, x: np.ndarray, x_prime: np.ndarray, e_x: np.ndarray, 
                    e_x_prime: np.ndarray, h_x: np.ndarray, h_x_prime: np.ndarray) -> Optional[float]:
        x = np.array(x, dtype=np.float64)
        x_prime = np.array(x_prime, dtype=np.float64)
        h_x = np.array(h_x, dtype=np.float64)
        h_x_prime = np.array(h_x_prime, dtype=np.float64)

        # Perturbation constraint (optional for regression)
        perturb_norm = np.linalg.norm(x - x_prime, ord=self.p)
        if h_x.shape[1] > 1:  # Classification
            if perturb_norm > self.eps_perturb:
                print(f"Perturbation too large: {perturb_norm:.6f} > {self.eps_perturb}")
                return None
            y_x = np.argmax(h_x)
            y_x_prime = np.argmax(h_x_prime)
            if y_x != y_x_prime:
                print(f"Class mismatch: y_x={y_x}, y_x_prime={y_x_prime}")
                return None
        
        e_x = self.get_predicted_class_shap(e_x, h_x)
        e_x_prime = self.get_predicted_class_shap(e_x_prime, h_x_prime)
        
        if e_x.shape != (1, 120) or e_x_prime.shape != (1, 120):
            raise ValueError(f"e_x shape {e_x.shape} or e_x_prime shape {e_x_prime.shape} must be (1, 120).")

        e_diff = e_x - e_x_prime
        normalization_factor = np.max(np.abs(e_x)) or self.eps_min
        relative_change = e_diff / normalization_factor
        e_diff_norm = np.linalg.norm(relative_change, ord=self.p)

        x_diff = (x - x_prime) / (x)  # Added eps_min to avoid division by zero
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
        
        e_x = self.get_predicted_class_shap(e_x, h_x)
        e_x_prime = self.get_predicted_class_shap(e_x_prime, h_x_prime)

        if e_x.shape != (1, 120) or e_x_prime.shape != (1, 120):
            raise ValueError(f"e_x shape {e_x.shape} or e_x_prime shape {e_x_prime.shape} must be (1, 120).")
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

    def calculate_stability(self, x: np.ndarray, metric: str = "ris") -> Tuple[float, float]:
        h_x = self.predict_fn(x)
        e_x = self.compute_shap_values(x)
        
        stability_values = []
        for _ in range(self.n_perturbations):
            x_prime = self.perturbation_function(x)
            h_x_prime = self.predict_fn(x_prime)
            e_x_prime = self.compute_shap_values(x_prime)
            
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
                                       n_perturbations: int = 5) -> Tuple[float, float]:
    stability = StabilityMetrics(model, background_data, n_perturbations, noise_scale)
    return stability.calculate_stability(test_instance, "ris")

def calculate_relative_output_stability(model: Callable, test_instance: np.ndarray, 
                                       background_data: np.ndarray, noise_scale: float = 0.1, 
                                       n_perturbations: int = 5) -> Tuple[float, float]:
    stability = StabilityMetrics(model, background_data, n_perturbations, noise_scale)
    return stability.calculate_stability(test_instance, "ros")

if __name__ == "__main__":
    pass