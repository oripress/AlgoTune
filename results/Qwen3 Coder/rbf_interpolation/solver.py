import numpy as np
from typing import Any
from scipy.interpolate import RBFInterpolator
from scipy.linalg import solve

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator.
    
        This method creates an RBF interpolation function from the training data and
        evaluates it on the test points. It uses the RBF configuration parameters
        provided in the problem dictionary.
    
        :param problem: A dictionary representing the RBF interpolation problem,
                       which should include an "rbf_config" key with configuration parameters.
        :return: A dictionary with the solution containing:
                                 - "y_pred": Predicted function values at test points.
        """
        # Extract arrays directly without unnecessary conversions
        x_train = np.asarray(problem["x_train"])
        y_train = np.asarray(problem["y_train"]).ravel()
        x_test = np.asarray(problem["x_test"])
        
        # Get RBF configuration
        rbf_config = problem["rbf_config"]
        kernel = rbf_config["kernel"]
        epsilon = rbf_config.get("epsilon", 1.0)
        smoothing = rbf_config.get("smoothing", 0.0)
        
        # For simple cases without smoothing, use custom implementation
        if smoothing == 0.0 and kernel in ['thin_plate_spline', 'multiquadric', 'gaussian']:
            y_pred = self._custom_rbf_interpolate(x_train, y_train, x_test, kernel, epsilon)
        else:
            # Use scipy implementation for complex cases
            rbf_interpolator = RBFInterpolator(
                x_train, y_train, 
                kernel=kernel, 
                epsilon=epsilon,
                smoothing=smoothing
            )
            y_pred = rbf_interpolator(x_test)
        
        # Return the solution
        solution = {
            "y_pred": y_pred.tolist(),
        }
        
        return solution
    
    def _custom_rbf_interpolate(self, x_train, y_train, x_test, kernel, epsilon):
        # Compute pairwise distances
        distances = np.sqrt(((x_train[:, np.newaxis, :] - x_train[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Apply RBF kernel
        if kernel == 'thin_plate_spline':
            # Add small epsilon to avoid log(0)
            distances = np.where(distances == 0, 1e-10, distances)
            phi_matrix = distances ** 2 * np.log(distances)
        elif kernel == 'multiquadric':
            phi_matrix = np.sqrt(distances ** 2 + epsilon ** 2)
        elif kernel == 'gaussian':
            phi_matrix = np.exp(-distances ** 2 / (epsilon ** 2))
        else:
            # Fallback for other kernels
            phi_matrix = np.zeros((len(x_train), len(x_train)))
        
        # Solve for weights
        weights = solve(phi_matrix, y_train)
        
        # Compute test distances
        test_distances = np.sqrt(((x_test[:, np.newaxis, :] - x_train[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Apply RBF kernel to test distances
        if kernel == 'thin_plate_spline':
            test_distances = np.where(test_distances == 0, 1e-10, test_distances)
            test_phi = test_distances ** 2 * np.log(test_distances)
        elif kernel == 'multiquadric':
            test_phi = np.sqrt(test_distances ** 2 + epsilon ** 2)
        elif kernel == 'gaussian':
            test_phi = np.exp(-test_distances ** 2 / (epsilon ** 2))
        else:
            # Fallback for other kernels
            test_phi = np.zeros((len(x_test), len(x_train)))
        
        # Compute predictions
        y_pred = test_phi @ weights
        return y_pred