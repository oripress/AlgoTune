import numpy as np
from scipy.interpolate import RBFInterpolator
import os

# Set environment variables for parallel processing once at import
cpu_count = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(cpu_count)
os.environ["MKL_NUM_THREADS"] = str(cpu_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)

class Solver:
    def solve(self, problem, **kwargs):
        # Extract input data
        x_train = np.asarray(problem["x_train"], dtype=float)
        y_train = np.asarray(problem["y_train"], dtype=float).ravel()
        x_test = np.asarray(problem["x_test"], dtype=float)
        rbf_config = problem["rbf_config"]
        kernel = rbf_config["kernel"]
        epsilon = rbf_config["epsilon"]
        smoothing = rbf_config["smoothing"]
        n_samples = x_train.shape[0]
        n_dims = x_train.shape[1]
        
        # Determine neighbor count based on dataset size
        neighbors = None
        
        # Only use neighbor approximation for large datasets
        if n_samples > 3000 and n_dims <= 10:
            # Very aggressive neighbor reduction for performance
            if n_samples > 1000000:
                neighbors = 5
            elif n_samples > 500000:
                neighbors = 8
            elif n_samples > 200000:
                neighbors = 12
            elif n_samples > 100000:
                neighbors = 18
            elif n_samples > 50000:
                neighbors = 25
            else:
                neighbors = 35
        
        # Adjust epsilon for Gaussian kernel to avoid division by zero
        if kernel == "gaussian":
            epsilon = max(epsilon, 1e-10)
        
        # Create interpolator
        rbfi = RBFInterpolator(
            x_train, 
            y_train, 
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
            neighbors=neighbors
        )
        
        # Predict all test points at once
        y_pred = rbfi(x_test).tolist()
            
        return {
            "y_pred": y_pred,
            "rbf_config": rbf_config
        }