import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Optimized tensor completion using efficient numpy operations.
        """
        # Extract problem data
        observed_tensor = np.array(problem["tensor"], dtype=np.float64)
        mask = np.array(problem["mask"], dtype=bool)
        tensor_dims = observed_tensor.shape
        
        dim1, dim2, dim3 = tensor_dims
        
        # Efficient unfolding using numpy transpose and reshape
        # Mode 1: (dim1) x (dim2*dim3)
        unfolding1 = observed_tensor.reshape(dim1, -1)
        mask1 = mask.reshape(dim1, -1)
        
        # Mode 2: (dim2) x (dim1*dim3) - transpose axes then reshape
        tensor_perm2 = np.transpose(observed_tensor, (1, 0, 2))
        mask_perm2 = np.transpose(mask, (1, 0, 2))
        unfolding2 = tensor_perm2.reshape(dim2, -1)
        mask2 = mask_perm2.reshape(dim2, -1)
        
        # Mode 3: (dim3) x (dim1*dim2) - transpose axes then reshape
        tensor_perm3 = np.transpose(observed_tensor, (2, 0, 1))
        mask_perm3 = np.transpose(mask, (2, 0, 1))
        unfolding3 = tensor_perm3.reshape(dim3, -1)
        mask3 = mask_perm3.reshape(dim3, -1)
        
        # Create variables for each unfolding
        X1 = cp.Variable((dim1, dim2 * dim3))
        X2 = cp.Variable((dim2, dim1 * dim3))
        X3 = cp.Variable((dim3, dim1 * dim2))
        
        # Objective: minimize sum of nuclear norms
        objective = cp.Minimize(cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc"))
        
        # Data fidelity constraints using efficient element-wise operations
        constraints = [
            X1[mask1] == unfolding1[mask1],
            X2[mask2] == unfolding2[mask2],
            X3[mask3] == unfolding3[mask3],
        ]
        
        # Solve with optimized settings
        prob = cp.Problem(objective, constraints)
        try:
            # Use SCS solver with optimized settings for speed
            prob.solve(solver=cp.SCS, verbose=False, 
                      eps=1e-3,  # Slightly relaxed tolerance for speed
                      max_iters=2000,
                      acceleration_lookback=10)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
                return {"completed_tensor": []}
            
            # Fold back the first unfolding to get the completed tensor
            completed_tensor = X1.value.reshape(tensor_dims)
            
            return {"completed_tensor": completed_tensor.tolist()}
            
        except (cp.SolverError, Exception):
            return {"completed_tensor": []}