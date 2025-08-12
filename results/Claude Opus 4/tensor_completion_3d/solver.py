import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solve the tensor completion problem using nuclear norm minimization.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with the completed tensor
        """
        # Extract problem data
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        
        # Matrix unfolding approach for tensor completion
        dim1, dim2, dim3 = tensor_dims
        
        # Optimize unfolding operations using numpy operations
        # Mode 1: (dim1) x (dim2*dim3)
        unfolding1 = observed_tensor.reshape(dim1, -1)
        mask1 = mask.reshape(dim1, -1)
        
        # Mode 2: (dim2) x (dim1*dim3) - use transpose and reshape
        unfolding2 = observed_tensor.transpose(1, 0, 2).reshape(dim2, -1)
        mask2 = mask.transpose(1, 0, 2).reshape(dim2, -1)
        
        # Mode 3: (dim3) x (dim1*dim2) - use transpose and reshape
        unfolding3 = observed_tensor.transpose(2, 0, 1).reshape(dim3, -1)
        mask3 = mask.transpose(2, 0, 1).reshape(dim3, -1)
        
        # Create variables for each unfolding
        X1 = cp.Variable((dim1, dim2 * dim3))
        X2 = cp.Variable((dim2, dim1 * dim3))
        X3 = cp.Variable((dim3, dim1 * dim2))
        
        # Objective: minimize sum of nuclear norms
        objective = cp.Minimize(cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc"))
        
        # Data fidelity constraints using indexing instead of multiplication
        # Extract indices where mask is True
        idx1 = np.where(mask1)
        idx2 = np.where(mask2)
        idx3 = np.where(mask3)
        
        constraints = [
            X1[idx1] == unfolding1[idx1],
            X2[idx2] == unfolding2[idx2],
            X3[idx3] == unfolding3[idx3],
        ]
        
        # Solve the problem with optimized solver settings
        prob = cp.Problem(objective, constraints)
        try:
            # Use SCS solver with optimized settings for speed
            prob.solve(solver=cp.SCS, eps=1e-3, max_iters=1000, verbose=False)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
                return {"completed_tensor": []}
            
            # Fold back the first unfolding to get the completed tensor
            completed_tensor = X1.value.reshape(tensor_dims)
            
            return {"completed_tensor": completed_tensor.tolist()}
            
        except (cp.SolverError, Exception):
            return {"completed_tensor": []}