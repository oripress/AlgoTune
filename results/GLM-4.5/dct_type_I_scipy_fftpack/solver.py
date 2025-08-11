import numpy as np
import scipy.fftpack
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using optimized approach.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # For small arrays, use direct matrix multiplication
        if problem.size <= 64:
            def get_dct1_matrix(n):
                """Generate DCT Type I transformation matrix"""
                if n == 1:
                    return np.array([[1.0]])
                matrix = np.zeros((n, n))
                for k in range(n):
                    for i in range(n):
                        angle = np.pi * k * i / (n - 1)
                        matrix[k, i] = np.cos(angle)
                return matrix
            
            # Apply matrix multiplication along each dimension
            result = problem.copy()
            for axis in range(problem.ndim):
                n = problem.shape[axis]
                dct_matrix = get_dct1_matrix(n)
                
                # Use numpy.dot instead of tensordot for better performance
                if axis == 0:
                    result = np.dot(dct_matrix, result)
                elif axis == 1:
                    result = np.dot(result, dct_matrix.T)
                else:
                    # For higher dimensions, fall back to tensordot
                    result = np.tensordot(dct_matrix, result, axes=(1, axis))
                    result = np.moveaxis(result, 0, axis)
            
            return result
        else:
            # For larger arrays, use optimized approach
            if problem.ndim == 2:
                # For 2D arrays, use the fastest approach
                # Use scipy.fftpack.dct with overwrite_x=True
                result = scipy.fftpack.dct(problem, type=1, axis=1, overwrite_x=True)
                result = scipy.fftpack.dct(result, type=1, axis=0, overwrite_x=True)
                return result
            else:
                # For higher dimensions, use dctn with overwrite_x=True
                # But try to optimize by processing dimensions in order of increasing size
                # This can improve cache utilization
                axes_order = np.argsort(problem.shape)
                
                # Create a copy to avoid modifying the original
                result = problem.copy()
                
                # Process dimensions in order of increasing size
                for axis in axes_order:
                    result = scipy.fftpack.dct(result, type=1, axis=axis, overwrite_x=True)
                
                return result