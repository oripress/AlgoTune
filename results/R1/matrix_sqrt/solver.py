import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        matrix_data = problem.get("matrix")
        
        # Handle empty input properly
        if matrix_data is None:
            return {"sqrtm": {"X": []}}
        if isinstance(matrix_data, list) and not matrix_data:
            return {"sqrtm": {"X": []}}
        if isinstance(matrix_data, np.ndarray) and matrix_data.size == 0:
            return {"sqrtm": {"X": []}}
        
        # Convert input to numpy array
        try:
            A = np.array(matrix_data, dtype=np.complex128)
        except:
            return {"sqrtm": {"X": []}}
        
        # Check matrix dimensions
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            return {"sqrtm": {"X": []}}
        
        n = A.shape[0]
        
        # Fast path for 1x1 matrices
        if n == 1:
            try:
                return {"sqrtm": {"X": [[np.sqrt(A[0,0])]]}}
            except:
                return {"sqrtm": {"X": []}}
        
        # Optimize identity matrices
        if np.array_equal(A, np.eye(n)):
            return {"sqrtm": {"X": np.eye(n).tolist()}}
        
        # Use explicit formula for 2x2 matrices
        if n == 2:
            try:
                a, b = A[0,0], A[0,1]
                c, d = A[1,0], A[1,1]
                trace = a + d
                det = a*d - b*c
                s = np.sqrt(det)
                t = np.sqrt(trace + 2*s)
                if abs(t) > 1e-12:
                    X = (A + s * np.eye(2)) / t
                    return {"sqrtm": {"X": X.tolist()}}
            except:
                pass
        
        # Use SciPy with optimized blocksize for small matrices
        try:
            if n <= 4:
                X, _ = scipy.linalg.sqrtm(A, disp=False, blocksize=1)
            else:
                X, _ = scipy.linalg.sqrtm(A, disp=False)
            return {"sqrtm": {"X": X.tolist()}}
        except:
            return {"sqrtm": {"X": []}}