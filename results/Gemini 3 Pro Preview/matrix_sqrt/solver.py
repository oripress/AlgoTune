import numpy as np
import scipy.linalg
import cmath
try:
    import utils
except ImportError:
    utils = None

class Solver:
    def solve(self, problem: dict, **kwargs):
        matrix_data = problem["matrix"]
        if isinstance(matrix_data, np.ndarray):
            A = matrix_data
        else:
            A = np.array(matrix_data, dtype=complex)

        n = A.shape[0]
        
        # 2x2 Optimization
        if n == 2:
            if utils:
                try:
                    if A.dtype != np.complex128:
                        A_c = A.astype(np.complex128)
                    else:
                        A_c = A
                    X = utils.sqrtm_2x2(A_c)
                    if X is not None:
                        return {"sqrtm": {"X": X.tolist()}}
                except:
                    pass
            else:
                try:
                    a = A[0, 0]
                    b = A[0, 1]
                    c = A[1, 0]
                    d = A[1, 1]
                    
                    detA = a*d - b*c
                    s = cmath.sqrt(detA)
                    trA = a + d
                    t = cmath.sqrt(trA + 2*s)
                    
                    if abs(t) > 1e-9:
                        inv_t = 1.0 / t
                        X = np.empty((2, 2), dtype=complex)
                        X[0, 0] = (a + s) * inv_t
                        X[0, 1] = b * inv_t
                        X[1, 0] = c * inv_t
                        X[1, 1] = (d + s) * inv_t
                        return {"sqrtm": {"X": X.tolist()}}
                except:
                    pass

        try:
            # Check if Hermitian
            is_herm = False
            if utils:
                if A.dtype != np.complex128:
                    A_c = A.astype(np.complex128)
                else:
                    A_c = A
                is_herm = utils.is_hermitian(A_c)
            else:
                is_herm = np.allclose(A, A.conj().T)

            if is_herm:
                evals, evecs = np.linalg.eigh(A)
                sqrt_evals = np.sqrt(evals.astype(complex))
                # X = V @ diag(sqrt_evals) @ V.H
                # Optimized: (V * sqrt_evals) @ V.H
                X = (evecs * sqrt_evals) @ evecs.conj().T
            else:
                evals, evecs = np.linalg.eig(A)
                sqrt_evals = np.sqrt(evals)
                # X = V @ diag(sqrt_evals) @ V^-1
                # B = V @ diag(sqrt_evals)
                B = evecs * sqrt_evals
                # X = B @ inv(V) -> X @ V = B -> V.T @ X.T = B.T
                X = np.linalg.solve(evecs.T, B.T).T
            
            return {"sqrtm": {"X": X.tolist()}}
        except Exception:
            try:
                X = scipy.linalg.sqrtm(A)
                return {"sqrtm": {"X": X.tolist()}}
            except:
                return {"sqrtm": {"X": []}}