import numpy as np
import scipy.linalg
from scipy.linalg.lapack import dposv, dsysv
try:
    from solver_fast import construct_Q
except ImportError:
    construct_Q = None

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n_in, edges = problem
        M = n_in
        
        f1, f2 = edges
        w1 = np.pi * f1
        w2 = np.pi * f2
        
        # Indices
        m = np.arange(1, 2 * M + 1, dtype=np.float64)
        
        # Compute sines
        sin_m_w1 = np.sin(m * w1)
        sin_m_w2 = np.sin(m * w2)
        
        # Compute S
        # S[m] = (sin(m*w1) - sin(m*w2)) / m
        # Handle small m? No, m >= 1.
        S_rest = (sin_m_w1 - sin_m_w2) / m
        
        S = np.empty(2 * M + 1, dtype=np.float64)
        S[0] = w1 + (np.pi - w2)
        S[1:] = S_rest
        
        # Construct Q
        if construct_Q is not None:
            Q = construct_Q(S, M)
        else:
            k = np.arange(M + 1, dtype=np.int32)
            Q = 0.5 * (S[np.abs(k[:, None] - k)] + S[k[:, None] + k])
        
        # Construct b
        b = np.empty(M + 1, dtype=np.float64)
        b[0] = w1
        b[1:] = sin_m_w1[:M] / m[:M]
        
        # Solve Q a = b
        # Try Cholesky (fastest)
        try:
            c, a, info = dposv(Q, b, lower=False)
            if info != 0:
                raise ValueError("Cholesky failed")
        except Exception:
            # Try Symmetric Indefinite
            try:
                _, _, a, info = dsysv(Q, b, lower=False)
                if info != 0:
                    raise ValueError("Sysv failed")
            except Exception:
                # Fallback to least squares (most robust)
                a, _, _, _ = scipy.linalg.lstsq(Q, b, lapack_driver='gelsy')

        # Reconstruct h
        h = np.empty(2 * M + 1, dtype=np.float64)
        h[M] = a[0]
        halved_a = a[1:] * 0.5
        h[M+1:] = halved_a
        h[:M] = halved_a[::-1]
        
        return h