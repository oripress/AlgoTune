from typing import Any
import numpy as np
from scipy.linalg.lapack import ztrsyl, zgees

_select_func = lambda x: False

class Solver:
    def __init__(self):
        self._lwork_cache = {}
    
    def _get_optimal_lwork(self, n):
        if n not in self._lwork_cache:
            dummy = np.empty((n, n), dtype=np.complex128, order='F')
            # lwork query
            res = zgees(_select_func, dummy, compute_v=1, sort_t=0, lwork=-1, overwrite_a=1)
            # Find the work array
            for i, r in enumerate(res):
                if isinstance(r, np.ndarray) and r.dtype == np.complex128 and r.ndim == 1 and len(r) == 1:
                    self._lwork_cache[n] = int(r[0].real)
                    break
            else:
                self._lwork_cache[n] = max(1, 2*n)
        return self._lwork_cache[n]
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"], dtype=np.complex128, order='F')
        B = np.array(problem["B"], dtype=np.complex128, order='F')
        Q = np.asarray(problem["Q"], dtype=np.complex128)
        
        n = A.shape[0]
        m = B.shape[0]
        
        lwork_a = self._get_optimal_lwork(n)
        lwork_b = self._get_optimal_lwork(m)
        
        # Direct LAPACK zgees call
        res_A = zgees(_select_func, A, compute_v=1, sort_t=0,
                      lwork=lwork_a, overwrite_a=1)
        T, U = res_A[0], res_A[3]
        
        res_B = zgees(_select_func, B, compute_v=1, sort_t=0,
                      lwork=lwork_b, overwrite_a=1)
        S, V = res_B[0], res_B[3]
        
        # F = U^H @ Q @ V
        F = U.conj().T @ (Q @ V)
        
        # Solve triangular Sylvester
        Y, scale, info = ztrsyl('N', 'N', T, S, F)
        if scale != 1.0:
            Y /= scale
        
        # X = U @ Y @ V^H
        X = U @ (Y @ V.conj().T)
        
        return {"X": X}