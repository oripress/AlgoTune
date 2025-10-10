from typing import Any
import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import ztrsyl
from scipy.linalg.blas import zgemm

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        
        # Compute Schur decompositions
        T_A, U_A = schur(A, output='complex')
        T_B, U_B = schur(B, output='complex')
        
        # Transform Q: F = U_A^H @ Q @ U_B using optimized BLAS
        # F = U_A^H @ Q @ U_B
        temp = zgemm(1.0, Q, U_B, trans_a=0, trans_b=0)
        F = zgemm(1.0, U_A, temp, trans_a=2, trans_b=0)  # trans_a=2 means conjugate transpose
        
        # Solve triangular Sylvester equation using LAPACK
        Y, scale, info = ztrsyl(T_A, T_B, F)
        
        # Scale if necessary
        if scale != 1.0:
            Y *= (1.0 / scale)
        
        # Back transform: X = U_A @ Y @ U_B^H using optimized BLAS
        temp = zgemm(1.0, Y, U_B, trans_a=0, trans_b=2)  # trans_b=2 means conjugate transpose
        X = zgemm(1.0, U_A, temp, trans_a=0, trans_b=0)
        
        return {"X": X}