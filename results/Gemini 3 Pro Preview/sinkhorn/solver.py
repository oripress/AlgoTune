import numpy as np
from scipy.linalg import blas
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.array(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])

        # Kernel matrix
        inv_reg = -1.0 / reg
        K = np.exp(M * inv_reg)
        
        # We use K.T (F-contiguous view of C-contiguous K) for BLAS calls
        KT = K.T
        
        # Get BLAS functions
        gemv = blas.get_blas_funcs("gemv", (KT,))
        nrm2 = blas.get_blas_funcs("nrm2", (a,))
        
        # Initialize vectors
        u = np.ones_like(a)
        v = np.ones_like(b)
        u_prev = np.empty_like(a)
        Kv = np.empty_like(a)
        KTu = np.empty_like(b)
        
        threshold = 1e-9
        max_iter = 1000
        
        for i in range(max_iter):
            # Check convergence every 10 iterations
            check = (i % 20 == 0)
            
            if check:
                np.copyto(u_prev, u)
            
            # Kv = K @ v = KT.T @ v
            # trans=1 means transpose of KT, so (KT).T = K
            gemv(alpha=1.0, a=KT, x=v, y=Kv, trans=1, overwrite_y=True)
            np.divide(a, Kv, out=u)
            
            # KTu = K.T @ u = KT @ u
            # trans=0 means no transpose of KT
            gemv(alpha=1.0, a=KT, x=u, y=KTu, trans=0, overwrite_y=True)
            np.divide(b, KTu, out=v)
            
            if check:
                # Check convergence
                # u_prev = u_prev - u
                u_prev -= u
                
                # err = |u_prev| / |u|
                err = nrm2(u_prev) / nrm2(u)
                
                if err < threshold:
                    break
                break
                
        G = u[:, None] * K * v[None, :]
        
        return {"transport_plan": G}