import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        try:
            np.divide(M, -reg, out=M)
            K = np.exp(M, out=M)
            
            u = np.empty_like(a)
            u.fill(1.0 / a.shape[0])
            v = np.empty_like(b)
            v.fill(1.0 / b.shape[0])
            
            KtransposeU = np.empty_like(b)
            Kv = np.empty_like(a)
            valid_u = u.copy()
            valid_v = v.copy()
            
            for ii in range(100):
                np.dot(u, K, out=KtransposeU)
                np.divide(b, KtransposeU, out=v)
                np.dot(K, v, out=Kv)
                np.divide(a, Kv, out=u)
                
                if not (np.isfinite(u).all() and np.isfinite(v).all() and np.all(KtransposeU != 0)):
                    u = valid_u
                    v = valid_v
                    break
                np.copyto(valid_u, u)
                np.copyto(valid_v, v)
                
                tmp2 = KtransposeU * v
                tmp2 -= b
                err = np.sum(np.square(tmp2))
                if err < 1e-18:
                    break
                    
                for _ in range(9):
                    np.dot(u, K, out=KtransposeU)
                    np.divide(b, KtransposeU, out=v)
                    np.dot(K, v, out=Kv)
                    np.divide(a, Kv, out=u)
                        
            G = u.reshape(-1, 1) * K * v.reshape(1, -1)
            
            if not np.isfinite(G).all():
                raise ValueError("Non‑finite values in transport plan")
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}