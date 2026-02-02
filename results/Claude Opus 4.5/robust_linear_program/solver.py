import numpy as np
import ecos
from scipy.sparse import csc_matrix

class Solver:
    def solve(self, problem, **kwargs):
        c = np.asarray(problem["c"], dtype=np.float64)
        b_vec = np.asarray(problem["b"], dtype=np.float64)
        P_list = problem["P"]
        q_list = problem["q"]
        
        m = len(P_list)
        n = len(c)
        
        if m == 0:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}
        
        # Pre-compute sizes and total rows in a single pass
        total_rows = 0
        cone_dims = []
        
        for i in range(m):
            Pi = P_list[i]
            if isinstance(Pi, list):
                if len(Pi) > 0 and isinstance(Pi[0], list):
                    k_i = len(Pi[0])
                else:
                    k_i = 1
            else:
                k_i = Pi.shape[1] if Pi.ndim > 1 else 1
            cone_dims.append(1 + k_i)
            total_rows += 1 + k_i
        
        # Pre-allocate G and h
        G = np.zeros((total_rows, n), dtype=np.float64)
        h = np.zeros(total_rows, dtype=np.float64)
        
        row_idx = 0
        for i in range(m):
            Pi = np.asarray(P_list[i], dtype=np.float64)
            qi = np.asarray(q_list[i], dtype=np.float64)
            if Pi.ndim == 1:
                Pi = Pi.reshape(-1, 1)
            k_i = Pi.shape[1]
            
            G[row_idx, :] = qi
            G[row_idx+1:row_idx+1+k_i, :] = -Pi.T
            h[row_idx] = b_vec[i]
            
            row_idx += 1 + k_i
        
        G_sparse = csc_matrix(G)
        
        try:
            solution = ecos.solve(c, G_sparse, h, {'l': 0, 'q': cone_dims}, 
                                  verbose=False, feastol=1e-8, abstol=1e-8, reltol=1e-8)
            
            if solution['info']['exitFlag'] in [0, 10]:
                return {"objective_value": solution['info']['pcost'], "x": solution['x']}
            else:
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}
        except Exception:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}