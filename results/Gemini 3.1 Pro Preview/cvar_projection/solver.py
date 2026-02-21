import osqp
import scipy.sparse as sparse
import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        self.cache = {}

    def solve(self, problem: dict, **kwargs) -> Any:
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 0.0))

        n, d = A.shape
        k = int((1 - beta) * n)
        alpha = kappa * k

        if k == 0:
            return {"x_proj": x0.tolist()}

        key = (n, d, k)
        if key not in self.cache:
            P = sparse.diags([2.0]*d + [0.0]*(1+n), format='csc')

            indptr = np.empty(d + 1 + n + 1, dtype=int)
            indptr[0 : d+1] = np.arange(0, d*n + 1, n)
            indptr[d+1] = d*n + 1 + n
            indptr[d+2 : d+2+n] = d*n + 1 + n + np.arange(3, 3*n + 1, 3)

            indices_x = np.tile(np.arange(1, n+1), d)
            indices_t = np.arange(0, n+1)
            indices_u = np.empty(3*n, dtype=int)
            indices_u[0::3] = 0
            indices_u[1::3] = np.arange(1, n+1)
            indices_u[2::3] = np.arange(n+1, 2*n+1)
            indices = np.concatenate([indices_x, indices_t, indices_u])

            data_t = np.empty(1+n)
            data_t[0] = k
            data_t[1:] = -1.0
            
            data_u = np.tile([1.0, -1.0, 1.0], n)

            data_x = A.flatten('F')
            data = np.concatenate([data_x, data_t, data_u])

            C = sparse.csc_matrix((data, indices, indptr), shape=(1+2*n, d+1+n))

            l = np.empty(1 + 2*n)
            l[0] = -np.inf
            l[1:n+1] = -np.inf
            l[n+1:2*n+1] = 0

            u_bound = np.empty(1 + 2*n)
            u_bound[0] = alpha
            u_bound[1:n+1] = 0
            u_bound[n+1:2*n+1] = np.inf

            q = np.zeros(d + 1 + n)
            q[:d] = -2.0 * x0

            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=C, l=l, u=u_bound, verbose=False, warm_start=True, 
                         eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
            
            self.cache[key] = (solver, C)
        else:
            solver, C = self.cache[key]
            
            q = np.zeros(d + 1 + n)
            q[:d] = -2.0 * x0
            
            C.data[:n*d] = A.flatten('F')
            
            u_bound = np.empty(1 + 2*n)
            u_bound[0] = alpha
            u_bound[1:n+1] = 0
            u_bound[n+1:2*n+1] = np.inf
            
            solver.update(q=q, Ax=C.data, u=u_bound)

        results = solver.solve()
        if results.info.status_val in (1, 2):
            x_proj = results.x[:d]
            return {"x_proj": x_proj.tolist()}
        else:
            # Fallback if OSQP fails
            return {"x_proj": []}