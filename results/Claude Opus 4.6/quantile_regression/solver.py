import numpy as np
from scipy.sparse import csc_matrix

class Solver:
    def __init__(self):
        # Pre-import
        try:
            import highspy
            self._hs = highspy
            self._has_hs = True
        except ImportError:
            self._has_hs = False
            from scipy.optimize import linprog
            self._linprog = linprog
    
    def solve(self, problem, **kwargs):
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        q = problem["quantile"]
        fit_int = problem["fit_intercept"]
        
        n, p = X.shape
        k = p + int(fit_int)
        m = k + 2 * n
        
        if fit_int:
            Xa = np.empty((n, k))
            Xa[:, :p] = X
            Xa[:, p] = 1.0
        else:
            Xa = X
        
        if self._has_hs:
            return self._solve_hs(X, Xa, y, q, fit_int, n, p, k, m)
        else:
            return self._solve_scipy(X, Xa, y, q, fit_int, n, p, k, m)
    
    def _solve_hs(self, X, Xa, y, q, fit_int, n, p, k, m):
        hs = self._hs
        
        h = hs.Highs()
        h.silent()
        
        # Costs
        col_cost = np.zeros(m)
        col_cost[k:k+n] = q
        col_cost[k+n:] = 1.0 - q
        
        # Bounds
        inf = hs.kHighsInf
        col_lower = np.empty(m)
        col_lower[:k] = -inf
        col_lower[k:] = 0.0
        col_upper = np.full(m, inf)
        
        # Build CSC sparse matrix for the constraint [Xa | I | -I]
        nnz_xa = n * k
        nnz = nnz_xa + 2 * n
        
        indptr = np.empty(m + 1, dtype=np.int32)
        indptr[:k + 1] = np.arange(k + 1, dtype=np.int32) * n
        indptr[k + 1:k + n + 1] = nnz_xa + np.arange(1, n + 1, dtype=np.int32)
        indptr[k + n + 1:] = nnz_xa + n + np.arange(1, n + 1, dtype=np.int32)
        
        row_idx = np.empty(nnz, dtype=np.int32)
        ri = np.arange(n, dtype=np.int32)
        row_idx[:nnz_xa] = np.tile(ri, k)
        row_idx[nnz_xa:nnz_xa + n] = ri
        row_idx[nnz_xa + n:] = ri
        
        dat = np.empty(nnz)
        dat[:nnz_xa] = Xa.ravel(order='F')
        dat[nnz_xa:nnz_xa + n] = 1.0
        dat[nnz_xa + n:] = -1.0
        
        # Use passModel with HighsLp
        lp = hs.HighsLp()
        lp.num_col_ = int(m)
        lp.num_row_ = int(n)
        lp.col_cost_ = col_cost
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = y.copy()
        lp.row_upper_ = y.copy()
        
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = int(m)
        lp.a_matrix_.num_row_ = int(n)
        lp.a_matrix_.start_ = indptr
        lp.a_matrix_.index_ = row_idx
        lp.a_matrix_.value_ = dat
        
        h.passModel(lp)
        h.run()
        
        sol = h.getSolution()
        x_arr = np.array(sol.col_value)
        beta = x_arr[:k]
        
        if fit_int:
            coef = beta[:p]
            intercept = float(beta[p])
        else:
            coef = beta
            intercept = 0.0
        
        preds = (X @ coef + intercept).tolist()
        return {
            "coef": coef.tolist(),
            "intercept": [intercept],
            "predictions": preds
        }
    
    def _solve_scipy(self, X, Xa, y, q, fit_int, n, p, k, m):
        nnz_xa = n * k
        nnz = nnz_xa + 2 * n
        
        indptr = np.empty(m + 1, dtype=np.int32)
        indptr[:k + 1] = np.arange(k + 1, dtype=np.int32) * n
        indptr[k + 1:k + n + 1] = nnz_xa + np.arange(1, n + 1, dtype=np.int32)
        indptr[k + n + 1:] = nnz_xa + n + np.arange(1, n + 1, dtype=np.int32)
        
        indices = np.empty(nnz, dtype=np.int32)
        ri = np.arange(n, dtype=np.int32)
        indices[:nnz_xa] = np.tile(ri, k)
        indices[nnz_xa:nnz_xa + n] = ri
        indices[nnz_xa + n:] = ri
        
        dat = np.empty(nnz)
        dat[:nnz_xa] = Xa.ravel(order='F')
        dat[nnz_xa:nnz_xa + n] = 1.0
        dat[nnz_xa + n:] = -1.0
        
        A_eq = csc_matrix((dat, indices, indptr), shape=(n, m))
        
        c = np.zeros(m)
        c[k:k + n] = q
        c[k + n:] = 1.0 - q
        
        bds = np.empty((m, 2))
        bds[:k, 0] = -np.inf
        bds[:k, 1] = np.inf
        bds[k:, 0] = 0.0
        bds[k:, 1] = np.inf
        
        res = self._linprog(c, A_eq=A_eq, b_eq=y, bounds=bds, method='highs')
        
        beta = res.x[:k]
        if fit_int:
            coef = beta[:p]
            intercept = float(beta[p])
        else:
            coef = beta
            intercept = 0.0
        
        preds = (X @ coef + intercept).tolist()
        return {
            "coef": coef.tolist(),
            "intercept": [intercept],
            "predictions": preds
        }