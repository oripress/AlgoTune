import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit
    def _svt_numba(mat, tau):
        rows, cols = mat.shape

        if rows <= cols:
            gram = mat @ mat.T
            w, u = np.linalg.eigh(gram)
            res = np.zeros((rows, cols), dtype=np.float64)
            tmp = u.T @ mat
            tau2 = tau * tau

            for i in range(w.shape[0]):
                wi = w[i]
                if wi > tau2:
                    s = np.sqrt(wi)
                    coef = (s - tau) / s
                    for r in range(rows):
                        ur = u[r, i] * coef
                        for c in range(cols):
                            res[r, c] += ur * tmp[i, c]
            return res

        gram = mat.T @ mat
        w, v = np.linalg.eigh(gram)
        res = np.zeros((rows, cols), dtype=np.float64)
        tmp = mat @ v
        tau2 = tau * tau

        for i in range(w.shape[0]):
            wi = w[i]
            if wi > tau2:
                s = np.sqrt(wi)
                coef = (s - tau) / s
                for r in range(rows):
                    tr = tmp[r, i] * coef
                    for c in range(cols):
                        res[r, c] += tr * v[c, i]
        return res

    @njit
    def _solve_unfolding_numba(m, obs_idx):
        rows, cols = m.shape
        total = rows * cols
        obs_count = obs_idx.size

        if obs_count == total:
            return m.copy()
        if obs_count == 0:
            return np.zeros((rows, cols), dtype=np.float64)
        if rows == 1 or cols == 1:
            return m.copy()

        x = m.copy()
        z = m.copy()
        udual = np.zeros((rows, cols), dtype=np.float64)

        rho = 1.6
        tau = 1.0 / rho
        norm_m = np.linalg.norm(m)
        abs_tol = 1e-5 * (1.0 + norm_m)
        rel_tol = 2e-4

        flat_m = m.ravel()
        z_prev = z.copy()

        for it in range(32):
            x = z - udual
            flat_x = x.ravel()
            for j in range(obs_count):
                idx = obs_idx[j]
                flat_x[idx] = flat_m[idx]

            z = _svt_numba(x + udual, tau)
            udual += x - z

            if it >= 8 and (it & 3) == 3:
                r_norm = np.linalg.norm(x - z)
                s_norm = rho * np.linalg.norm(z - z_prev)
                x_norm = np.linalg.norm(x)
                z_norm = np.linalg.norm(z)
                eps_pri = abs_tol + rel_tol * max(x_norm, z_norm)
                eps_dual = abs_tol + rel_tol * rho * np.linalg.norm(udual)
                if r_norm <= eps_pri and s_norm <= eps_dual:
                    break
                z_prev = z.copy()

        x = z - udual
        flat_x = x.ravel()
        for j in range(obs_count):
            idx = obs_idx[j]
            flat_x[idx] = flat_m[idx]
        return x

class Solver:
    def __init__(self):
        self._numba_ready = False
        if HAVE_NUMBA:
            try:
                dummy_m = np.zeros((2, 2), dtype=np.float64)
                dummy_idx = np.array([0], dtype=np.int64)
                _solve_unfolding_numba(dummy_m, dummy_idx)
                self._numba_ready = True
            except Exception:
                self._numba_ready = False

    @staticmethod
    def _svt(mat, tau):
        m, n = mat.shape
        if m <= n:
            gram = mat @ mat.T
            w, u = np.linalg.eigh(gram)
            s = np.sqrt(np.clip(w, 0.0, None))
            keep = s > tau
            if not np.any(keep):
                return np.zeros_like(mat)
            u = u[:, keep]
            s = s[keep]
            coef = (s - tau) / s
            return (u * coef) @ (u.T @ mat)

        gram = mat.T @ mat
        w, v = np.linalg.eigh(gram)
        s = np.sqrt(np.clip(w, 0.0, None))
        keep = s > tau
        if not np.any(keep):
            return np.zeros_like(mat)
        v = v[:, keep]
        s = s[keep]
        coef = (s - tau) / s
        return (mat @ (v * coef)) @ v.T

    def _solve_unfolding(self, m, mask):
        if mask.all():
            return m.copy()
        if not mask.any():
            return np.zeros_like(m)
        n_rows, n_cols = m.shape
        if n_rows == 1 or n_cols == 1:
            return m.copy()

        x = m.copy()
        z = x.copy()
        udual = np.zeros_like(m)

        rho = 1.6
        tau = 1.0 / rho

        norm_m = np.linalg.norm(m)
        abs_tol = 1e-5 * (1.0 + norm_m)
        rel_tol = 2e-4

        z_prev = z
        for it in range(40):
            x = z - udual
            x[mask] = m[mask]

            z = self._svt(x + udual, tau)
            udual += x - z

            if it >= 8 and (it & 3) == 3:
                r_norm = np.linalg.norm(x - z)
                s_norm = rho * np.linalg.norm(z - z_prev)
                x_norm = np.linalg.norm(x)
                z_norm = np.linalg.norm(z)
                eps_pri = abs_tol + rel_tol * max(x_norm, z_norm)
                eps_dual = abs_tol + rel_tol * rho * np.linalg.norm(udual)
                if r_norm <= eps_pri and s_norm <= eps_dual:
                    break
                z_prev = z.copy()

        x = z - udual
        x[mask] = m[mask]
        return x

    def solve(self, problem, **kwargs):
        observed_tensor = np.asarray(problem["tensor"], dtype=np.float64)
        mask = np.asarray(problem["mask"], dtype=bool)

        d1, d2, d3 = observed_tensor.shape
        unfolding = observed_tensor.reshape(d1, d2 * d3)

        if self._numba_ready:
            obs_idx = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
            completed = _solve_unfolding_numba(unfolding, obs_idx).reshape(d1, d2, d3)
        else:
            mask1 = mask.reshape(d1, d2 * d3)
            completed = self._solve_unfolding(unfolding, mask1).reshape(d1, d2, d3)

        return {"completed_tensor": completed}