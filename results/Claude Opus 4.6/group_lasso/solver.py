import numpy as np
from scipy.special import expit
import numba as nb

@nb.njit(cache=True)
def _prox_group_lasso(u, group_starts, group_sizes, thresholds, out):
    """Group soft-thresholding for contiguous groups."""
    m = group_starts.shape[0]
    for j in range(m):
        s = group_starts[j]
        sz = group_sizes[j]
        norm_sq = 0.0
        for k in range(s, s + sz):
            norm_sq += u[k] * u[k]
        norm = np.sqrt(norm_sq)
        if norm > thresholds[j]:
            scale = 1.0 - thresholds[j] / norm
            for k in range(s, s + sz):
                out[k] = u[k] * scale
        else:
            for k in range(s, s + sz):
                out[k] = 0.0

@nb.njit(cache=True)
def _group_penalty(beta, group_starts, group_sizes, weights):
    """Compute weighted group lasso penalty."""
    m = group_starts.shape[0]
    penalty = 0.0
    for j in range(m):
        s = group_starts[j]
        sz = group_sizes[j]
        norm_sq = 0.0
        for k in range(s, s + sz):
            norm_sq += beta[k] * beta[k]
        penalty += weights[j] * np.sqrt(norm_sq)
    return penalty


class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        gl = np.array(problem["gl"])
        lba = float(problem["lba"])
        
        n, p1 = X.shape
        p = p1 - 1
        
        # Group structure
        unique_labels, inverse_indices = np.unique(gl, return_inverse=True)
        m = len(unique_labels)
        
        # Sort features by group for contiguous blocks
        perm = np.argsort(inverse_indices, kind='stable')
        inv_perm = np.empty(p, dtype=np.int64)
        inv_perm[perm] = np.arange(p)
        
        X_feat = np.ascontiguousarray(X[:, 1:][:, perm])
        
        # Group structure (contiguous)
        _, gsizes = np.unique(inverse_indices, return_counts=True)
        group_sizes = gsizes.astype(np.int64)
        group_starts = np.zeros(m, dtype=np.int64)
        if m > 1:
            np.cumsum(group_sizes[:-1], out=group_starts[1:])
        group_weights = np.sqrt(group_sizes.astype(np.float64))
        
        # Lipschitz constant of gradient of logistic loss
        if n <= p1:
            eigmax = np.linalg.eigvalsh(X @ X.T)[-1]
        else:
            eigmax = np.linalg.eigvalsh(X.T @ X)[-1]
        L = max(eigmax / 4.0, 1e-10)
        step = 1.0 / L
        thresholds = lba * group_weights * step
        
        # FISTA
        beta = np.zeros(p)
        beta0 = 0.0
        z_beta = np.zeros(p)
        z_beta0 = 0.0
        t = 1.0
        beta_old = np.zeros(p)
        
        for it in range(10000):
            beta_old[:] = beta
            beta0_old = beta0
            
            # Gradient at extrapolated point z
            eta = X_feat.dot(z_beta)
            eta += z_beta0
            sig = expit(eta)
            r = sig - y
            
            grad_b = X_feat.T.dot(r)
            grad_b0 = r.sum()
            
            # Gradient step
            u = z_beta - step * grad_b
            u0 = z_beta0 - step * grad_b0
            
            # Proximal step (group soft-thresholding)
            _prox_group_lasso(u, group_starts, group_sizes, thresholds, beta)
            beta0 = u0
            
            # FISTA momentum
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            mom = (t - 1.0) / t_new
            z_beta = beta + mom * (beta - beta_old)
            z_beta0 = beta0 + mom * (beta0 - beta0_old)
            t = t_new
            
            # Convergence check
            if it % 10 == 9:
                if (np.max(np.abs(beta - beta_old)) < 1e-11 and
                    abs(beta0 - beta0_old) < 1e-11):
                    break
        
        # Compute objective
        eta = X_feat.dot(beta) + beta0
        obj = -y.dot(eta) + np.logaddexp(0, eta).sum()
        obj += lba * _group_penalty(beta, group_starts, group_sizes, group_weights)
        
        # Un-permute beta back to original ordering
        beta_orig = beta[inv_perm]
        
        return {
            "beta0": float(beta0),
            "beta": beta_orig.tolist(),
            "optimal_value": float(obj)
        }