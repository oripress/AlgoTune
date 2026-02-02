import numpy as np
from numba import njit

@njit(cache=True)
def khachiyan_mvee(points, tol=1e-8, max_iter=2000):
    n, d = points.shape
    d1 = d + 1
    
    Q = np.empty((n, d1))
    Q[:, :d] = points
    Q[:, d] = 1.0
    
    u = np.ones(n) / n
    
    # Initial P = Q^T diag(u) Q
    P = np.zeros((d1, d1))
    for i in range(n):
        for j in range(d1):
            for k in range(d1):
                P[j, k] += u[i] * Q[i, j] * Q[i, k]
    
    P_inv = np.linalg.inv(P)
    
    for _ in range(max_iter):
        # g[i] = Q[i]^T @ P_inv @ Q[i]
        QP = Q @ P_inv
        g = np.sum(QP * Q, axis=1)
        
        j = np.argmax(g)
        max_g = g[j]
        
        if max_g <= d1 + tol:
            break
        
        step = (max_g - d1) / (d1 * (max_g - 1))
        
        # Sherman-Morrison: (P + alpha*q*q^T)^{-1}
        alpha = step / (1 - step)
        q = Q[j]
        Pq = P_inv @ q
        denom = 1 + alpha * (q @ Pq)
        
        # Update P_inv using rank-1 formula
        for a in range(d1):
            for b in range(d1):
                P_inv[a, b] = (P_inv[a, b] - alpha * Pq[a] * Pq[b] / denom) / (1 - step)
        
        u = (1 - step) * u
        u[j] += step
    
    return u

class Solver:
    def solve(self, problem, **kwargs):
        points = np.array(problem["points"], dtype=np.float64)
        n, d = points.shape
        
        u = khachiyan_mvee(points, tol=1e-8, max_iter=2000)
        
        center = u @ points
        Q_centered = points - center
        S = (Q_centered.T * u) @ Q_centered
        
        S_inv = np.linalg.inv(S)
        A = S_inv / d
        
        resid = points - center
        dists_sq = np.sum((resid @ A) * resid, axis=1)
        max_dist_sq = np.max(dists_sq)
        
        if max_dist_sq > 1.0:
            A = A / max_dist_sq
        
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-12)
        X = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        X = (X + X.T) / 2
        
        Y = -X @ center
        
        transformed = (X @ points.T).T + Y
        norms = np.linalg.norm(transformed, axis=1)
        max_norm = np.max(norms)
        
        if max_norm > 1.0:
            scale = max_norm + 1e-9
            X = X / scale
            Y = Y / scale
        
        sign, logdet = np.linalg.slogdet(X)
        objective_value = -logdet if sign > 0 else float('inf')
        
        return {
            "objective_value": float(objective_value),
            "ellipsoid": {"X": X.tolist(), "Y": Y.tolist()}
        }