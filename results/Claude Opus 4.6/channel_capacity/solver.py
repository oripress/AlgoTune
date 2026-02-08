import numpy as np
import math
import numba as nb

@nb.njit(cache=True, fastmath=True)
def ba_update(P, PlogP_colsum, x, y_arr, c, m, n):
    """One Blahut-Arimoto update step. Returns x_new in-place via c array."""
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += P[i, j] * x[j]
        y_arr[i] = s
    
    for j in range(n):
        s = 0.0
        for i in range(m):
            yi = y_arr[i]
            if yi > 0.0:
                s += P[i, j] * math.log(yi)
        c[j] = PlogP_colsum[j] - s
    
    Z = 0.0
    for j in range(n):
        v = x[j] * math.exp(c[j])
        c[j] = v
        Z += v
    
    inv_Z = 1.0 / Z
    for j in range(n):
        c[j] = c[j] * inv_Z


@nb.njit(cache=True, fastmath=True)
def blahut_arimoto(P, m, n):
    """Blahut-Arimoto with SQUAREM acceleration."""
    PlogP_colsum = np.zeros(n)
    for j in range(n):
        s = 0.0
        for i in range(m):
            p = P[i, j]
            if p > 0.0:
                s += p * math.log(p)
        PlogP_colsum[j] = s
    
    x = np.full(n, 1.0 / n)
    x1 = np.empty(n)
    x2 = np.empty(n)
    r = np.empty(n)
    v = np.empty(n)
    y_arr = np.empty(m)
    c = np.empty(n)
    ln2 = math.log(2.0)
    tol = 1e-11
    
    for iteration in range(500):
        # Step 1: x -> x1
        ba_update(P, PlogP_colsum, x, y_arr, c, m, n)
        for j in range(n):
            x1[j] = c[j]
        
        # Check convergence
        converged = True
        for j in range(n):
            if abs(x1[j] - x[j]) >= tol:
                converged = False
                break
        if converged:
            for j in range(n):
                x[j] = x1[j]
            break
        
        # Step 2: x1 -> x2
        ba_update(P, PlogP_colsum, x1, y_arr, c, m, n)
        for j in range(n):
            x2[j] = c[j]
        
        # SQUAREM acceleration
        # r = x1 - x, v = (x2 - x1) - r
        r_norm_sq = 0.0
        v_norm_sq = 0.0
        for j in range(n):
            rj = x1[j] - x[j]
            vj = (x2[j] - x1[j]) - rj
            r[j] = rj
            v[j] = vj
            r_norm_sq += rj * rj
            v_norm_sq += vj * vj
        
        if v_norm_sq < 1e-30:
            for j in range(n):
                x[j] = x2[j]
            continue
        
        alpha = -math.sqrt(r_norm_sq / v_norm_sq)
        if alpha > -1.0:
            alpha = -1.0
        
        # x_new = x - 2*alpha*r + alpha^2*v
        # Project onto simplex (ensure non-negative, sum to 1)
        Z = 0.0
        for j in range(n):
            xj = x[j] - 2.0 * alpha * r[j] + alpha * alpha * v[j]
            if xj < 0.0:
                xj = 0.0
            c[j] = xj
            Z += xj
        
        if Z > 0.0:
            inv_Z = 1.0 / Z
            for j in range(n):
                x[j] = c[j] * inv_Z
        else:
            for j in range(n):
                x[j] = x2[j]
        
        # Do one stabilization step
        ba_update(P, PlogP_colsum, x, y_arr, c, m, n)
        for j in range(n):
            x[j] = c[j]
    
    # Final capacity
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += P[i, j] * x[j]
        y_arr[i] = s
    
    C = 0.0
    for j in range(n):
        s = 0.0
        for i in range(m):
            yi = y_arr[i]
            if yi > 0.0:
                s += P[i, j] * math.log(yi)
        cj = PlogP_colsum[j] - s
        C += x[j] * cj
    C = C / ln2
    if C < 0.0:
        C = 0.0
    
    return x, C


def blahut_arimoto_numpy(P, m, n):
    ln2 = math.log(2.0)
    mask = P > 0
    logP = np.zeros_like(P)
    logP[mask] = np.log(P[mask])
    PlogP_colsum = (P * logP).sum(axis=0)
    PT = np.ascontiguousarray(P.T)
    x = np.ones(n) / n
    
    for iteration in range(2000):
        y = P @ x
        np.maximum(y, 1e-300, out=y)
        logy = np.log(y)
        c = PlogP_colsum - PT @ logy
        x_new = x * np.exp(c)
        s = x_new.sum()
        x_new *= (1.0 / s)
        
        if iteration % 5 == 4:
            if np.max(np.abs(x_new - x)) < 1e-11:
                x = x_new
                break
        x = x_new
    
    y = P @ x
    np.maximum(y, 1e-300, out=y)
    logy = np.log(y)
    c_final = PlogP_colsum - PT @ logy
    C = float(np.dot(x, c_final) / ln2)
    if C < 0:
        C = 0.0
    
    return x, C


class Solver:
    def __init__(self):
        P_dummy = np.array([[0.5, 0.5], [0.5, 0.5]])
        blahut_arimoto(P_dummy, 2, 2)
    
    def solve(self, problem, **kwargs):
        P = np.asarray(problem["P"], dtype=np.float64)
        m, n = P.shape
        
        if n <= 1:
            if n == 0 or m == 0:
                return None
            return {"x": [1.0], "C": 0.0}
        
        if not P.flags['C_CONTIGUOUS']:
            P = np.ascontiguousarray(P)
        
        if m * n <= 10000:
            x, C = blahut_arimoto(P, m, n)
        else:
            x, C = blahut_arimoto_numpy(P, m, n)
        
        return {"x": x.tolist(), "C": float(C)}