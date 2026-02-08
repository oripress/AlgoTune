import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        n_orig, edges = problem
        f1, f2 = float(edges[0]), float(edges[1])
        
        M = n_orig
        N = 2 * M + 1
        
        pif1 = np.pi * f1
        pif2 = np.pi * f2
        
        # Compute phi(k) for k = 0, 1, ..., 2M
        # phi(k) = integral of cos(k*omega) over bands [0, pi*f1] and [pi*f2, pi]
        k_max = 2 * M
        
        phi = np.empty(k_max + 1)
        phi[0] = np.pi * (f1 + 1.0 - f2)
        if k_max > 0:
            k_nz = np.arange(1, k_max + 1, dtype=np.float64)
            phi[1:] = (np.sin(k_nz * pif1) - np.sin(k_nz * pif2)) / k_nz
        
        # Build Q matrix: Q[m,k] = 0.5 * (phi[|m-k|] + phi[m+k])
        idx = np.arange(M + 1)
        diff_idx = np.abs(idx[:, None] - idx[None, :])
        sum_idx = idx[:, None] + idx[None, :]
        
        Q = 0.5 * (phi[diff_idx] + phi[sum_idx])
        
        # Build b vector: b[m] = integral of D(omega)*cos(m*omega) over bands
        # Only passband [0, pi*f1] contributes since D=1 there, D=0 in stopband
        b = np.empty(M + 1)
        b[0] = pif1
        if M > 0:
            m = np.arange(1, M + 1, dtype=np.float64)
            b[1:] = np.sin(m * pif1) / m
        
        # Solve the linear system
        a = np.linalg.solve(Q, b)
        
        # Convert to filter coefficients
        # A(omega) = sum_{k=0}^{M} a[k] * cos(k*omega)
        # h[M] = a[0], h[M±k] = a[k]/2 for k >= 1
        h = np.empty(N)
        h[M] = a[0]
        if M > 0:
            half_a = a[1:] * 0.5
            h[:M] = half_a[::-1]
            h[M+1:] = half_a
        
        return h