import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        n, edges = problem
        f1, f2 = edges
        
        numtaps = 2 * n + 1
        M = n
        
        n_arr = np.arange(numtaps)
        
        # q = f1 * sinc(f1 * n) + 1 * sinc(1 * n) - f2 * sinc(f2 * n)
        q = f1 * np.sinc(f1 * n_arr) + np.sinc(n_arr) - f2 * np.sinc(f2 * n_arr)
        
        # Q1 = toeplitz(q[:M+1])
        # Q2 = hankel(q[:M+1], q[M:])
        # Q = Q1 + Q2
        
        q_M = q[:M+1]
        
        # Construct Q directly
        i = np.arange(M + 1)[:, None]
        j = np.arange(M + 1)[None, :]
        
        Q = q[np.abs(i - j)] + q[i + j]
        
        # b = f1 * sinc(f1 * n[:M+1])
        b = f1 * np.sinc(f1 * np.arange(M + 1))
        
        # Solve Q a = b
        a = scipy.linalg.solve(Q, b, assume_a='pos', check_finite=False)
        
        # coeffs = np.hstack((a[:0:-1], 2 * a[0], a[1:]))
        coeffs = np.empty(numtaps)
        coeffs[M] = 2 * a[0]
        coeffs[M+1:] = a[1:]
        coeffs[:M] = a[1:][::-1]
        
        return coeffs