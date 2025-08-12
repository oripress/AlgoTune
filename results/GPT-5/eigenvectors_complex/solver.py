from typing import Any

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute eigenvectors of a real square matrix, sorted by corresponding eigenvalues in
        descending order by real part, then imaginary part. Each eigenvector is normalized to
        unit Euclidean norm.

        Returns:
            List[List[complex]]: list of eigenvectors (each a list of complex numbers), sorted
            and normalized. The i-th vector corresponds to the i-th largest eigenvalue under
            the specified ordering.
        """
        # Prepare input as float to leverage real-valued LAPACK path when possible, avoiding extra copies
        A = np.asarray(problem, dtype=float)
        n = A.shape[0]

        # Trivial case
        if n == 1:
            return [[1.0]]

        # 2x2 closed-form fast path with degeneracy guard
        if n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            disc = (a - d) * (a - d) + 4.0 * b * c
            scale = abs(a) + abs(b) + abs(c) + abs(d) + 1.0
            if abs(disc) > 1e-14 * scale * scale:
                sqrt_disc = np.sqrt(disc + 0j)
                lam1 = 0.5 * (a + d + sqrt_disc)
                lam2 = 0.5 * (a + d - sqrt_disc)

                def eigvec_for(lam):
                    # Solve (A - lam I)v = 0 using a robust 2x2 nullspace vector
                    r1 = (a - lam, b)
                    r2 = (c, d - lam)
                    if (abs(r1[0]) + abs(r1[1])) >= (abs(r2[0]) + abs(r2[1])):
                        v = np.array([r1[1], -r1[0]], dtype=complex)
                    else:
                        v = np.array([-r2[1], r2[0]], dtype=complex)
                    nv = np.linalg.norm(v)
                    if nv > 1e-18 * (abs(lam) + scale):
                        v /= nv
                    else:
                        v = np.array([1.0 + 0j, 0.0 + 0j])
                    return v

                w2 = np.array([lam1, lam2], dtype=complex)
                v1 = eigvec_for(lam1)
                v2 = eigvec_for(lam2)
                idx2 = np.array([0, 1])
                order2 = np.lexsort((idx2, -w2.imag, -w2.real))
                return [
                    (v1 if order2[0] == 0 else v2).tolist(),
                    (v2 if order2[1] == 1 else v1).tolist(),
                ]

        # General case
        w, V = np.linalg.eig(A)

        # Stable order: descending by real part, then imaginary part, tie-break by original index
        idx = np.arange(n)
        order = np.lexsort((idx, -w.imag, -w.real))

        # Build output as list of eigenvectors corresponding to sorted eigenvalues in one shot
        return (V[:, order].T).tolist()