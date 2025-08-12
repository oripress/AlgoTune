import numpy as np

# Aliases for speed
_eig = np.linalg.eig
_inv = np.linalg.inv
_dot = np.dot
_solve = np.linalg.solve

class Solver:
    def solve(self, problem):
        # Load & validate A
        try:
            A = np.array(problem["A"], dtype=float)
        except Exception:
            return {"is_stable": False, "P": None}
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return {"is_stable": False, "P": None}
        n = A.shape[0]
        # Special-case 1x1
        if n == 1:
            a = A[0, 0]
            if abs(a) >= 1.0:
                return {"is_stable": False, "P": None}
            return {"is_stable": True, "P": np.array([[1.0/(1.0 - a*a)]])}
        # Special-case 2x2
        if n == 2:
            a, b = A[0, 0], A[0, 1]
            c, d = A[1, 0], A[1, 1]
            tau = a + d
            delta = a*d - b*c
            disc = tau*tau - 4.0*delta
            # sqrt may produce complex for disc<0
            sqrt_disc = disc**0.5
            lam1 = 0.5*(tau + sqrt_disc)
            lam2 = 0.5*(tau - sqrt_disc)
            if abs(lam1) >= 1.0 or abs(lam2) >= 1.0:
                return {"is_stable": False, "P": None}
            # solve linear system for [p,q,r]
            M = np.empty((3, 3), dtype=float)
            M[0, 0], M[0, 1], M[0, 2] = a*a - 1.0, 2.0*a*c, c*c
            M[1, 0], M[1, 1], M[1, 2] = a*b, a*d + b*c - 1.0, c*d
            M[2, 0], M[2, 1], M[2, 2] = b*b, 2.0*b*d, d*d - 1.0
            v = np.array([-1.0, 0.0, -1.0], dtype=float)
            p, q, r = _solve(M, v)
            P = np.array([[p, q], [q, r]], dtype=float)
            return {"is_stable": True, "P": P}
        # General case: diagonalize
        lam, V = _eig(A)
        if (abs(lam) >= 1.0).any():
            return {"is_stable": False, "P": None}
        V_inv = _inv(V)
        C = _dot(V.T, V)
        Z = C / (1.0 - lam[:, None] * lam[None, :])
        P = _dot(V_inv.T, _dot(Z, V_inv))
        P = 0.5*(P + P.T)
        return {"is_stable": True, "P": P}