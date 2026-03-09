from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def __init__(self) -> None:
        self._scipy_minimize = None

    @staticmethod
    def _parse_problem(problem: dict[str, Any]):
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float).reshape(-1)
        tau = float(problem["tau"])
        M = float(problem["M"])

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if A.ndim == 1:
            A = A.reshape(1, 1)
        if B.ndim == 1:
            B = B.reshape(A.shape[0], -1)
        if C.ndim == 1:
            C = C.reshape(1, -1)

        return A, B, C, y, x0, tau, M

    @staticmethod
    def _forward_states(
        A: np.ndarray,
        B: np.ndarray,
        x0: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        N = w.shape[0]
        n = x0.shape[0]
        x = np.empty((N + 1, n), dtype=float)
        x[0] = x0
        for t in range(N):
            x[t + 1] = A @ x[t] + B @ w[t]
        return x

    @staticmethod
    def _objective_from_residuals(
        w: np.ndarray,
        residuals: np.ndarray,
        tau: float,
        M: float,
    ) -> float:
        norms = np.linalg.norm(residuals, axis=1)
        hub = np.where(norms <= M, norms * norms, 2.0 * M * norms - M * M)
        return float(np.sum(w * w) + tau * np.sum(hub))

    @staticmethod
    def _objective_and_grad(
        w_flat: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        tau: float,
        M: float,
    ):
        N = y.shape[0]
        p = B.shape[1]
        w = w_flat.reshape(N, p)
        x = Solver._forward_states(A, B, x0, w)
        residuals = y - x[:-1] @ C.T
        norms = np.linalg.norm(residuals, axis=1)

        hub = np.where(norms <= M, norms * norms, 2.0 * M * norms - M * M)
        obj = float(np.sum(w * w) + tau * np.sum(hub))

        coeff = np.empty_like(norms)
        mask = norms <= M
        coeff[mask] = 2.0 * tau
        coeff[~mask] = 2.0 * tau * M / np.maximum(norms[~mask], 1e-300)

        qx = -(coeff[:, None] * residuals) @ C

        grad = np.empty_like(w)
        lam_next = np.zeros(A.shape[0], dtype=float)
        BT = B.T
        AT = A.T
        for t in range(N - 1, -1, -1):
            grad[t] = 2.0 * w[t] + BT @ lam_next
            lam_next = qx[t] + AT @ lam_next

        return obj, grad.reshape(-1), x, residuals, norms

    @staticmethod
    def _solve_weighted_lqr(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        q: np.ndarray,
        CtC: np.ndarray | None = None,
        yC: np.ndarray | None = None,
        AT: np.ndarray | None = None,
        I_p: np.ndarray | None = None,
    ):
        N, m = y.shape
        n = A.shape[0]
        p = B.shape[1]

        if N == 0:
            return (
                x0.reshape(1, -1).copy(),
                np.empty((0, p), dtype=float),
                np.empty((0, m), dtype=float),
            )

        if p == 0:
            x = np.empty((N + 1, n), dtype=float)
            x[0] = x0
            for t in range(N):
                x[t + 1] = A @ x[t]
            w = np.empty((N, 0), dtype=float)
            v = y - x[:-1] @ C.T
            return x, w, v

        if n == 1 and p == 1:
            a = float(A[0, 0])
            b = float(B[0, 0])
            ccol = C[:, 0]
            ctc = float(ccol @ ccol) if CtC is None else float(CtC[0, 0])
            yc = y @ ccol if yC is None else yC[:, 0]

            K = np.empty(N, dtype=float)
            k = np.empty(N, dtype=float)

            P = 0.0
            s = 0.0
            a2 = a * a
            ab = a * b
            b2 = b * b

            for t in range(N - 1, -1, -1):
                qt = q[t]
                invG = 1.0 / (1.0 + b2 * P)
                K[t] = ab * P * invG
                k[t] = b * s * invG
                P = qt * ctc + a2 * P * invG
                s = -qt * yc[t] + a * s * invG

            x = np.empty((N + 1, 1), dtype=float)
            w = np.empty((N, 1), dtype=float)
            xs = float(x0[0])
            x[0, 0] = xs

            for t in range(N):
                wt = -(K[t] * xs + k[t])
                w[t, 0] = wt
                xs = a * xs + b * wt
                x[t + 1, 0] = xs

            v = y - x[:-1] * ccol.reshape(1, -1)
            return x, w, v

        if CtC is None:
            CtC = C.T @ C
        if yC is None:
            yC = y @ C
        if I_p is None:
            I_p = np.eye(p, dtype=float)
        if AT is None:
            AT = A.T

        K = np.empty((N, p, n), dtype=float)
        k = np.empty((N, p), dtype=float)

        P_next = np.zeros((n, n), dtype=float)
        s_next = np.zeros(n, dtype=float)

        if p == 1:
            bcol = B[:, 0]
            for t in range(N - 1, -1, -1):
                qt = q[t]
                Qxx = qt * CtC
                qx = -qt * yC[t]

                PA = P_next @ A
                PB0 = P_next @ bcol
                H0 = PB0 @ A
                h0 = bcol @ s_next
                invG0 = 1.0 / (1.0 + PB0 @ bcol)

                GH = H0[None, :] * invG0
                gh0 = h0 * invG0

                P_t = Qxx + AT @ PA - H0[:, None] @ GH
                P_t = 0.5 * (P_t + P_t.T)
                s_t = qx + AT @ s_next - H0 * gh0

                K[t] = GH
                k[t, 0] = gh0
                P_next = P_t
                s_next = s_t
        else:
            for t in range(N - 1, -1, -1):
                qt = q[t]
                Qxx = qt * CtC
                qx = -qt * yC[t]

                PA = P_next @ A
                PB = P_next @ B
                H = PB.T @ A
                h = B.T @ s_next
                G = I_p + PB.T @ B

                GH = np.linalg.solve(G, H)
                Gh = np.linalg.solve(G, h)

                P_t = Qxx + AT @ PA - H.T @ GH
                P_t = 0.5 * (P_t + P_t.T)
                s_t = qx + AT @ s_next - H.T @ Gh

                K[t] = GH
                k[t] = Gh
                P_next = P_t
                s_next = s_t

        x = np.empty((N + 1, n), dtype=float)
        w = np.empty((N, p), dtype=float)
        x[0] = x0

        for t in range(N):
            wt = -(K[t] @ x[t] + k[t])
            w[t] = wt
            x[t + 1] = A @ x[t] + B @ wt

        v = y - x[:-1] @ C.T
        return x, w, v

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        A, B, C, y, x0, tau, M = self._parse_problem(problem)
        N, m = y.shape
        n = A.shape[0]
        p = B.shape[1]

        if x0.shape[0] != n:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        if N == 0:
            return {
                "x_hat": x0.reshape(1, -1),
                "w_hat": np.empty((0, p), dtype=float),
                "v_hat": np.empty((0, m), dtype=float),
            }

        if n == 1 and p == 1 and m == 1:
            a = float(A[0, 0])
            b = float(B[0, 0])
            c = float(C[0, 0])
            x_init = float(x0[0])
            yv = y[:, 0]

            u = np.ones(N, dtype=float)
            q = np.empty(N, dtype=float)
            K = np.empty(N, dtype=float)
            k = np.empty(N, dtype=float)
            x1 = np.empty(N + 1, dtype=float)
            w1 = np.empty(N, dtype=float)
            v1 = np.empty(N, dtype=float)

            a2 = a * a
            ab = a * b
            b2 = b * b
            c2 = c * c
            yc = yv * c
            max_iter = 18
            tol = 1e-8

            for _ in range(max_iter):
                np.divide(tau, u, out=q)

                P = 0.0
                s = 0.0
                for t in range(N - 1, -1, -1):
                    invG = 1.0 / (1.0 + b2 * P)
                    K[t] = ab * P * invG
                    k[t] = b * s * invG
                    P = q[t] * c2 + a2 * P * invG
                    s = -q[t] * yc[t] + a * s * invG

                xs = x_init
                x1[0] = xs
                for t in range(N):
                    wt = -(K[t] * xs + k[t])
                    w1[t] = wt
                    v1[t] = yv[t] - c * xs
                    xs = a * xs + b * wt
                    x1[t + 1] = xs

                if M <= 0:
                    break

                norms = np.abs(v1)
                u_new = np.maximum(1.0, norms / M)
                if np.max(np.abs(u_new - u)) <= tol * (1.0 + np.max(u)):
                    break
                u = u_new

            x = x1[:, None]
            w = w1[:, None]
            v = v1[:, None]
        else:
            CtC = C.T @ C
            yC = y @ C
            AT = A.T
            I_p = np.eye(p, dtype=float) if p > 0 else None

            u = np.ones(N, dtype=float)
            q = np.empty(N, dtype=float)
            max_iter = 20
            tol = 3e-6 if N > 8 else 1e-8

            for _ in range(max_iter):
                np.divide(tau, u, out=q)
                x, w, v = self._solve_weighted_lqr(
                    A, B, C, y, x0, q, CtC, yC, AT, I_p
                )
                if M > 0:
                    norms = np.linalg.norm(v, axis=1)
                    u_new = np.maximum(1.0, norms / M)
                else:
                    break

                if np.max(np.abs(u_new - u)) <= tol * (1.0 + np.max(u)):
                    break
                u = u_new

        if not (
            np.isfinite(x).all()
            and np.isfinite(w).all()
            and np.isfinite(v).all()
        ):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {"x_hat": x, "w_hat": w, "v_hat": v}