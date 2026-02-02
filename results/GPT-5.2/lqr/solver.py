from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        Q = np.asarray(problem["Q"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=float)

        n, m = B.shape
        if x0.ndim == 1:
            x0 = x0.reshape(n, 1)
        else:
            x0 = x0.reshape(n, 1)

        if T <= 0:
            return {"U": np.zeros((0, m), dtype=float)}

        AT = A.T
        BT = B.T
        dot = np.dot

        # Store feedback gains K_t for forward rollout
        K = np.empty((T, m, n), dtype=float)

        # Buffers for backward Riccati recursion
        S = P.copy()
        Snew = np.empty((n, n), dtype=float)

        SA = np.empty((n, n), dtype=float)
        SB = np.empty((n, m), dtype=float)
        ATA = np.empty((n, n), dtype=float)
        ATB = np.empty((n, m), dtype=float)

        M2 = np.empty((m, n), dtype=float)
        BtSB = np.empty((m, m), dtype=float)
        M1 = np.empty((m, m), dtype=float)

        tmp_nn = np.empty((n, n), dtype=float)

        solve = np.linalg.solve

        for t in range(T - 1, -1, -1):
            # SA = S A, SB = S B
            dot(S, A, out=SA)
            dot(S, B, out=SB)

            # M2 = B^T S A
            dot(BT, SA, out=M2)

            # M1 = R + B^T S B
            dot(BT, SB, out=BtSB)
            M1[...] = BtSB
            M1 += R

            try:
                K[t] = solve(M1, M2)
            except np.linalg.LinAlgError:
                K[t] = np.linalg.pinv(M1) @ M2

            # Snew = Q + A^T S A - A^T S B K
            dot(AT, SA, out=ATA)
            dot(AT, SB, out=ATB)
            dot(ATB, K[t], out=tmp_nn)

            Snew[...] = Q
            Snew += ATA
            Snew -= tmp_nn

            # Symmetrize without allocating a new matrix
            np.add(Snew, Snew.T, out=tmp_nn)
            tmp_nn *= 0.5

            # rotate buffers: S <- symmetrized, reuse old S as next Snew
            S, Snew, tmp_nn = tmp_nn, S, Snew

        # Forward rollout
        U = np.empty((T, m), dtype=float)
        x = x0.copy()
        x_next = np.empty((n, 1), dtype=float)
        u = np.empty((m, 1), dtype=float)
        bu = np.empty((n, 1), dtype=float)

        for t in range(T):
            dot(K[t], x, out=u)
            u *= -1.0
            U[t, :] = u[:, 0]

            dot(A, x, out=x_next)
            dot(B, u, out=bu)
            x_next += bu
            x, x_next = x_next, x

        return {"U": U}