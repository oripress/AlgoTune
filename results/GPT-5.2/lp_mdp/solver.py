from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def __init__(self) -> None:
        # Small caches to reduce allocations across calls (if any).
        self._arange_cache: dict[int, np.ndarray] = {}
        self._eye_cache: dict[int, np.ndarray] = {}

    def _arange(self, n: int) -> np.ndarray:
        arr = self._arange_cache.get(n)
        if arr is None:
            arr = np.arange(n, dtype=np.int64)
            self._arange_cache[n] = arr
        return arr

    def _eye(self, n: int) -> np.ndarray:
        eye = self._eye_cache.get(n)
        if eye is None:
            eye = np.eye(n, dtype=np.float64)
            self._eye_cache[n] = eye
        return eye

    @staticmethod
    def _q_inplace_from_v(
        P_flat: np.ndarray, rbar: np.ndarray, V: np.ndarray, gamma: float, S: int, A: int
    ) -> np.ndarray:
        """
        Compute Q(s,a) = rbar(s,a) + gamma * sum_{s'} P(s,a,s') * V(s')
        Returns a (S,A) array, using the matvec result as workspace.
        """
        Q = (P_flat @ V).reshape(S, A)  # new array from matvec, reshaped view
        Q *= gamma
        Q += rbar
        return Q

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        S = int(problem["num_states"])
        A = int(problem["num_actions"])
        gamma = float(problem["discount"])

        P = np.asarray(problem["transitions"], dtype=np.float64)  # (S,A,S)
        R = np.asarray(problem["rewards"], dtype=np.float64)  # (S,A,S)

        # Expected immediate reward rbar(s,a) = sum_{s'} P(s'|s,a) * R(s,a,s')
        # Einsum avoids allocating the full (S,A,S) product array.
        rbar = np.einsum("sat,sat->sa", P, R, optimize=False)

        # Flatten transitions for fast Q computation via GEMV.
        P_flat = P.reshape(S * A, S)

        idx = self._arange(S)

        # For small/medium S: do policy iteration; for larger S: pure value iteration.
        if S <= 512:
            policy = np.zeros(S, dtype=np.int64)
            V = np.zeros(S, dtype=np.float64)
            I = self._eye(S)

            for _ in range(60):
                Ppi = P[idx, policy, :]  # (S,S)
                rpi = rbar[idx, policy]  # (S,)

                M = I - gamma * Ppi
                try:
                    V = np.linalg.solve(M, rpi)
                except np.linalg.LinAlgError:
                    V = np.zeros(S, dtype=np.float64)
                    break

                Q = self._q_inplace_from_v(P_flat, rbar, V, gamma, S, A)
                new_policy = Q.argmax(axis=1)

                if np.array_equal(new_policy, policy):
                    policy = new_policy
                    break
                policy = new_policy

            # Final policy/value from V
            Q = self._q_inplace_from_v(P_flat, rbar, V, gamma, S, A)
            policy = Q.argmax(axis=1)
            return {"value_function": V.tolist(), "policy": policy.tolist()}

        # Large S: value iteration only (avoid O(S^3) solves).
        V = np.zeros(S, dtype=np.float64)

        # Use contraction bound to stop once we're safely within 1e-4 of V*.
        # ||V_k - V*||_inf <= gamma/(1-gamma) * ||V_{k}-V_{k-1}||_inf
        # So we stop when successive diff is small enough.
        target_v_err = 3e-5
        diff_tol = (1.0 - gamma) / max(gamma, 1e-12) * target_v_err

        for _ in range(20000):
            Q = self._q_inplace_from_v(P_flat, rbar, V, gamma, S, A)
            Vn = Q.max(axis=1)
            if np.max(np.abs(Vn - V)) < diff_tol:
                V = Vn
                break
            V = Vn

        Q = self._q_inplace_from_v(P_flat, rbar, V, gamma, S, A)
        policy = Q.argmax(axis=1)
        return {"value_function": V.tolist(), "policy": policy.tolist()}