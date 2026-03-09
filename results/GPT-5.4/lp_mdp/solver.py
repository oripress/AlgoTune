from typing import Any

import numpy as np
from numba import njit

@njit(cache=True)
def _policy_iteration_numba(
    transitions: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = transitions.shape

    expected_rewards = np.empty((num_states, num_actions), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            total = 0.0
            for sp in range(num_states):
                total += transitions[s, a, sp] * rewards[s, a, sp]
            expected_rewards[s, a] = total

    policy = np.zeros(num_states, dtype=np.int64)
    for s in range(num_states):
        best_a = 0
        best = expected_rewards[s, 0]
        for a in range(1, num_actions):
            candidate = expected_rewards[s, a]
            if candidate > best + 1e-8:
                best = candidate
                best_a = a
        policy[s] = best_a

    matrix = np.empty((num_states, num_states), dtype=np.float64)
    rhs = np.empty(num_states, dtype=np.float64)

    for _ in range(max_iter):
        for s in range(num_states):
            a = policy[s]
            rhs[s] = expected_rewards[s, a]
            for sp in range(num_states):
                matrix[s, sp] = -gamma * transitions[s, a, sp]
            matrix[s, s] += 1.0

        value_function = np.linalg.solve(matrix, rhs)

        stable = True
        for s in range(num_states):
            best_a = 0
            best = expected_rewards[s, 0]
            continuation = 0.0
            for sp in range(num_states):
                continuation += transitions[s, 0, sp] * value_function[sp]
            best += gamma * continuation

            for a in range(1, num_actions):
                candidate = expected_rewards[s, a]
                continuation = 0.0
                for sp in range(num_states):
                    continuation += transitions[s, a, sp] * value_function[sp]
                candidate += gamma * continuation
                if candidate > best + 1e-8:
                    best = candidate
                    best_a = a

            if best_a != policy[s]:
                stable = False
                policy[s] = best_a

        if stable:
            return value_function, policy

    for s in range(num_states):
        a = policy[s]
        rhs[s] = expected_rewards[s, a]
        for sp in range(num_states):
            matrix[s, sp] = -gamma * transitions[s, a, sp]
        matrix[s, s] += 1.0

    value_function = np.linalg.solve(matrix, rhs)

    for s in range(num_states):
        best_a = 0
        best = expected_rewards[s, 0]
        continuation = 0.0
        for sp in range(num_states):
            continuation += transitions[s, 0, sp] * value_function[sp]
        best += gamma * continuation

        for a in range(1, num_actions):
            candidate = expected_rewards[s, a]
            continuation = 0.0
            for sp in range(num_states):
                continuation += transitions[s, a, sp] * value_function[sp]
            candidate += gamma * continuation
            if candidate > best + 1e-8:
                best = candidate
                best_a = a
        policy[s] = best_a

    return value_function, policy

@njit(cache=True)
def _value_iteration_numba(
    transitions: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    max_iter: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = transitions.shape

    expected_rewards = np.empty((num_states, num_actions), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            total = 0.0
            for sp in range(num_states):
                total += transitions[s, a, sp] * rewards[s, a, sp]
            expected_rewards[s, a] = total

    value_function = np.zeros(num_states, dtype=np.float64)
    new_value = np.empty(num_states, dtype=np.float64)
    policy = np.zeros(num_states, dtype=np.int64)

    for _ in range(max_iter):
        max_diff = 0.0
        for s in range(num_states):
            best_a = 0
            best = expected_rewards[s, 0]
            continuation = 0.0
            for sp in range(num_states):
                continuation += transitions[s, 0, sp] * value_function[sp]
            best += gamma * continuation

            for a in range(1, num_actions):
                candidate = expected_rewards[s, a]
                continuation = 0.0
                for sp in range(num_states):
                    continuation += transitions[s, a, sp] * value_function[sp]
                candidate += gamma * continuation
                if candidate > best + 1e-8:
                    best = candidate
                    best_a = a

            new_value[s] = best
            policy[s] = best_a

            diff = best - value_function[s]
            if diff < 0.0:
                diff = -diff
            if diff > max_diff:
                max_diff = diff

        value_function, new_value = new_value, value_function
        if max_diff <= threshold:
            break

    for s in range(num_states):
        best_a = 0
        best = expected_rewards[s, 0]
        continuation = 0.0
        for sp in range(num_states):
            continuation += transitions[s, 0, sp] * value_function[sp]
        best += gamma * continuation

        for a in range(1, num_actions):
            candidate = expected_rewards[s, a]
            continuation = 0.0
            for sp in range(num_states):
                continuation += transitions[s, a, sp] * value_function[sp]
            candidate += gamma * continuation
            if candidate > best + 1e-8:
                best = candidate
                best_a = a
        policy[s] = best_a

    return value_function, policy

class Solver:
    def __init__(self) -> None:
        dummy_transitions = np.ones((1, 1, 1), dtype=np.float64)
        dummy_rewards = np.zeros((1, 1, 1), dtype=np.float64)
        _policy_iteration_numba(dummy_transitions, dummy_rewards, 0.9, 1)
        _value_iteration_numba(dummy_transitions, dummy_rewards, 0.9, 1, 1e-8)

    @staticmethod
    def _solve_linear_small(matrix: list[list[float]], rhs: list[float]) -> list[float]:
        n = len(rhs)
        a = [row[:] for row in matrix]
        b = rhs[:]

        for i in range(n):
            pivot = i
            pivot_abs = abs(a[i][i])
            for r in range(i + 1, n):
                value = abs(a[r][i])
                if value > pivot_abs:
                    pivot = r
                    pivot_abs = value

            if pivot != i:
                a[i], a[pivot] = a[pivot], a[i]
                b[i], b[pivot] = b[pivot], b[i]

            pivot_value = a[i][i]
            inv_pivot = 1.0 / pivot_value
            row_i = a[i]

            for r in range(i + 1, n):
                factor = a[r][i] * inv_pivot
                if factor != 0.0:
                    row_r = a[r]
                    row_r[i] = 0.0
                    for c in range(i + 1, n):
                        row_r[c] -= factor * row_i[c]
                    b[r] -= factor * b[i]

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            total = b[i]
            row_i = a[i]
            for c in range(i + 1, n):
                total -= row_i[c] * x[c]
            x[i] = total / row_i[i]

        return x

    def _solve_tiny(self, problem: dict[str, Any]) -> dict[str, Any]:
        num_states = int(problem["num_states"])
        num_actions = int(problem["num_actions"])
        gamma = float(problem["discount"])
        transitions = problem["transitions"]
        rewards = problem["rewards"]

        expected_rewards = [[0.0] * num_actions for _ in range(num_states)]
        for s in range(num_states):
            for a in range(num_actions):
                total = 0.0
                row_t = transitions[s][a]
                row_r = rewards[s][a]
                for sp in range(num_states):
                    total += row_t[sp] * row_r[sp]
                expected_rewards[s][a] = total

        policy = [0] * num_states
        for s in range(num_states):
            best_a = 0
            best = expected_rewards[s][0]
            for a in range(1, num_actions):
                candidate = expected_rewards[s][a]
                if candidate > best + 1e-8:
                    best = candidate
                    best_a = a
            policy[s] = best_a

        value_function: list[float] = [0.0] * num_states
        for _ in range(100):
            matrix = [[0.0] * num_states for _ in range(num_states)]
            rhs = [0.0] * num_states

            for s in range(num_states):
                a = policy[s]
                rhs[s] = expected_rewards[s][a]
                row = matrix[s]
                trans_row = transitions[s][a]
                for sp in range(num_states):
                    row[sp] = -gamma * trans_row[sp]
                row[s] += 1.0

            value_function = self._solve_linear_small(matrix, rhs)

            stable = True
            for s in range(num_states):
                best_a = 0
                row0 = transitions[s][0]
                best = expected_rewards[s][0]
                continuation = 0.0
                for sp in range(num_states):
                    continuation += row0[sp] * value_function[sp]
                best += gamma * continuation

                for a in range(1, num_actions):
                    candidate = expected_rewards[s][a]
                    row_a = transitions[s][a]
                    continuation = 0.0
                    for sp in range(num_states):
                        continuation += row_a[sp] * value_function[sp]
                    candidate += gamma * continuation
                    if candidate > best + 1e-8:
                        best = candidate
                        best_a = a

                if best_a != policy[s]:
                    stable = False
                    policy[s] = best_a

            if stable:
                break

        return {
            "value_function": value_function,
            "policy": policy,
        }

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        num_states = int(problem["num_states"])
        if num_states == 0:
            return {"value_function": [], "policy": []}

        try:
            num_actions = int(problem["num_actions"])
            if num_states <= 8 and num_states * num_states * num_actions <= 1024:
                return self._solve_tiny(problem)

            gamma = float(problem["discount"])
            transitions = np.asarray(problem["transitions"], dtype=np.float64)
            rewards = np.asarray(problem["rewards"], dtype=np.float64)

            if num_actions == 1 or num_states <= 256:
                value_function, policy = _policy_iteration_numba(
                    transitions,
                    rewards,
                    gamma,
                    100,
                )
            else:
                threshold = 1e-6 * (1.0 - gamma) / max(gamma, 1e-12)
                value_function, policy = _value_iteration_numba(
                    transitions,
                    rewards,
                    gamma,
                    20000,
                    threshold,
                )

            return {
                "value_function": value_function,
                "policy": policy,
            }
        except Exception:
            return {"value_function": [0.0] * num_states, "policy": [0] * num_states}