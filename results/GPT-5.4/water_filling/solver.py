from math import log
from typing import Any

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _water_fill_numba(alpha: np.ndarray, p_total: float):
    n = alpha.size
    a_sorted = np.sort(alpha)

    prefix = 0.0
    water_level = 0.0
    for i in range(n):
        prefix += a_sorted[i]
        w_candidate = (p_total + prefix) / (i + 1)
        if i == n - 1 or w_candidate <= a_sorted[i + 1]:
            water_level = w_candidate
            break

    x = np.empty(n, dtype=np.float64)
    power_sum = 0.0
    first_active = -1
    active_count = 0
    capacity = 0.0

    for i in range(n):
        xi = water_level - alpha[i]
        if xi > 0.0:
            x[i] = xi
            power_sum += xi
            active_count += 1
            if first_active == -1:
                first_active = i
        else:
            x[i] = 0.0
            capacity += np.log(alpha[i])

    if first_active != -1:
        x[first_active] += p_total - power_sum
        if active_count == 1:
            capacity += np.log(alpha[first_active] + x[first_active])
        else:
            capacity += (active_count - 1) * np.log(water_level)
            capacity += np.log(alpha[first_active] + x[first_active])

    return x, capacity

class Solver:
    def __init__(self):
        _water_fill_numba(np.array([1.0, 2.0], dtype=np.float64), 1.0)

    def solve(self, problem, **kwargs) -> Any:
        alpha = problem["alpha"]
        p_total = float(problem["P_total"])
        n = len(alpha)

        if n == 0 or p_total <= 0.0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        if n == 1:
            a0 = float(alpha[0])
            if a0 <= 0.0:
                return {"x": [float("nan")], "Capacity": float("nan")}
            return {"x": [p_total], "Capacity": log(a0 + p_total)}

        if n == 2:
            a0 = float(alpha[0])
            a1 = float(alpha[1])
            if a0 <= 0.0 or a1 <= 0.0:
                return {"x": [float("nan"), float("nan")], "Capacity": float("nan")}
            if a0 <= a1:
                gap = a1 - a0
                if p_total <= gap:
                    x0 = p_total
                    x1 = 0.0
                else:
                    w = 0.5 * (p_total + a0 + a1)
                    x0 = w - a0
                    x1 = w - a1
            else:
                gap = a0 - a1
                if p_total <= gap:
                    x0 = 0.0
                    x1 = p_total
                else:
                    w = 0.5 * (p_total + a0 + a1)
                    x0 = w - a0
                    x1 = w - a1
            return {"x": [x0, x1], "Capacity": log(a0 + x0) + log(a1 + x1)}

        if n <= 32:
            a_sorted = sorted(alpha)
            if a_sorted[0] <= 0.0:
                return {"x": [float("nan")] * n, "Capacity": float("nan")}

            prefix = 0.0
            water_level = 0.0
            for i, a in enumerate(a_sorted, 1):
                prefix += a
                w_candidate = (p_total + prefix) / i
                if i == n or w_candidate <= a_sorted[i]:
                    water_level = w_candidate
                    break

            x = [0.0] * n
            power_sum = 0.0
            first_active = -1
            for i, a in enumerate(alpha):
                xi = water_level - a
                if xi > 0.0:
                    x[i] = xi
                    power_sum += xi
                    if first_active == -1:
                        first_active = i

            x[first_active] += p_total - power_sum

            return {
                "x": x,
                "Capacity": sum(log(a + xi) for a, xi in zip(alpha, x)),
            }

        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if float(alpha_arr.min()) <= 0.0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        x, capacity = _water_fill_numba(alpha_arr, p_total)
        return {"x": x, "Capacity": float(capacity)}