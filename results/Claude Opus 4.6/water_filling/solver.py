import math

try:
    from solver_core import solve_water_filling
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

class Solver:
    def solve(self, problem, **kwargs):
        alpha_list = problem["alpha"]
        P_total = float(problem["P_total"])
        n = len(alpha_list)

        if n == 0 or P_total <= 0:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        if HAS_CYTHON:
            al = alpha_list if isinstance(alpha_list, list) else list(alpha_list)
            result_x, result_cap = solve_water_filling(al, P_total)
            if result_x is None:
                return {"x": [float("nan")] * n, "Capacity": float("nan")}
            return {"x": result_x, "Capacity": result_cap}

        # Fallback pure Python
        alpha = alpha_list if isinstance(alpha_list, list) else list(alpha_list)
        for a in alpha:
            if a <= 0:
                return {"x": [float("nan")] * n, "Capacity": float("nan")}
        sorted_alpha = sorted(alpha)
        prefix_sum = 0.0
        w = 0.0
        for k_idx in range(n):
            prefix_sum += sorted_alpha[k_idx]
            k = k_idx + 1
            w_candidate = (P_total + prefix_sum) / k
            if k == n or w_candidate <= sorted_alpha[k]:
                w = w_candidate
                break
        _log = math.log
        x_opt = [0.0] * n
        s = 0.0
        for i in range(n):
            val = w - alpha[i]
            if val > 0.0:
                x_opt[i] = val
                s += val
        if s > 0.0:
            scale = P_total / s
            cap = 0.0
            for i in range(n):
                xi = x_opt[i] * scale
                x_opt[i] = xi
                cap += _log(alpha[i] + xi)
        else:
            cap = 0.0
            for i in range(n):
                cap += _log(alpha[i])
        return {"x": x_opt, "Capacity": cap}