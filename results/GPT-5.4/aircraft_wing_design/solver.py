from __future__ import annotations

from typing import Any

import math

import numpy as np

class Solver:
    @staticmethod
    def _fail() -> dict[str, Any]:
        return {"A": [], "S": [], "avg_drag": 0.0, "condition_results": []}

    @staticmethod
    def _golden_search(func, left: float, right: float, tol: float, max_iter: int) -> tuple[float, float]:
        invphi = (math.sqrt(5.0) - 1.0) * 0.5
        c = right - invphi * (right - left)
        d = left + invphi * (right - left)
        fc = func(c)
        fd = func(d)

        for _ in range(max_iter):
            if right - left <= tol:
                break
            if fc <= fd:
                right = d
                d = c
                fd = fc
                c = right - invphi * (right - left)
                fc = func(c)
            else:
                left = c
                c = d
                fc = fd
                d = left + invphi * (right - left)
                fd = func(d)

        x = 0.5 * (left + right)
        fx = func(x)
        if fc < fx:
            x, fx = c, fc
        if fd < fx:
            x, fx = d, fd
        return x, fx

    def _fallback_cvxpy(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            import cvxpy as cp
        except Exception:
            return self._fail()

        try:
            num_conditions = problem["num_conditions"]
            conditions = problem["conditions"]

            A = cp.Variable(pos=True, name="A")
            S = cp.Variable(pos=True, name="S")
            V = [cp.Variable(pos=True, name=f"V_{i}") for i in range(num_conditions)]
            W = [cp.Variable(pos=True, name=f"W_{i}") for i in range(num_conditions)]
            Re = [cp.Variable(pos=True, name=f"Re_{i}") for i in range(num_conditions)]
            C_D = [cp.Variable(pos=True, name=f"C_D_{i}") for i in range(num_conditions)]
            C_L = [cp.Variable(pos=True, name=f"C_L_{i}") for i in range(num_conditions)]
            C_f = [cp.Variable(pos=True, name=f"C_f_{i}") for i in range(num_conditions)]
            W_w = [cp.Variable(pos=True, name=f"W_w_{i}") for i in range(num_conditions)]

            constraints = []
            total_drag = 0.0

            for i, condition in enumerate(conditions):
                CDA0 = float(condition["CDA0"])
                C_Lmax = float(condition["C_Lmax"])
                N_ult = float(condition["N_ult"])
                S_wetratio = float(condition["S_wetratio"])
                V_min = float(condition["V_min"])
                W_0 = float(condition["W_0"])
                W_W_coeff1 = float(condition["W_W_coeff1"])
                W_W_coeff2 = float(condition["W_W_coeff2"])
                e = float(condition["e"])
                k = float(condition["k"])
                mu = float(condition["mu"])
                rho = float(condition["rho"])
                tau = float(condition["tau"])

                drag_i = 0.5 * rho * V[i] ** 2 * C_D[i] * S
                total_drag += drag_i

                constraints.append(
                    C_D[i]
                    >= CDA0 / S + k * C_f[i] * S_wetratio + C_L[i] ** 2 / (math.pi * A * e)
                )
                constraints.append(C_f[i] >= 0.074 / Re[i] ** 0.2)
                constraints.append(Re[i] * mu >= rho * V[i] * cp.sqrt(S / A))
                constraints.append(
                    W_w[i]
                    >= W_W_coeff2 * S
                    + W_W_coeff1 * N_ult * (A ** 1.5) * cp.sqrt(W_0 * W[i]) / tau
                )
                constraints.append(W[i] >= W_0 + W_w[i])
                constraints.append(W[i] <= 0.5 * rho * V[i] ** 2 * C_L[i] * S)
                constraints.append(2 * W[i] / (rho * V_min**2 * S) <= C_Lmax)

            prob = cp.Problem(cp.Minimize(total_drag / num_conditions), constraints)
            prob.solve(gp=True)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or A.value is None:
                return self._fail()

            condition_results = []
            for i in range(num_conditions):
                condition_results.append(
                    {
                        "condition_id": conditions[i]["condition_id"],
                        "V": float(V[i].value),
                        "W": float(W[i].value),
                        "W_w": float(W_w[i].value),
                        "C_L": float(C_L[i].value),
                        "C_D": float(C_D[i].value),
                        "C_f": float(C_f[i].value),
                        "Re": float(Re[i].value),
                        "drag": float(
                            0.5 * conditions[i]["rho"] * V[i].value ** 2 * C_D[i].value * S.value
                        ),
                    }
                )

            return {
                "A": float(A.value),
                "S": float(S.value),
                "avg_drag": float(prob.value),
                "condition_results": condition_results,
            }
        except Exception:
            return self._fail()

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        conditions = problem.get("conditions", [])
        n = int(problem.get("num_conditions", len(conditions)))
        if n <= 0 or len(conditions) != n:
            return self._fail()

        try:
            cond_ids = np.array([int(c["condition_id"]) for c in conditions], dtype=np.int64)
            CDA0 = np.array([float(c["CDA0"]) for c in conditions], dtype=float)
            C_Lmax = np.array([float(c["C_Lmax"]) for c in conditions], dtype=float)
            N_ult = np.array([float(c["N_ult"]) for c in conditions], dtype=float)
            S_wetratio = np.array([float(c["S_wetratio"]) for c in conditions], dtype=float)
            V_min = np.array([float(c["V_min"]) for c in conditions], dtype=float)
            W0 = np.array([float(c["W_0"]) for c in conditions], dtype=float)
            coeff1 = np.array([float(c["W_W_coeff1"]) for c in conditions], dtype=float)
            coeff2 = np.array([float(c["W_W_coeff2"]) for c in conditions], dtype=float)
            e = np.array([float(c["e"]) for c in conditions], dtype=float)
            k = np.array([float(c["k"]) for c in conditions], dtype=float)
            mu = np.array([float(c["mu"]) for c in conditions], dtype=float)
            rho = np.array([float(c["rho"]) for c in conditions], dtype=float)
            tau = np.array([float(c["tau"]) for c in conditions], dtype=float)

            sigma = 0.5 * rho * V_min * V_min * C_Lmax
            if np.any(sigma <= coeff2):
                return self._fail()

            beta = coeff1 * N_ult / tau
            p = beta * np.sqrt(W0)
            alpha = 2.0 * np.sqrt(CDA0 / (math.pi * e))

            def stall_min_s(A: float) -> float:
                a32 = A ** 1.5
                u = p * a32 * np.sqrt(sigma)
                disc = np.sqrt(u * u + 4.0 * W0 * (sigma - coeff2))
                x = (u + disc) / (2.0 * (sigma - coeff2))
                return float(np.max(x * x))

            def weights(A: float, S: float) -> np.ndarray:
                g = p * (A ** 1.5)
                r = np.sqrt(g * g + 4.0 * (W0 + coeff2 * S))
                y = 0.5 * (g + r)
                return y * y

            def avg_drag_relaxed(A: float, S: float) -> float:
                return float(np.mean(alpha * weights(A, S) / math.sqrt(A * S)))

            def optimize_s(A: float) -> tuple[float, float]:
                s_min = stall_min_s(A)
                step = 0.4
                logs = [math.log(s_min) + step * j for j in range(28)]
                vals = [avg_drag_relaxed(A, math.exp(z)) for z in logs]

                for _ in range(8):
                    if vals[-1] < vals[-2]:
                        start = logs[-1]
                        for j in range(1, 9):
                            z = start + step * j
                            logs.append(z)
                            vals.append(avg_drag_relaxed(A, math.exp(z)))
                    else:
                        break

                j = int(np.argmin(vals))
                if j == 0 or j == len(vals) - 1:
                    return math.exp(logs[j]), float(vals[j])

                log_s, f_star = self._golden_search(
                    lambda z: avg_drag_relaxed(A, math.exp(z)),
                    logs[j - 1],
                    logs[j + 1],
                    1e-6,
                    28,
                )
                return math.exp(log_s), float(f_star)

            def objective_log_a(log_a: float) -> float:
                A = math.exp(log_a)
                _, val = optimize_s(A)
                return val

            a_step = 0.4
            a_logs = [-12.0 + a_step * j for j in range(61)]
            a_vals = [objective_log_a(z) for z in a_logs]
            j = int(np.argmin(a_vals))

            if j == 0 or j == len(a_logs) - 1:
                log_a_star = a_logs[j]
            else:
                log_a_star, _ = self._golden_search(
                    objective_log_a,
                    a_logs[j - 1],
                    a_logs[j + 1],
                    1e-5,
                    28,
                )

            A = math.exp(log_a_star)
            S, _ = optimize_s(A)

            W = weights(A, S)
            W_w = W - W0

            re_big = 1e80
            total_drag = 0.0
            condition_results = []

            for i in range(n):
                Re = re_big
                C_f = 0.074 / (Re ** 0.2)
                base = CDA0[i] / S + k[i] * S_wetratio[i] * C_f
                C_L = math.sqrt(base * math.pi * A * e[i])
                C_D = base + C_L * C_L / (math.pi * A * e[i])
                V = math.sqrt(2.0 * W[i] / (rho[i] * C_L * S))

                re_needed = rho[i] * V * math.sqrt(S / A) / mu[i]
                if Re < re_needed:
                    Re = re_needed * (1.0 + 1e-12)
                    C_f = 0.074 / (Re ** 0.2)
                    base = CDA0[i] / S + k[i] * S_wetratio[i] * C_f
                    C_L = math.sqrt(base * math.pi * A * e[i])
                    C_D = base + C_L * C_L / (math.pi * A * e[i])
                    V = math.sqrt(2.0 * W[i] / (rho[i] * C_L * S))

                drag = 0.5 * rho[i] * V * V * C_D * S
                total_drag += drag

                condition_results.append(
                    {
                        "condition_id": int(cond_ids[i]),
                        "V": float(V),
                        "W": float(W[i]),
                        "W_w": float(W_w[i]),
                        "C_L": float(C_L),
                        "C_D": float(C_D),
                        "C_f": float(C_f),
                        "Re": float(Re),
                        "drag": float(drag),
                    }
                )

            avg_drag = total_drag / n
            if not (math.isfinite(A) and math.isfinite(S) and math.isfinite(avg_drag)):
                return self._fallback_cvxpy(problem)

            return {
                "A": float(A),
                "S": float(S),
                "avg_drag": float(avg_drag),
                "condition_results": condition_results,
            }
        except Exception:
            return self._fallback_cvxpy(problem)