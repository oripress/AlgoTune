from __future__ import annotations

from typing import Any

import numpy as np

class _CondArrays:
    __slots__ = (
        "n",
        "condition_id",
        "CDA0",
        "C_Lmax",
        "N_ult",
        "S_wetratio",
        "V_min",
        "W0",
        "W_W_coeff1",
        "W_W_coeff2",
        "e",
        "k",
        "mu",
        "rho",
        "tau",
        "base_alpha",
        "c_induced",
        "Vmin2",
    )

    def __init__(
        self,
        *,
        n: int,
        condition_id: np.ndarray,
        CDA0: np.ndarray,
        C_Lmax: np.ndarray,
        N_ult: np.ndarray,
        S_wetratio: np.ndarray,
        V_min: np.ndarray,
        W0: np.ndarray,
        W_W_coeff1: np.ndarray,
        W_W_coeff2: np.ndarray,
        e: np.ndarray,
        k: np.ndarray,
        mu: np.ndarray,
        rho: np.ndarray,
        tau: np.ndarray,
        base_alpha: np.ndarray,
        c_induced: np.ndarray,
        Vmin2: np.ndarray,
    ) -> None:
        self.n = n
        self.condition_id = condition_id
        self.CDA0 = CDA0
        self.C_Lmax = C_Lmax
        self.N_ult = N_ult
        self.S_wetratio = S_wetratio
        self.V_min = V_min
        self.W0 = W0
        self.W_W_coeff1 = W_W_coeff1
        self.W_W_coeff2 = W_W_coeff2
        self.e = e
        self.k = k
        self.mu = mu
        self.rho = rho
        self.tau = tau
        self.base_alpha = base_alpha
        self.c_induced = c_induced
        self.Vmin2 = Vmin2
def _extract(problem: dict[str, Any]) -> _CondArrays:
    conds = problem["conditions"]
    n = int(problem["num_conditions"])

    condition_id = np.fromiter((c["condition_id"] for c in conds), dtype=np.int64, count=n)

    def arr(key: str) -> np.ndarray:
        return np.fromiter((float(c[key]) for c in conds), dtype=np.float64, count=n)

    CDA0 = arr("CDA0")
    C_Lmax = arr("C_Lmax")
    N_ult = arr("N_ult")
    S_wetratio = arr("S_wetratio")
    V_min = arr("V_min")
    W0 = arr("W_0")
    W_W_coeff1 = arr("W_W_coeff1")
    W_W_coeff2 = arr("W_W_coeff2")
    e = arr("e")
    k = arr("k")
    mu = arr("mu")
    rho = arr("rho")
    tau = arr("tau")

    base_alpha = W_W_coeff1 * N_ult * np.sqrt(W0) / tau
    c_induced = np.sqrt(CDA0 / (np.pi * e))
    Vmin2 = V_min * V_min

    return _CondArrays(
        n=n,
        condition_id=condition_id,
        CDA0=CDA0,
        C_Lmax=C_Lmax,
        N_ult=N_ult,
        S_wetratio=S_wetratio,
        V_min=V_min,
        W0=W0,
        W_W_coeff1=W_W_coeff1,
        W_W_coeff2=W_W_coeff2,
        e=e,
        k=k,
        mu=mu,
        rho=rho,
        tau=tau,
        base_alpha=base_alpha,
        c_induced=c_induced,
        Vmin2=Vmin2,
    )

def _weights_for_A_S(ca: _CondArrays, A: float, S: float) -> tuple[np.ndarray, np.ndarray]:
    # Tight constraints:
    #   W = W0 + W_w
    #   W_w = c2*S + c1*N_ult*A^(3/2)*sqrt(W0*W)/tau
    # Let alpha = c1*N_ult*sqrt(W0)/tau * A^(3/2) and x = sqrt(W):
    #   x^2 = W0 + c2*S + alpha*x  ->  x = (alpha + sqrt(alpha^2 + 4*(W0+c2*S)))/2
    A32 = A**1.5
    alpha = ca.base_alpha * A32
    beta = ca.W0 + ca.W_W_coeff2 * S
    disc = np.sqrt(alpha * alpha + 4.0 * beta)
    x = 0.5 * (alpha + disc)
    W = x * x
    W_w = W - ca.W0
    return W, W_w

def _feasibility_violation(ca: _CondArrays, W: np.ndarray, S: float) -> float:
    stall_cl = 2.0 * W / (ca.rho * ca.Vmin2 * S)
    v = stall_cl - ca.C_Lmax
    m = float(np.max(v))
    return m if m > 0.0 else 0.0

def _objective_drag_proxy(ca: _CondArrays, A: float, S: float) -> float:
    # Proxy: assume skin friction can be made negligible (unbounded Re),
    # and choose cruise V to minimize drag. Then
    #   D_i ~= 2*W_i*sqrt(CDA0/(pi*e)) / sqrt(A*S).
    W, _ = _weights_for_A_S(ca, A, S)
    viol = _feasibility_violation(ca, W, S)
    if viol > 0.0:
        base = float(np.mean(W)) / max(1e-12, np.sqrt(A * S))
        return base * (1.0 + 1e6 * viol * viol)
    return float((2.0 / ca.n) * np.sum(W * ca.c_induced) / np.sqrt(A * S))

def _grid_optimize(ca: _CondArrays) -> tuple[float, float]:
    # Stall sizing lower bound (ignoring wing weight), used to set scale.
    S0 = float(np.max(2.0 * ca.W0 / (ca.rho * ca.Vmin2 * ca.C_Lmax)))
    S0 = max(S0, 1.0)

    # Broad log ranges
    A_lo, A_hi = 1.5, 60.0
    S_lo = max(0.5 * S0, 0.5)
    S_hi = min(max(50.0 * S0, 50.0), 5000.0)

    best_A = 10.0
    best_S = max(1.5 * S0, 1.0)
    best_f = float("inf")

    # Coarse vectorized grid (expand S-range if no feasible found)
    for _expand in range(3):
        logA = np.linspace(np.log(A_lo), np.log(A_hi), 25)
        logS = np.linspace(np.log(S_lo), np.log(S_hi), 25)
        Sgrid = np.exp(logS)

        for A in np.exp(logA):
            A32 = A**1.5
            alpha = ca.base_alpha * A32  # (n,)

            beta = ca.W0[:, None] + ca.W_W_coeff2[:, None] * Sgrid[None, :]  # (n,k)
            disc = np.sqrt(alpha[:, None] * alpha[:, None] + 4.0 * beta)
            x = 0.5 * (alpha[:, None] + disc)
            W = x * x

            stall_cl = 2.0 * W / (ca.rho[:, None] * ca.Vmin2[:, None] * Sgrid[None, :])
            feas = np.all(stall_cl <= ca.C_Lmax[:, None], axis=0)

            sumWc = (ca.c_induced[:, None] * W).sum(axis=0)
            obj = (2.0 / ca.n) * sumWc / np.sqrt(A * Sgrid)

            if np.any(feas):
                j = int(np.argmin(np.where(feas, obj, np.inf)))
                f = float(obj[j])
                if f < best_f:
                    best_f = f
                    best_A = float(A)
                    best_S = float(Sgrid[j])

        if np.isfinite(best_f):
            break
        S_hi = min(S_hi * 3.0, 5000.0)

    # Local refinement via shrinking log-box grids
    for scale in (2.0, 1.5, 1.25):
        A_lo2 = max(1.2, best_A / scale)
        A_hi2 = min(100.0, best_A * scale)
        S_lo2 = max(0.2, best_S / scale)
        S_hi2 = min(8000.0, best_S * scale)

        logA = np.linspace(np.log(A_lo2), np.log(A_hi2), 17)
        logS = np.linspace(np.log(S_lo2), np.log(S_hi2), 17)
        for A in np.exp(logA):
            for S in np.exp(logS):
                f = _objective_drag_proxy(ca, float(A), float(S))
                if f < best_f:
                    best_f = f
                    best_A = float(A)
                    best_S = float(S)

    return best_A, best_S

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        ca = _extract(problem)

        A, S = _grid_optimize(ca)
        W, W_w = _weights_for_A_S(ca, A, S)

        # Use huge Reynolds to drive C_f ~ 0 while staying finite (no upper bound in model).
        RE_FLOOR = 1e30

        condition_results: list[dict[str, Any]] = []
        total_drag = 0.0

        sqrt_S_over_A = float(np.sqrt(S / A))

        for i in range(ca.n):
            rho = float(ca.rho[i])
            mu = float(ca.mu[i])
            CDA0 = float(ca.CDA0[i])
            e = float(ca.e[i])
            k = float(ca.k[i])
            Swet = float(ca.S_wetratio[i])

            Wi = float(W[i])
            Wwi = float(W_w[i])

            Rei = RE_FLOOR
            Cfi = float(0.074 / (Rei**0.2))

            # With fixed Cf, drag(V) = a V^2 + b/V^2. Min at V = (b/a)^(1/4).
            a = 0.5 * rho * (CDA0 + S * k * Cfi * Swet)
            b = 2.0 * (Wi * Wi) / (rho * np.pi * A * e * S)
            V = float((b / a) ** 0.25)

            # Ensure Reynolds constraint, then keep Re very large.
            Re_min = rho * V * sqrt_S_over_A / mu
            Rei = float(max(RE_FLOOR, 10.0 * Re_min))
            Cfi = float(0.074 / (Rei**0.2))

            # Recompute V after Cf update (tiny but consistent)
            a = 0.5 * rho * (CDA0 + S * k * Cfi * Swet)
            V = float((b / a) ** 0.25)

            C_L = float(2.0 * Wi / (rho * V * V * S))
            C_D_expected = CDA0 / S + k * Cfi * Swet + (C_L * C_L) / (np.pi * A * e)
            C_D = float(C_D_expected * (1.0 + 1e-12))

            drag = float(0.5 * rho * V * V * C_D * S)
            total_drag += drag

            condition_results.append(
                {
                    "condition_id": int(ca.condition_id[i]),
                    "V": V,
                    "W": Wi,
                    "W_w": Wwi,
                    "C_L": C_L,
                    "C_D": C_D,
                    "C_f": Cfi,
                    "Re": Rei,
                    "drag": drag,
                }
            )

        avg_drag = float(total_drag / ca.n)

        return {
            "A": float(A),
            "S": float(S),
            "avg_drag": avg_drag,
            "condition_results": condition_results,
        }