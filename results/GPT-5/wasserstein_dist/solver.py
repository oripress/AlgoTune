from typing import Any, Dict, List
import numpy as np
import math

try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _wd_full(u: np.ndarray, v: np.ndarray) -> float:
        n = u.shape[0]
        if n <= 1:
            return 0.0
        su = 0.0
        sv = 0.0
        neg = False
        # First pass: sums and negativity
        for i in range(n):
            ui = u[i]
            vi = v[i]
            if ui < 0.0 or vi < 0.0:
                neg = True
            su += ui
            sv += vi
        # Validate like scipy (non-negative, positive finite sums)
        if not np.isfinite(su) or not np.isfinite(sv) or su <= 0.0 or sv <= 0.0 or neg:
            return -1.0

        # Second pass: accumulate difference of normalized cdfs
        cum = 0.0
        total = 0.0
        if su == sv:
            inv = 1.0 / su
            for i in range(n - 1):
                cum += u[i] - v[i]
                total += abs(cum)
            total *= inv
        else:
            inv_su = 1.0 / su
            inv_sv = 1.0 / sv
            for i in range(n - 1):
                cum += u[i] * inv_su - v[i] * inv_sv
                total += abs(cum)
        return total

    # Warm up JIT to avoid first-call compilation in solve
    _wd_full(np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64))
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

# Threshold below which pure-Python loops avoid NumPy/Numba overhead
_SMALL_N_PY = 2048

class Solver:
    def solve(self, problem: Dict[str, List[float]], **kwargs) -> Any:
        """
        Compute the 1D Wasserstein (Earth Mover's) distance between two discrete
        distributions supported on positions [1, 2, ..., n].

        Matches scipy.stats.wasserstein_distance for fixed support [1..n]:
        - Independently normalize u and v to sum to 1
        - Return sum_{i=1}^{n-1} |cumsum(u - v)[i]|
        - On any error, return float(len(u)) as fallback
        """
        try:
            if not isinstance(problem, dict) or "u" not in problem or "v" not in problem:
                raise ValueError("Problem must be a dict with 'u' and 'v'.")

            u_in = problem["u"]
            v_in = problem["v"]
            n = len(u_in)
            if n != len(v_in):
                raise ValueError("u and v must have the same length.")
            if n <= 1:
                return 0.0

            # Small-n pure-Python fast path to avoid array creation/JIT overhead
            if n <= _SMALL_N_PY:
                su = 0.0
                sv = 0.0
                neg = False
                for i in range(n):
                    ui = u_in[i]
                    vi = v_in[i]
                    if ui < 0.0 or vi < 0.0:
                        neg = True
                    su += ui
                    sv += vi
                if not (math.isfinite(su) and math.isfinite(sv)) or su <= 0.0 or sv <= 0.0 or neg:
                    raise ValueError("Invalid weights.")

                cum = 0.0
                total = 0.0
                if su == sv:
                    inv = 1.0 / su
                    for i in range(n - 1):
                        cum += u_in[i] - v_in[i]
                        total += abs(cum)
                    return float(total * inv)
                else:
                    inv_su = 1.0 / su
                    inv_sv = 1.0 / sv
                    for i in range(n - 1):
                        cum += u_in[i] * inv_su - v_in[i] * inv_sv
                        total += abs(cum)
                    return float(total)

            # Prefer Numba full-path when available
            if _NUMBA_OK:
                u = np.asarray(u_in, dtype=np.float64)
                v = np.asarray(v_in, dtype=np.float64)
                d = _wd_full(u, v)
                if d < 0.0 or not np.isfinite(d):
                    raise ValueError("Invalid weights or non-finite result.")
                return float(d)

            # NumPy fallback: validate, normalize, and accumulate
            u = np.array(u_in, dtype=np.float64, copy=True)
            v = np.array(v_in, dtype=np.float64, copy=True)

            if np.any(u < 0.0) or np.any(v < 0.0):
                raise ValueError("Weights must be non-negative.")

            su = float(u.sum(dtype=np.float64))
            sv = float(v.sum(dtype=np.float64))
            if not np.isfinite(su) or not np.isfinite(sv) or su <= 0.0 or sv <= 0.0:
                raise ValueError("Sum of weights must be positive and finite.")

            if su == sv:
                u -= v
                np.add.accumulate(u, out=u)
                np.abs(u[:-1], out=u[:-1])
                d = float(u[:-1].sum(dtype=np.float64) / su)
            else:
                u *= (1.0 / su)
                u -= v * (1.0 / sv)
                np.add.accumulate(u, out=u)
                np.abs(u[:-1], out=u[:-1])
                d = float(u[:-1].sum(dtype=np.float64))

            if not np.isfinite(d):
                raise ValueError("Non-finite result.")
            return d

        except Exception:
            try:
                return float(len(problem["u"]))
            except Exception:
                return 0.0