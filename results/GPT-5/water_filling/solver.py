from typing import Any, Dict
import numpy as np
import math

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f

        return wrap

@njit(cache=True, fastmath=True, nogil=True)
def _wf_prune_numba(alpha: np.ndarray, P_total: float):
    """
    Pruning-based water-level search without sorting.

    Returns:
      w: water level
      m: number of active channels (alpha_i < w)
      tail_log_sum: sum of log(alpha_i) for inactive channels (alpha_i >= w)
    """
    n = alpha.shape[0]
    # Initial water level using all channels
    sum_all = 0.0
    for i in range(n):
        sum_all += alpha[i]
    k = n
    w = (P_total + sum_all) / k

    # Iterate until the active set (alpha_i < w) stabilizes
    # Guard with a max iteration count to avoid infinite loops in pathological cases
    for _ in range(2 * n + 1):
        sum_act = 0.0
        cnt = 0
        for i in range(n):
            if alpha[i] < w:
                sum_act += alpha[i]
                cnt += 1
        if cnt == 0:
            # No active channels (should not happen for P_total > 0), indicate fallback
            return w, 0, 0.0
        w_new = (P_total + sum_act) / cnt
        if cnt == k:
            w = w_new
            k = cnt
            break
        w = w_new
        k = cnt

    # Compute tail log sum over inactive channels (alpha_i >= w)
    tail_log_sum = 0.0
    for i in range(n):
        if alpha[i] >= w:
            tail_log_sum += math.log(alpha[i])
    return w, k, tail_log_sum

@njit(cache=True, fastmath=True, nogil=True)
def _wf_prune_numba_full(alpha: np.ndarray, P_total: float, x_out: np.ndarray):
    """
    Pruning-based water-level search without sorting.
    Fills x_out with optimal allocations and returns (w, capacity).
    Returns (-1.0, 0.0) if it fails (e.g., zero active channels).
    """
    n = alpha.shape[0]
    # Initial water level using all channels
    sum_all = 0.0
    for i in range(n):
        sum_all += alpha[i]
    k = n
    w = (P_total + sum_all) / k

    # Iterate until the active set stabilizes
    for _ in range(2 * n + 1):
        sum_act = 0.0
        cnt = 0
        for i in range(n):
            if alpha[i] < w:
                sum_act += alpha[i]
                cnt += 1
        if cnt == 0:
            return -1.0, 0.0
        w_new = (P_total + sum_act) / cnt
        if cnt == k:
            w = w_new
            k = cnt
            break
        w = w_new
        k = cnt

    # Compute x_out and capacity in a single pass
    logw = math.log(w)
    cap = 0.0
    for i in range(n):
        ai = alpha[i]
        v = w - ai
        if v > 0.0:
            x_out[i] = v
            cap += logw
        else:
            x_out[i] = 0.0
            cap += math.log(ai)
    return w, cap

class Solver:
    def __init__(self) -> None:
        # Reusable work buffers (grown on demand)
        self._cap = 0
        self._a = np.empty(0, dtype=np.float64)  # sorted alpha buffer
        self._w = np.empty(0, dtype=np.float64)  # work buffer (cumsum, logs)
        self._den = np.empty(0, dtype=np.float64)  # [1, 2, ..., cap]
        self._cond = np.empty(0, dtype=np.bool_)  # condition buffer (length cap-1)
        self._x = np.empty(0, dtype=np.float64)  # output allocation buffer

        # Numba warmup (compilation in __init__ doesn't count towards runtime)
        self._use_numba = NUMBA_AVAILABLE
        if self._use_numba:
            try:
                _ = _wf_prune_numba(np.array([1.0, 2.0], dtype=np.float64), 1.0)
                dummy_x = np.empty(2, dtype=np.float64)
                _ = _wf_prune_numba_full(np.array([1.0, 2.0], dtype=np.float64), 1.0, dummy_x)
            except Exception:
                self._use_numba = False

    def _ensure_capacity(self, n: int) -> None:
        if n <= self._cap:
            return
        # Grow capacity (geometric growth to reduce reallocations)
        new_cap = max(2 * self._cap, n, 32)
        self._a = np.empty(new_cap, dtype=np.float64)
        self._w = np.empty(new_cap, dtype=np.float64)
        self._den = np.arange(1, new_cap + 1, dtype=np.float64)
        self._cond = np.empty(new_cap, dtype=np.bool_)
        self._x = np.empty(new_cap, dtype=np.float64)
        self._cap = new_cap

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        alpha = np.asarray(problem["alpha"], dtype=np.float64)
        P_total = float(problem["P_total"])
        n = alpha.size

        # Handle edge cases conservatively
        if n == 0:
            return {"x": [], "Capacity": 0.0}
        if not np.all(np.isfinite(alpha)) or not np.isfinite(P_total):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        if P_total < 0:
            # Infeasible with x >= 0; return NaNs
            return {"x": [float("nan")] * n, "Capacity": float("nan")}
        if P_total == 0:
            # No power allocated
            if np.any(alpha <= 0):
                return {"x": [float("nan")] * n, "Capacity": float("nan")}
            capacity = float(np.sum(np.log(alpha)))
            return {"x": [0.0] * n, "Capacity": capacity}
        if np.any(alpha <= 0):
            # Log undefined for non-positive alpha with x >= 0
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Ensure buffers
        self._ensure_capacity(n)

        # Prefer Numba pruning when available; fallback to NumPy if not
        if self._use_numba:
            x = self._x[:n]
            w_level, capacity = _wf_prune_numba_full(alpha, P_total, x)
            if w_level >= 0.0 and np.isfinite(w_level) and np.isfinite(capacity):
                return {"x": x.tolist(), "Capacity": float(capacity)}
            # If numba path failed (rare), fall back to NumPy path

        # --- Fast vectorized analytic water-filling (in-place, cached buffers) ---
        a = self._a[:n]
        a[...] = alpha
        a.sort()  # in-place sort

        # Compute w_all = (P_total + cumsum(a)) / den
        w = self._w  # reuse buffer
        w_view = w[:n]
        np.cumsum(a, out=w_view)
        w_view += P_total
        np.divide(w_view, self._den[:n], out=w_view)

        # Find smallest k such that w_k <= a_{k+1}; fallback to k = n-1
        if n > 1:
            cond = self._cond[: n - 1]
            np.less_equal(w_view[:-1], a[1:], out=cond)
            idx = int(np.argmax(cond))
            k_idx = idx if cond[idx] else n - 1
        else:
            k_idx = 0

        w_level = float(w_view[k_idx])

        # Optimal allocation x = max(0, w - alpha) computed in-place
        x = self._x[:n]
        np.subtract(w_level, alpha, out=x)
        np.maximum(x, 0.0, out=x)

        # Capacity: sum log(alpha + x) = m*log(w) + sum_{i>=m} log(a[i])
        m = k_idx + 1
        if m < n:
            # Use work buffer to store logs to avoid a new allocation
            logs = w[: n - m]
            np.log(a[m:], out=logs)
            tail_log_sum = float(np.sum(logs))
        else:
            tail_log_sum = 0.0
        capacity = float(m * math.log(w_level) + tail_log_sum)

        return {"x": x.tolist(), "Capacity": capacity}