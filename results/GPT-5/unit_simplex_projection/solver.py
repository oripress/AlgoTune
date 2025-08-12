from typing import Any, Dict

import numpy as np

# Optional: use numba for a very fast in-place selection algorithm
try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

# Workspace caches to reduce allocations across calls
_INV_J: np.ndarray | None = None
_INV_J_SIZE: int = 0
_BUF: np.ndarray | None = None
_BUF_SIZE: int = 0
_MASK: np.ndarray | None = None
_MASK_SIZE: int = 0

# Threshold for switching to selection-based O(n) expected algorithm
# Larger threshold favors highly optimized numpy sort for small/medium n.
_SELECT_THRESHOLD = 2048

def _ensure_capacity(n: int) -> tuple[np.ndarray, np.ndarray]:
    global _INV_J, _INV_J_SIZE, _BUF, _BUF_SIZE
    if _INV_J_SIZE < n:
        _INV_J = 1.0 / np.arange(1.0, n + 1.0)
        _INV_J_SIZE = n
    if _BUF_SIZE < n:
        _BUF = np.empty(n, dtype=np.float64)
        _BUF_SIZE = n
    # Type checkers: these will be set by now
    return _INV_J[:n], _BUF[:n]  # type: ignore[index]

def _ensure_select_capacity(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure buffers for selection-based algorithm: a float buffer and a boolean mask.
    """
    global _BUF, _BUF_SIZE, _MASK, _MASK_SIZE
    if _BUF_SIZE < n:
        _BUF = np.empty(n, dtype=np.float64)
        _BUF_SIZE = n
    if _MASK_SIZE < n:
        _MASK = np.empty(n, dtype=bool)
        _MASK_SIZE = n
    return _BUF[:n], _MASK[:n]  # type: ignore[index]

def _theta_select(y: np.ndarray) -> float:
    """
    Compute theta such that sum(max(y - theta, 0)) = 1 using a selection-based
    algorithm in expected O(n) time. Does not modify input y.
    """
    arr = y.copy()
    Tv = 0.0  # accumulated sum of elements definitely in the active set
    Kv = 0    # accumulated count of elements definitely in the active set

    # Loop until all elements are processed
    while arr.size > 0:
        m = arr.size
        buf, mask = _ensure_select_capacity(m)

        # Choose pivot as median (uses partition under the hood)
        m_idx = m // 2
        arr.partition(m_idx)
        p = float(arr[m_idx])

        # buf = arr - p
        np.subtract(arr, p, out=buf[:m])

        # Compute K_gt and S = sum(max(arr - p, 0)) without creating temporaries
        np.greater(buf[:m], 0.0, out=mask[:m])
        K_gt = int(mask[:m].sum())
        # Zero-out negatives in-place, then sum positives
        np.maximum(buf[:m], 0.0, out=buf[:m])
        S = float(buf[:m].sum())

        # Evaluate phi(p) = Tv - Kv*p - 1 + S
        phi = (Tv - Kv * p - 1.0) + S

        if phi > 0.0:
            # Root is above p: discard elements <= p (keep strictly greater)
            arr = arr[mask[:m]]
            continue

        # Root is at or below p: elements > p are definitely active
        # T_gt = S + K_gt * p
        Tv += S + K_gt * p
        Kv += K_gt

        if phi == 0.0:
            # Root exactly at p
            return p

        # Root is strictly below p: elements equal to p are also active
        # Count elements strictly less than p to derive count of equals
        np.less(arr, p, out=mask[:m])
        K_lt = int(mask[:m].sum())
        eq_count = m - K_gt - K_lt
        if eq_count:
            Tv += p * eq_count
            Kv += eq_count

        # Continue with elements strictly less than p
        arr = arr[mask[:m]]

    # Final threshold from accumulated active set
    theta = (Tv - 1.0) / Kv
    return float(theta)

if _NUMBA_AVAILABLE:
    # Numba-accelerated in-place three-way partition selection for theta
    @njit(cache=True, fastmath=True)
    def _theta_select_numba(y: np.ndarray) -> float:
        n = y.size
        # Work on a copy to avoid modifying input
        arr = y.copy()
        low = 0
        hi = n
        Tv = 0.0
        Kv = 0

        while hi - low > 0:
            m = hi - low
            mid = low + (m // 2)
            p = arr[mid]

            # Three-way partition: move >p to the front, ==p to middle, <p to end
            left = low
            right = hi - 1
            i = left

            S = 0.0
            K_gt = 0
            eq_count = 0

            while i <= right:
                v = arr[i]
                if v > p:
                    S += v - p
                    K_gt += 1
                    # swap arr[i] with arr[left]
                    tmp = arr[left]
                    arr[left] = v
                    arr[i] = tmp
                    left += 1
                    i += 1
                elif v < p:
                    # swap arr[i] with arr[right]
                    tmp = arr[right]
                    arr[right] = v
                    arr[i] = tmp
                    right -= 1
                    # i not incremented to process swapped-in element
                else:
                    eq_count += 1
                    i += 1

            # Evaluate phi(p)
            phi = (Tv - Kv * p - 1.0) + S

            if phi > 0.0:
                # Keep > p region [low, left)
                hi = left
            elif phi < 0.0:
                # >p and =p are active; accumulate and continue with <p region
                Tv += S + (K_gt + eq_count) * p
                Kv += K_gt + eq_count
                # Keep < p region [right+1, hi)
                low = right + 1
            else:
                return p

        # Final threshold from accumulated active set
        return (Tv - 1.0) / Kv
else:
    _theta_select_numba = None  # type: ignore[assignment]

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, np.ndarray]:
        """
        Euclidean projection of y onto the probability simplex:
            minimize (1/2) ||x - y||^2
            subject to sum(x) = 1, x >= 0

        Uses either:
        - A Numba-accelerated in-place selection algorithm (for large n), or
        - An expected O(n) NumPy selection algorithm, or
        - An O(n log n) sorting-based algorithm with micro-optimizations.

        :param problem: dict with key "y": list/array-like of shape (n,)
        :return: dict with key "solution": np.ndarray of shape (n,)
        """
        y = np.asarray(problem.get("y", []), dtype=np.float64).ravel()
        n = y.size

        if n == 0:
            return {"solution": y.copy()}
        if n == 1:
            return {"solution": np.array([1.0], dtype=np.float64)}

        # Fast path: if y is already nonnegative and sums to <= 1
        # - If sum == 1: y is already feasible.
        # - If sum < 1: the projection is y shifted uniformly.
        s = y.sum()
        y_min = y.min()
        if y_min >= 0.0:
            if s == 1.0:
                return {"solution": y.copy()}
            if s < 1.0:
                x = y + (1.0 - s) / n
                return {"solution": x}

        # Choose algorithm based on problem size
        if n >= _SELECT_THRESHOLD:
            if _NUMBA_AVAILABLE and _theta_select_numba is not None:
                theta = float(_theta_select_numba(y))
            else:
                theta = _theta_select(y)
            x = y - theta
            np.maximum(x, 0.0, out=x)
            return {"solution": x}

        # Sorting-based path for smaller n
        inv_j, buf = _ensure_capacity(n)

        # Sort y in ascending order on a copy, then create a descending view
        u = y.copy()
        u.sort()  # ascending
        v = u[::-1]  # descending view

        # Compute cumulative sums minus 1 into buffer, then divide elementwise by indices via cached reciprocals
        np.cumsum(v, out=buf)
        buf -= 1.0
        np.multiply(buf, inv_j, out=buf)  # buf now holds thresholds t_k

        # Find rho: largest index where v_i > t_i
        rho = np.count_nonzero(v > buf) - 1

        # Threshold
        theta = buf[rho]

        # Projection
        x = y - theta
        np.maximum(x, 0.0, out=x)

        return {"solution": x}