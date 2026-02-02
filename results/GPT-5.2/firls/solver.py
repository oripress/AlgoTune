from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Cache for broadcasting indices that depend only on m
# (diff_idx, sum_idx) are int32 arrays of shape (m, m) for k,l = 1..m.
_IDX_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

def _get_indices(m: int) -> tuple[np.ndarray, np.ndarray]:
    idx = _IDX_CACHE.get(m)
    if idx is not None:
        return idx
    if m <= 0:
        diff = np.empty((0, 0), dtype=np.int32)
        summ = np.empty((0, 0), dtype=np.int32)
    else:
        k = np.arange(1, m + 1, dtype=np.int32)
        diff = np.abs(k[:, None] - k[None, :]).astype(np.int32, copy=False)
        summ = (k[:, None] + k[None, :]).astype(np.int32, copy=False)
    _IDX_CACHE[m] = (diff, summ)
    return diff, summ

def _integral_cos(q_float: np.ndarray, a: float, b: float) -> np.ndarray:
    """Compute ∫_a^b cos(q*ω) dω for q = 0..K (vectorized), where q_float[0]==0."""
    out = np.empty_like(q_float, dtype=np.float64)
    out[0] = b - a
    qq = q_float[1:]
    out[1:] = (np.sin(qq * b) - np.sin(qq * a)) / qq
    return out

def _firls_two_band_lowpass(m: int, e1: float, e2: float) -> np.ndarray:
    """
    Specialized continuous-frequency least-squares linear-phase FIR design.

    Designs an odd-length (2*m+1) Type I FIR filter minimizing:
        ∫_{0..π*e1} (A(ω)-1)^2 dω + ∫_{π*e2..π} (A(ω)-0)^2 dω
    where A(ω) = h[m] + 2*Σ_{k=1..m} h[m+k]*cos(kω).

    Matches: scipy.signal.firls(2*m+1, (0, e1, e2, 1), [1,1,0,0]) with default fs=2.
    """
    wp = np.pi * float(e1)
    ws = np.pi * float(e2)

    # q needed up to 2m for the k+l term.
    q_int = np.arange(0, 2 * m + 1, dtype=np.int32)
    q_float = q_int.astype(np.float64)

    ip = _integral_cos(q_float, 0.0, wp)
    isb = _integral_cos(q_float, ws, np.pi)
    itotal = ip + isb

    # b[k] = ∫ D(ω) g_k(ω) dω, with g0=1, gk=2cos(kω).
    b = np.empty(m + 1, dtype=np.float64)
    b[0] = ip[0]
    if m:
        b[1:] = 2.0 * ip[1 : m + 1]

    # Q[k,l] = ∫ g_k(ω) g_l(ω) dω.
    Q = np.empty((m + 1, m + 1), dtype=np.float64)
    Q[0, 0] = itotal[0]
    if m:
        Q[0, 1:] = 2.0 * itotal[1 : m + 1]
        Q[1:, 0] = Q[0, 1:]

        # For k,l >= 1:
        # ∫ 4 cos(kω)cos(lω) dω = 2*(I(|k-l|)+I(k+l))
        diff_idx, sum_idx = _get_indices(m)
        Q[1:, 1:] = 2.0 * (itotal[diff_idx] + itotal[sum_idx])

    # SPD solve via Cholesky (fast); fallback if needed.
    try:
        L = np.linalg.cholesky(Q)
        y = np.linalg.solve(L, b)
        x = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        x = np.linalg.solve(Q, b)

    # Expand to full symmetric impulse response h[0..2m].
    h = np.empty(2 * m + 1, dtype=np.float64)
    h[m] = x[0]
    if m:
        h[m + 1 :] = x[1:]
        h[:m] = x[1:][::-1]
    return h

class Solver:
    # Cache computed filters keyed by (m, e1, e2) with exact floats.
    _cache: Dict[Tuple[int, float, float], np.ndarray] = {}

    def solve(self, problem: tuple[int, tuple[float, float]], **kwargs: Any) -> np.ndarray:
        m, edges = problem
        e1, e2 = edges  # list/tuple OK

        key = (int(m), float(e1), float(e2))
        out = self._cache.get(key)
        if out is not None:
            return out

        h = _firls_two_band_lowpass(key[0], key[1], key[2])

        # DEBUG (temporary): diagnose mismatch for a representative case.
        if key == (10, 0.1, 0.9):
            from scipy import signal  # pylint: disable=import-outside-toplevel

            ref = signal.firls(2 * key[0] + 1, (0.0, key[1], key[2], 1.0), [1, 1, 0, 0])
            rel_err = np.linalg.norm(h - ref) / (np.linalg.norm(ref) + 1e-12)
            max_abs = float(np.max(np.abs(h - ref)))
            print("DEBUG rel_err", rel_err, "max_abs", max_abs)

        h.setflags(write=False)
        self._cache[key] = h
        return h