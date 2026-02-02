from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    __slots__ = ("_abs_buf", "_out_buf")

    def __init__(self) -> None:
        # Reusable buffers to reduce per-call allocations.
        self._abs_buf = np.empty(0, dtype=np.float64)
        self._out_buf = np.empty(0, dtype=np.float64)

    def solve(self, problem: dict[str, Any] | str, **kwargs: Any) -> dict[str, Any]:
        # Allow profiling tools that may pass a raw string.
        if not isinstance(problem, dict):
            try:
                import json

                problem = json.loads(problem)
            except Exception:
                import ast

                problem = ast.literal_eval(problem)

        v_in = problem["v"]
        v = v_in if isinstance(v_in, np.ndarray) else np.asarray(v_in)
        if v.ndim != 1:
            v = v.ravel()

        k = int(problem["k"])
        n = v.size

        if k <= 0:
            # Return fresh array (no shared buffer) for safety.
            return {"solution": np.zeros(n, dtype=v.dtype)}
        if k >= n:
            return {"solution": v}

        abs_buf = self._abs_buf
        if abs_buf.size < n or abs_buf.dtype != v.dtype:
            abs_buf = np.empty(n, dtype=v.dtype)
            self._abs_buf = abs_buf
        abs_v = abs_buf[:n]
        np.abs(v, out=abs_v)

        idx0 = n - k
        part = np.argpartition(abs_v, (idx0 - 1, idx0))
        t = abs_v[part[idx0]]

        out_buf = self._out_buf
        if out_buf.size < n or out_buf.dtype != v.dtype:
            out_buf = np.empty(n, dtype=v.dtype)
            self._out_buf = out_buf
        out = out_buf[:n]

        # Fast path: no tie crossing the boundary.
        if abs_v[part[idx0 - 1]] < t:
            # Choose the cheaper write pattern depending on sparsity.
            if k <= idx0:
                out.fill(0)
                keep_idx = part[idx0:]
                out[keep_idx] = v[keep_idx]
            else:
                out[:] = v
                drop_idx = part[:idx0]
                out[drop_idx] = 0
            return {"solution": out}

        # Tie case: stable behavior => among abs == t keep largest indices.
        top = part[idx0:]
        gt_mask = abs_v[top] > t
        gt_idx = top[gt_mask]
        rem = k - gt_idx.size

        out.fill(0)
        if gt_idx.size:
            out[gt_idx] = v[gt_idx]

        if rem:
            eq_idx = np.flatnonzero(abs_v == t)  # sorted ascending indices
            if eq_idx.size > rem:
                eq_idx = eq_idx[-rem:]
            out[eq_idx] = v[eq_idx]

        return {"solution": out}