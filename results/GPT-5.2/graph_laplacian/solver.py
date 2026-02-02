from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numba import njit

@njit(fastmath=True, cache=False)
def _laplacian_unnormed(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pass A: compute degrees + per-row diagonal position (relative) + insertion split + nnz counts
    deg = np.empty(n, dtype=np.float64)
    diagpos = np.empty(n, dtype=np.int32)  # -1 if absent, else relative position in row
    splitpos = np.empty(n, dtype=np.int32)  # first position with col >= row (in sorted rows)

    nnz_counts = np.empty(n, dtype=np.int32)

    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row_nnz = row_end - row_start

        s = 0.0
        dpos = -1
        spos = -1

        for kk in range(row_start, row_end):
            j = indices[kk]
            s += data[kk]
            rel = kk - row_start
            if spos == -1 and j >= i:
                spos = rel
            if j == i and dpos == -1:
                dpos = rel

        if spos == -1:
            spos = row_nnz

        deg[i] = s
        diagpos[i] = dpos
        splitpos[i] = spos

        # nnz count after inserting diagonal (if needed) and eliminating the singleton self-loop case
        if dpos != -1:
            # If the only entry is a self-loop, Laplacian diagonal becomes 0 => removed.
            nnz_counts[i] = 0 if row_nnz == 1 else row_nnz
        else:
            nnz_counts[i] = row_nnz + (1 if s != 0.0 else 0)

    # Build indptr via prefix sum
    out_indptr = np.empty(n + 1, dtype=np.int32)
    out_indptr[0] = 0
    total = 0
    for i in range(n):
        total += nnz_counts[i]
        out_indptr[i + 1] = total

    out_indices = np.empty(total, dtype=np.int32)
    out_data = np.empty(total, dtype=np.float64)

    # Pass B: fill
    pos = 0
    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row_nnz = row_end - row_start
        dpos = diagpos[i]
        s = deg[i]

        if dpos != -1:
            if row_nnz == 1:
                continue  # singleton self-loop -> eliminated
            for rel in range(row_nnz):
                j = indices[row_start + rel]
                val = -data[row_start + rel]
                if rel == dpos:
                    val += s
                out_indices[pos] = j
                out_data[pos] = val
                pos += 1
        else:
            if s == 0.0:
                # Rare deg==0 with explicit entries (e.g., canceling weights); mimic as best-effort.
                for rel in range(row_nnz):
                    out_indices[pos] = indices[row_start + rel]
                    out_data[pos] = -data[row_start + rel]
                    pos += 1
                continue

            spos = splitpos[i]

            # copy cols < i
            for rel in range(spos):
                out_indices[pos] = indices[row_start + rel]
                out_data[pos] = -data[row_start + rel]
                pos += 1

            # insert diagonal
            out_indices[pos] = i
            out_data[pos] = s
            pos += 1

            # copy cols > i
            for rel in range(spos, row_nnz):
                out_indices[pos] = indices[row_start + rel]
                out_data[pos] = -data[row_start + rel]
                pos += 1

    return out_data, out_indices, out_indptr

@njit(fastmath=True, cache=False)
def _laplacian_normed(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pass A: degrees + diag/split positions
    deg = np.empty(n, dtype=np.float64)
    diagpos = np.empty(n, dtype=np.int32)
    splitpos = np.empty(n, dtype=np.int32)
    row_nnz_arr = np.empty(n, dtype=np.int32)

    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row_nnz = row_end - row_start
        row_nnz_arr[i] = row_nnz

        s = 0.0
        dpos = -1
        spos = -1

        for kk in range(row_start, row_end):
            j = indices[kk]
            s += data[kk]
            rel = kk - row_start
            if spos == -1 and j >= i:
                spos = rel
            if j == i and dpos == -1:
                dpos = rel

        if spos == -1:
            spos = row_nnz

        deg[i] = s
        diagpos[i] = dpos
        splitpos[i] = spos

    invsqrt = np.empty(n, dtype=np.float64)
    for i in range(n):
        di = deg[i]
        invsqrt[i] = 1.0 / np.sqrt(di) if di > 0.0 else 0.0

    # nnz counts (no extra CSR scan)
    nnz_counts = np.empty(n, dtype=np.int32)
    for i in range(n):
        row_nnz = row_nnz_arr[i]
        dpos = diagpos[i]
        if dpos != -1:
            nnz_counts[i] = 0 if row_nnz == 1 else row_nnz
        else:
            nnz_counts[i] = row_nnz + (1 if deg[i] > 0.0 else 0)

    out_indptr = np.empty(n + 1, dtype=np.int32)
    out_indptr[0] = 0
    total = 0
    for i in range(n):
        total += nnz_counts[i]
        out_indptr[i + 1] = total

    out_indices = np.empty(total, dtype=np.int32)
    out_data = np.empty(total, dtype=np.float64)

    # Pass B: fill
    pos = 0
    for i in range(n):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row_nnz = row_end - row_start
        dpos = diagpos[i]
        is_i = invsqrt[i]

        if dpos != -1:
            if row_nnz == 1:
                continue  # singleton self-loop -> eliminated (diag becomes 0)
            for rel in range(row_nnz):
                j = indices[row_start + rel]
                val = -data[row_start + rel] * is_i * invsqrt[j]
                if rel == dpos:
                    val += 1.0
                out_indices[pos] = j
                out_data[pos] = val
                pos += 1
        else:
            if deg[i] <= 0.0:
                continue  # isolated node => empty row

            spos = splitpos[i]

            # copy cols < i
            for rel in range(spos):
                j = indices[row_start + rel]
                out_indices[pos] = j
                out_data[pos] = -data[row_start + rel] * is_i * invsqrt[j]
                pos += 1

            # insert diagonal (=1)
            out_indices[pos] = i
            out_data[pos] = 1.0
            pos += 1

            # copy cols > i
            for rel in range(spos, row_nnz):
                j = indices[row_start + rel]
                out_indices[pos] = j
                out_data[pos] = -data[row_start + rel] * is_i * invsqrt[j]
                pos += 1

    return out_data, out_indices, out_indptr

class Solver:
    def __init__(self) -> None:
        # Trigger JIT compilation outside the timed solve() path.
        data = np.array([1.0], dtype=np.float64)
        indices = np.array([0], dtype=np.int32)
        indptr = np.array([0, 1], dtype=np.int32)
        _laplacian_unnormed(data, indices, indptr, 1)
        _laplacian_normed(data, indices, indptr, 1)

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, dict[str, Any]]:
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            normed = bool(problem["normed"])
        except Exception:
            shape = tuple(problem.get("shape", (0, 0)))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": tuple(shape)}}

        try:
            n = int(shape[0])
            data_arr = np.asarray(data, dtype=np.float64)
            indices_arr = np.asarray(indices, dtype=np.int32)
            indptr_arr = np.asarray(indptr, dtype=np.int32)

            if normed:
                out_data, out_indices, out_indptr = _laplacian_normed(
                    data_arr, indices_arr, indptr_arr, n
                )
            else:
                out_data, out_indices, out_indptr = _laplacian_unnormed(
                    data_arr, indices_arr, indptr_arr, n
                )

            return {
                "laplacian": {
                    "data": out_data,
                    "indices": out_indices,
                    "indptr": out_indptr,
                    "shape": (n, n),
                }
            }
        except Exception:
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": tuple(shape)}}