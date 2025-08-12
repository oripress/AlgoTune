from typing import Any, Dict, Union

import numpy as np
try:
    from ot import sinkhorn as _sinkhorn
except Exception as _exc:  # pragma: no cover
    _sinkhorn = None  # type: ignore[attr-defined]
    _sinkhorn_import_error = _exc  # type: ignore[attr-defined]
else:  # pragma: no cover
    _sinkhorn_import_error = None  # type: ignore[attr-defined]

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Union[np.ndarray, None, str]]:
        # Convert inputs
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)

        n, m = a.shape[0], b.shape[0]

        # Trivial fast paths
        if n == 1:
            return {"transport_plan": b.reshape(1, m)}
        if m == 1:
            return {"transport_plan": a.reshape(n, 1)}

        M_in = problem["cost_matrix"]

        # Early conversion (reused throughout)
        M_arr = np.asarray(M_in, dtype=np.float64)

        # Exact fast paths: constant matrix or additive separable matrix -> plan = a b^T
        if n >= 2 and m >= 2:
            try:
                v00 = M_arr[0, 0]
                v01 = M_arr[0, 1]
                v10 = M_arr[1, 0]

                # Quick constant-matrix precheck
                if (v01 == v00) and (v10 == v00):
                    if np.all(M_arr == v00):
                        return {"transport_plan": np.outer(a, b)}

                # Quick additive-separable precheck using a 2x2 minor
                v11 = M_arr[1, 1]
                if (v11 - v10 - v01 + v00) == 0.0:
                    # Full separability verification: M - M[:,0] - M[0,:] + M[0,0] == 0
                    B = M_arr - M_arr[:, [0]] - M_arr[[0], :] + v00
                    if not np.any(B):
                        return {"transport_plan": np.outer(a, b)}
            except Exception:
                # Fallback to standard path if any indexing/type issue occurs
                pass

        if _sinkhorn is None:  # pragma: no cover
            return {"transport_plan": None, "error_message": f"POT sinkhorn import failed: {_sinkhorn_import_error}"}

        reg = float(problem["reg"])

        # Zero-mass reduction if needed (boolean masks)
        if not (a.min() > 0.0 and b.min() > 0.0):
            mask_a = a > 0.0
            mask_b = b > 0.0
            M_sub = M_arr[np.ix_(mask_a, mask_b)]
            if not M_sub.flags.c_contiguous:
                M_sub = np.ascontiguousarray(M_sub)
            G_sub = _sinkhorn(a[mask_a], b[mask_b], M_sub, reg)
            G = np.zeros((n, m), dtype=np.float64)
            G[np.ix_(mask_a, mask_b)] = G_sub
            return {"transport_plan": G}

        # Common path
        if not M_arr.flags.c_contiguous:
            M_arr = np.ascontiguousarray(M_arr)
        G = _sinkhorn(a, b, M_arr, reg)
        return {"transport_plan": G}