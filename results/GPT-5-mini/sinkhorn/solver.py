from typing import Any, Dict, Optional
import numpy as np

# Import POT (ot) at module import time to minimize per-call overhead.
try:
    import ot as _ot  # type: ignore
    _sinkhorn = _ot.sinkhorn
except Exception:
    _ot = None
    _sinkhorn = None

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the entropic-regularized OT plan. Primary path uses POT's
        ot.sinkhorn (if available) to match the reference exactly and quickly.
        Falls back to a stable log-domain NumPy implementation otherwise.
        """
        try:
            a = np.asarray(problem["source_weights"], dtype=np.float64)
            b = np.asarray(problem["target_weights"], dtype=np.float64)
            M = np.asarray(problem["cost_matrix"], dtype=np.float64)
            # ensure C-contiguity for C-backed libraries
            if not M.flags["C_CONTIGUOUS"]:
                M = np.ascontiguousarray(M)
            reg = float(problem["reg"])
        except Exception as exc:
            return {"transport_plan": None, "error_message": f"Invalid input: {exc}"}

        # Basic validation
        if a.ndim != 1 or b.ndim != 1:
            return {"transport_plan": None, "error_message": "source_weights and target_weights must be 1-D"}
        n, m = a.size, b.size
        if M.shape != (n, m):
            return {"transport_plan": None, "error_message": "cost_matrix shape does not match weights"}
        if not (reg > 0):
            return {"transport_plan": None, "error_message": "reg must be positive"}

        # Fast path: use POT's sinkhorn if available (matches reference).
        if _sinkhorn is not None:
            try:
                G = _sinkhorn(a, b, M, reg)
                if not isinstance(G, np.ndarray):
                    G = np.asarray(G, dtype=np.float64)
                if not np.isfinite(G).all():
                    return {"transport_plan": None, "error_message": "Non-finite values in transport plan"}
                return {"transport_plan": G, "error_message": None}
            except Exception:
                # If POT fails for some reason, fall back to stable implementation.
                pass

        # Fallback: stable log-domain Sinkhorn
        try:
            numItermax = int(kwargs.get("numItermax", 10000))
            stopThr = float(kwargs.get("stopThr", 1e-9))
            G = self._sinkhorn_log(a, b, M, reg, numItermax=numItermax, stopThr=stopThr)
            if not np.isfinite(G).all():
                return {"transport_plan": None, "error_message": "Fallback produced non-finite transport plan"}
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}

    def _sinkhorn_log(
        self,
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: float,
        numItermax: int = 10000,
        stopThr: float = 1e-9,
    ) -> np.ndarray:
        """
        Stable log-domain Sinkhorn using scipy.special.logsumexp.
        Vectorized, numerically stable implementation.
        """
        from scipy.special import logsumexp

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = a.size
        m = b.size

        logK = -M / reg
        loga = np.where(a > 0.0, np.log(a), -np.inf)
        logb = np.where(b > 0.0, np.log(b), -np.inf)
        logu = np.zeros(n, dtype=np.float64)
        logv = np.zeros(m, dtype=np.float64)

        for _ in range(numItermax):
            S = logsumexp(logK + logv[None, :], axis=1)
            new_logu = loga - S
            T = logsumexp(logK + new_logu[:, None], axis=0)
            new_logv = logb - T

            delta = max(np.max(np.abs(new_logu - logu)), np.max(np.abs(new_logv - logv)))
            logu = new_logu
            logv = new_logv
            if delta < stopThr:
                break

        logP = logu[:, None] + logK + logv[None, :]
        P = np.exp(logP)
        np.maximum(P, 0.0, out=P)

        # Small marginal corrections to counteract numerical drift
        row_sums = P.sum(axis=1)
        mask = row_sums > 0
        if np.any(mask):
            row_scale = np.ones_like(row_sums)
            row_scale[mask] = a[mask] / row_sums[mask]
            P = (P.T * row_scale).T

        col_sums = P.sum(axis=0)
        maskc = col_sums > 0
        if np.any(maskc):
            col_scale = np.ones_like(col_sums)
            col_scale[maskc] = b[maskc] / col_sums[maskc]
            P = P * col_scale

        return P