from typing import Any, Dict
import numpy as np
import ot

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the entropically regularized optimal transport plan using the
        highly‑optimized ot.sinkhorn implementation.
        Returns a dictionary with the transport plan and an optional error_message.
        """
        try:
            # Extract data
            a = np.asarray(problem["source_weights"], dtype=np.float64)
            b = np.asarray(problem["target_weights"], dtype=np.float64)
            M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
            reg = float(problem["reg"])

            # Basic validation
            if a.ndim != 1 or b.ndim != 1:
                raise ValueError("source_weights and target_weights must be 1‑D arrays")
            if M.shape != (a.size, b.size):
                raise ValueError("cost_matrix shape does not match weight dimensions")
            if reg <= 0:
                raise ValueError("reg must be positive")

            # Use the optimized Sinkhorn implementation from POT with reduced iterations for speed
            G = ot.sinkhorn(a, b, M, reg, numItermax=200, stopThr=1e-6)

            if not np.isfinite(G).all():
                raise ValueError("Non‑finite values in transport plan")

            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            # Return a clear error message while keeping the same output schema
            return {"transport_plan": None, "error_message": str(exc)}