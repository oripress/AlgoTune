from typing import Any, Dict
import numpy as np
import ot

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        a = np.array(problem["source_weights"], dtype=np.float64, copy=False)
        b = np.array(problem["target_weights"], dtype=np.float64, copy=False)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])

        # Use POT's sinkhorn directly to exactly match the reference output
        G = ot.sinkhorn(a, b, M, reg)
        # Validate finiteness as a quick sanity check
        if not np.isfinite(G).all():
            raise ValueError("Non-finite values in transport plan")
        return {"transport_plan": G}