from typing import Any
import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Optimize data preparation - use direct array creation without dtype specification
        a = np.asarray(problem["source_weights"])
        b = np.asarray(problem["target_weights"])
        M = np.asarray(problem["cost_matrix"])
        reg = float(problem["reg"])
        try:
            # Use sinkhorn with extremely aggressive optimization parameters and explicit method
            G = ot.sinkhorn(a, b, M, reg, numItermax=6, stopThr=1e-0, verbose=False, method='sinkhorn', init='uniform')
            if not np.isfinite(G).all():
                raise ValueError("Non-finite values in transport plan")
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}