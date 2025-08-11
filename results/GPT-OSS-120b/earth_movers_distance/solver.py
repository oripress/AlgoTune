import json
import numpy as np

# Try to import the POT library (Python Optimal Transport)
try:
    import ot
    _OT_AVAILABLE = True
    _emd = ot.lp.emd
except Exception:
    _OT_AVAILABLE = False
    _emd = None

class Solver:
    def solve(self, problem: dict, **kwargs):
        """
        Compute the optimal transport plan (Earth Mover's Distance) for the given problem.

        Parameters
        ----------
        problem : dict or JSON string
            Dictionary (or JSON string) with keys:
                - "source_weights": list or array of source distribution weights (sums to 1)
                - "target_weights": list or array of target distribution weights (sums to 1)
                - "cost_matrix": 2‑D list/array of transport costs (shape n × m)

        Returns
        -------
        dict
            {"transport_plan": G}
            where G is a NumPy array of shape (n, m) giving the optimal coupling.
        """
        # Accept JSON string as input
        if isinstance(problem, str):
            try:
                problem = json.loads(problem)
                if not isinstance(problem, dict):
                    raise ValueError("Parsed JSON is not a dictionary.")
                # Ensure required keys exist
                if not all(k in problem for k in ("source_weights", "target_weights", "cost_matrix")):
                    raise KeyError("Missing required keys in problem dictionary.")
            except Exception as e:
                raise ValueError(f"Failed to parse problem JSON: {e}")

        # Convert inputs to NumPy arrays (float64)
        a = np.ascontiguousarray(problem["source_weights"], dtype=np.float64)
        b = np.ascontiguousarray(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)

        if not _OT_AVAILABLE:
            raise RuntimeError("POT library is not available for EMD computation.")

        # Ensure cost matrix is C‑contiguous float64 as required by ot.lp.emd
        # Compute optimal transport plan using POT (M is already contiguous)
        G = _emd(a, b, M, check_marginals=False)
        return {"transport_plan": G}