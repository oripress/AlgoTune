from typing import Any, Dict

import numpy as np
from scipy import stats

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        # Convert inputs to 1D float64 numpy arrays without unnecessary copies
        s1 = np.asarray(problem["sample1"], dtype=np.float64)
        s2 = np.asarray(problem["sample2"], dtype=np.float64)

        # Use SciPy's optimized KS two-sample implementation with auto method
        res = stats.ks_2samp(s1, s2, method="auto")

        # Ensure plain Python floats for compatibility
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}