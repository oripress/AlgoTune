from typing import Any, Dict
import numpy as np
from scipy.stats import ks_2samp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the two‑sample Kolmogorov‑Smirnov statistic and its p‑value
        using SciPy's reference implementation (method="auto").
        """
        # Convert inputs to NumPy arrays (float64) for consistency
        x = np.asarray(problem["sample1"], dtype=np.float64)
        y = np.asarray(problem["sample2"], dtype=np.float64)

        # Guard against empty inputs (should not occur in valid tests)
        if x.size == 0 or y.size == 0:
            return {"statistic": 0.0, "pvalue": 1.0}

        # Use SciPy's ks_2samp with default "auto" method
        result = ks_2samp(x, y, method="auto")
        return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}