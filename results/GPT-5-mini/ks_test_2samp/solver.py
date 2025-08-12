from typing import Any, Dict
import numpy as np
from scipy import stats

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform two-sample Kolmogorov-Smirnov test (two-sided) and return statistic and pvalue.

        Uses scipy.stats.ks_2samp with method="auto" to match the reference behavior.
        """
        sample1 = np.asarray(problem["sample1"], dtype=np.float64)
        sample2 = np.asarray(problem["sample2"], dtype=np.float64)
        result = stats.ks_2samp(sample1, sample2, method="auto")
        return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}