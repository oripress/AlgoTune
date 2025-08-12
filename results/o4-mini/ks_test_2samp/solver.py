import numpy as np
import scipy.stats as stats
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Two-sample Kolmogorov-Smirnov test using SciPy's optimized routines.
        """
        # Extract samples
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]
        # Use SciPy's ks_2samp with automatic exact/asymptotic choice
        result = stats.ks_2samp(sample1, sample2, method="auto")
        return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}