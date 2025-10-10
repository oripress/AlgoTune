from typing import Any
import numpy as np
import scipy.stats as stats

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two sample KS test to get statistic and pvalue."""
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]
        
        # Use scipy's implementation which is already optimized
        result = stats.ks_2samp(sample1, sample2, method="auto")
        
        return {"statistic": result.statistic, "pvalue": result.pvalue}