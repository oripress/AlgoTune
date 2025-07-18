import numpy as np
from typing import Any
from scipy import stats

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two sample KS test to get statistic and pvalue."""
        sample1 = np.array(problem["sample1"])
        sample2 = np.array(problem["sample2"])
        
        # Use scipy's implementation to ensure correctness
        result = stats.ks_2samp(sample1, sample2, method="auto")
        
        return {"statistic": result.statistic, "pvalue": result.pvalue}