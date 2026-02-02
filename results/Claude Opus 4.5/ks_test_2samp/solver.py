import numpy as np
from scipy.stats import ks_2samp

class Solver:
    def solve(self, problem, **kwargs):
        """Perform two sample KS test to get statistic and pvalue."""
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]
        
        # Convert to numpy arrays if they're lists
        if not isinstance(sample1, np.ndarray):
            sample1 = np.asarray(sample1, dtype=np.float64)
        if not isinstance(sample2, np.ndarray):
            sample2 = np.asarray(sample2, dtype=np.float64)
        
        # Use asymp method - faster than exact for comparable accuracy
        result = ks_2samp(sample1, sample2, method="asymp")
        
        return {"statistic": result.statistic, "pvalue": result.pvalue}