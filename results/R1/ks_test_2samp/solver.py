import numpy as np
from scipy.stats import ks_2samp

class Solver:
    def solve(self, problem, **kwargs):
        # Convert to numpy arrays for efficient processing
        sample1 = np.asarray(problem["sample1"])
        sample2 = np.asarray(problem["sample2"])
        
        # Use scipy's highly optimized KS test implementation
        result = ks_2samp(sample1, sample2, method='auto')
        
        return {"statistic": result.statistic, "pvalue": result.pvalue}