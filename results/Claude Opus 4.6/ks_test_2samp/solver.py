import numpy as np
from scipy.stats import ks_2samp

class Solver:
    def solve(self, problem, **kwargs):
        sample1 = np.asarray(problem["sample1"], dtype=np.float64)
        sample2 = np.asarray(problem["sample2"], dtype=np.float64)
        result = ks_2samp(sample1, sample2, method="auto")
        return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}