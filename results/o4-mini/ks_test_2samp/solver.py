# Kolmogorov-Smirnov two-sample test solver
from typing import Any
import scipy.stats as stats

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Perform two-sample Kolmogorov-Smirnov test and return statistic and p-value."""
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]
        res = stats.ks_2samp(sample1, sample2, method="auto")
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}