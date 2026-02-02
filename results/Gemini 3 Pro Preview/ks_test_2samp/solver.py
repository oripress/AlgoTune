from typing import Any
import numpy as np
import scipy.stats as stats

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Peform two sample ks test to get statistic and pvalue."""
        n1 = len(problem["sample1"])
        n2 = len(problem["sample2"])
        if n1 * n2 > 10000:
            result = stats.ks_2samp(problem["sample1"], problem["sample2"], method="asymp")
        else:
            result = stats.ks_2samp(problem["sample1"], problem["sample2"], method="auto")
        return {"statistic": result.statistic, "pvalue": result.pvalue}