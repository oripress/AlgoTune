import numpy as np
from typing import Any
import scipy.stats as stats
from numba import jit

@jit(nopython=True)
def compute_ks_statistic(sample1, sample2, n1):
    # Create combined array with sample indicators
    combined = np.concatenate([sample1, sample2])
    indicators = np.concatenate([np.ones(n1), -np.ones(len(sample2))])
    
    # Sort combined array and indicators together
    sort_order = np.argsort(combined)
    indicators_sorted = indicators[sort_order]
    
    # Compute cumulative sums to get ECDF differences
    cumsum = np.cumsum(indicators_sorted)
    statistic = np.max(np.abs(cumsum)) / n1
    
    return statistic
class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Perform two sample KS test to get statistic and pvalue."""
        # Convert to numpy arrays and sort in-place
        sample1 = np.array(problem["sample1"])
        sample2 = np.array(problem["sample2"])
        sample1.sort()
        sample2.sort()
        
        n1 = len(sample1)
        n2 = len(sample2)
        
        # Compute the KS statistic using the optimized approach
        # Create combined array with sample indicators
        combined = np.concatenate([sample1, sample2])
        indicators = np.concatenate([np.ones(n1), -np.ones(n2)])
        
        # Sort combined array and indicators together
        sort_order = np.argsort(combined)
        indicators_sorted = indicators[sort_order]
        
        # Compute cumulative sums to get ECDF differences
        cumsum = np.cumsum(indicators_sorted)
        statistic = np.max(np.abs(cumsum)) / n1
        
        # Compute p-value using scipy distributions directly
        en = n1 * n2 / (n1 + n2)
        pvalue = stats.kstwo.sf(statistic, np.round(en))
        
        return {"statistic": float(statistic), "pvalue": float(pvalue)}