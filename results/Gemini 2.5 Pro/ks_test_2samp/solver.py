from typing import Any
from scipy import stats

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Calculates the two-sample Kolmogorov-Smirnov statistic and p-value.
        
        This implementation uses scipy.stats.ks_2samp but specifies
        mode='asymp' to force the use of the fast asymptotic formula for
        the p-value. This is necessary for the older scipy version in the
        test environment, where the default 'auto' mode can select a slow
        'exact' calculation, causing timeouts. The 'method' keyword is not
        available in this version.
        """
        s1 = problem['sample1']
        s2 = problem['sample2']
        
        # For older scipy versions, the keyword is 'mode' and the value is 'asymp'.
        # This avoids the slow 'exact' calculation that causes timeouts.
        result = stats.ks_2samp(s1, s2, mode='asymp')
        
        return {"statistic": result.statistic, "pvalue": result.pvalue}