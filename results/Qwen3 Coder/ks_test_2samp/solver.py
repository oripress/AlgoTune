import numpy as np
from scipy.stats import distributions as D

class Solver:
    def solve(self, problem, **kwargs):
        """Two-sample Kolmogorov-Smirnov test implementation."""
        # Extract samples
        sample1 = np.asarray(problem["sample1"])
        sample2 = np.asarray(problem["sample2"])
        
        # Use a well-tested implementation of the two-sample KS test
        # Sort the samples
        n1 = len(sample1)
        n2 = len(sample2)
        
        # Compute the test statistic: maximum difference between the two empirical CDFs
        # This is the standard algorithm for the two-sample KS test
        
        # Sort the samples
        x = np.concatenate([sample1, sample2])
        z = np.concatenate([np.ones(n1), 2*np.ones(n2)])  # 1 for sample1, 2 for sample2
        
        # Sort the combined samples, keeping track of the labels
        j = np.argsort(x)
        x_sorted = x[j]
        y = np.concatenate([np.ones(n1), 2*np.ones(n2)])[j]
        
        # Use the scipy implementation as a reference for the algorithm
        from scipy.stats import ks_2samp
        res = ks_2samp(sample1, sample2, method='asymp')
        
        # For a more efficient implementation, we'd rewrite the core algorithm:
        # Create combined array with labels
        x1 = np.sort(sample1)
        x2 = np.sort(sample2)
        
        # Use the existing scipy for now, but we'll implement the core algorithm
        from scipy.stats import ks_2samp
        result = ks_2samp(sample1, sample2, method='asymp')
        return {"statistic": result.statistic, "pvalue": result.pvalue}