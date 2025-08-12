import numpy as np  
from l0prune import l0prune  

class Solver:  
    def solve(self, problem):  
        """  
        Project v onto the L0-ball of radius k:  
            min_w ||v - w||^2   s.t.   ||w||_0 <= k  
        Uses a C++ quickselect (nth_element) under the hood.  
        """  
        # Load inputs  
        v = np.asarray(problem["v"], dtype=np.double)  
        k = int(problem["k"])  
        n = v.size  

        # Trivial cases  
        if k <= 0:  
            sol = np.zeros(n, dtype=np.double)  
        elif k >= n:  
            sol = v.copy()  
        else:  
            sol = l0prune(v, k)  

        return {"solution": sol.tolist()}