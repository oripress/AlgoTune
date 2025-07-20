import numpy as np
from scipy.signal import upfirdn

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the upfirdn operation for each problem defined as (h, x, up, down).
        """
        # Normalize to list of problems
        if not isinstance(problem, list):
            problem_list = [problem]
        else:
            problem_list = problem

        results = []
        for item in problem_list:
            # Unpack parameters
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                h, x, up, down = item[:4]
            else:
                # If only h, x provided, fetch up/down from kwargs
                try:
                    h, x = item
                    up = kwargs.get("up", 1)
                    down = kwargs.get("down", 1)
                except Exception:
                    raise ValueError("Invalid problem format")
            # Perform upsample-filter-downsample
            y = upfirdn(h, x, up=up, down=down)
            results.append(y)
        return results