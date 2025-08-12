from typing import Any, List

import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the upfirdn operation for each problem definition in the list.

        :param problem: A list of tuples (h, x, up, down). If entries are (h, x),
                        defaults to up=1, down=1.
        :return: A list of 1D arrays representing the upfirdn results.
        """
        results: List[np.ndarray] = []
        for item in problem:
            try:
                h, x, up, down = item
            except Exception:
                h, x = item
                up, down = 1, 1
            up_i = int(up)
            down_i = int(down)
            # Fast path for up=1 (np.convolve is very fast). Guard empties.
            if up_i == 1 and len(h) > 0 and len(x) > 0:
                conv = np.convolve(x, h, mode="full")
                res = conv if down_i == 1 else conv[::down_i]
            else:
                res = signal.upfirdn(h, x, up=up_i, down=down_i)
            results.append(res)
        return results