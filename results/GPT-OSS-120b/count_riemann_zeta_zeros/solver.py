from typing import Any
from mpmath import mp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Count the zeros of the Riemann Zeta function in the critical strip
        (0 < Re(z) < 1) with imaginary part <= t.

        Parameters
        ----------
        problem : dict
            Dictionary containing the key "t" with a float value.

        Returns
        -------
        dict
            Dictionary with key "result" containing the integer count of zeros.
        """
        t = problem["t"]
        if t <= 0:
            return {"result": 0}
        # Use modest precision; mp.nzeros is efficient for the required range.
        mp.dps = 15
        return {"result": int(mp.nzeros(t))}