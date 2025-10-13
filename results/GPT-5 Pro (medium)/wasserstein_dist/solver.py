from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, List[float]], **kwargs) -> Any:
        """
        Compute the 1D Wasserstein-1 distance between two discrete distributions
        supported on positions [1, 2, ..., n], with weights given by u and v.

        This matches scipy.stats.wasserstein_distance with:
            x = y = [1, 2, ..., n], u_weights = u, v_weights = v

        Vectorized implementation:
            W1 = sum_{i=1}^{n-1} | sum_{j<=i} (u_j/sum_u - v_j/sum_v) |

        Fallback behavior mimics the reference implementation:
            - If inputs are invalid (negative weights, zero or non-finite sums,
              mismatched lengths), return float(n).
        """
        try:
            u = problem["u"]
            v = problem["v"]
            n = len(u)
            if n == 0 or len(v) != n:
                return float(n)

            # Convert to numpy arrays (float64) for vectorized ops
            u_arr = np.asarray(u, dtype=np.float64)
            v_arr = np.asarray(v, dtype=np.float64)
            if u_arr.ndim != 1 or v_arr.ndim != 1 or u_arr.shape[0] != n or v_arr.shape[0] != n:
                return float(n)

            # Check sums and validity: non-negative and finite sums > 0
            su = float(u_arr.sum())
            sv = float(v_arr.sum())
            if not (np.isfinite(su) and np.isfinite(sv)) or su <= 0.0 or sv <= 0.0:
                return float(n)
            if u_arr.min() < 0.0 or v_arr.min() < 0.0:
                return float(n)

            if n == 1:
                return 0.0

            # Compute Wasserstein-1 via cumulative differences of normalized weights
            diff = u_arr * (1.0 / su) - v_arr * (1.0 / sv)
            cdf_diff = np.cumsum(diff, dtype=np.float64)
            dist = float(np.abs(cdf_diff[:-1]).sum(dtype=np.float64))

            if not np.isfinite(dist):
                return float(n)

            return dist
        except Exception:
            # Match reference's fallback on unexpected errors
            try:
                return float(len(problem.get("u", [])))
            except Exception:
                return 0.0