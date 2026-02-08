import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        n = len(y)
        
        if n < 3:
            return np.array([], dtype=np.float64)
        
        result = np.empty(n - 1, dtype=np.float64)
        
        h3 = dx / 3.0
        h12 = dx / 12.0
        
        # Simpson's pairs: integral over (y[2k], y[2k+1], y[2k+2])
        # These fill odd result indices: result[1], result[3], ...
        pairs = h3 * (y[:-2:2] + 4.0 * y[1:-1:2] + y[2::2])
        cs = np.cumsum(pairs)
        result[1::2] = cs
        
        # result[0]: backward correction only
        # result[0] = result[1] - h/12*(-y[0] + 8*y[1] + 5*y[2])
        result[0] = result[1] - h12 * (-y[0] + 8.0 * y[1] + 5.0 * y[2])
        
        # Even indices >= 2
        if n - 1 > 2:
            even_idx = np.arange(2, n - 1, 2)
            if len(even_idx) > 0:
                # Forward: result[idx-1] + h/12*(-y[idx-1] + 8*y[idx] + 5*y[idx+1])
                fwd = result[even_idx - 1] + h12 * (-y[even_idx - 1] + 8.0 * y[even_idx] + 5.0 * y[even_idx + 1])
                
                last_even = even_idx[-1]
                if last_even + 1 > n - 2:
                    # Last even index has no backward, only forward
                    if len(even_idx) > 1:
                        interior = even_idx[:-1]
                        bwd = result[interior + 1] - h12 * (-y[interior] + 8.0 * y[interior + 1] + 5.0 * y[interior + 2])
                        result[interior] = (fwd[:-1] + bwd) * 0.5
                    result[last_even] = fwd[-1]
                else:
                    # All even indices have backward available
                    bwd = result[even_idx + 1] - h12 * (-y[even_idx] + 8.0 * y[even_idx + 1] + 5.0 * y[even_idx + 2])
                    result[even_idx] = (fwd + bwd) * 0.5
        
        return result