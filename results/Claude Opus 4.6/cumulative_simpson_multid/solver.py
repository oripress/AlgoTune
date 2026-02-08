import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y2"], dtype=np.float64)
        dx = float(problem["dx"])
        n = y.shape[-1]
        
        if n < 3:
            return np.cumsum(dx * (y[..., :-1] + y[..., 1:]) / 2, axis=-1)
        
        m = n - 1
        num_pairs = m // 2
        
        # Extract the three points for each Simpson pair
        y0 = y[..., :-2:2]   # y[0], y[2], ..., y[2*(num_pairs-1)]
        y1 = y[..., 1:-1:2]  # y[1], y[3], ..., y[2*(num_pairs-1)+1]
        y2 = y[..., 2::2]    # y[2], y[4], ..., y[2*num_pairs]
        
        # Simpson's 1/3 rule for each pair
        S = (dx / 3.0) * (y0 + 4.0 * y1 + y2)
        CS = np.cumsum(S, axis=-1)
        
        # Backward correction: integral of last sub-interval in each pair
        bc = (dx / 12.0) * (-y0 + 8.0 * y1 + 5.0 * y2)
        
        # Build result
        out_shape = list(y.shape)
        out_shape[-1] = m
        result = np.empty(out_shape, dtype=y.dtype)
        
        # Odd indices: result[1], result[3], ... = cumulative Simpson pairs
        result[..., 1::2] = CS
        
        # Even indices: result[0], result[2], ... = CS[k] - backward_correction[k]
        even_view = result[..., ::2]
        even_from_simpson = CS - bc
        
        if n % 2 == 0:  # Last result index is even, needs special handling
            even_view[..., :num_pairs] = even_from_simpson
            # Last even index uses backward formula for final interval
            even_view[..., num_pairs] = (CS[..., -1] + 
                (dx / 12.0) * (-y[..., -3] + 8.0 * y[..., -2] + 5.0 * y[..., -1]))
        else:
            even_view[..., :] = even_from_simpson
        
        return result