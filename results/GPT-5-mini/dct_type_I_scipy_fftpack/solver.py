import numpy as np
from scipy.fftpack import dct

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the N-D Discrete Cosine Transform Type-I by applying 1-D
        DCT (type=1) along each axis using scipy.fftpack.dct.

        Processing axes in reverse order (last axis first) improves
        memory locality. overwrite_x=True requests in-place work when possible.
        """
        arr = np.asarray(problem, dtype=np.float64)
        if arr.size == 0:
            return arr.copy()

        res = arr.copy()
        dct_func = dct
        # Apply DCT-I along each axis, last axis first for better cache usage
        for axis in range(res.ndim - 1, -1, -1):
            res = dct_func(res, type=1, axis=axis, overwrite_x=True)
        return np.asarray(res, dtype=np.float64)