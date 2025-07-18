import numpy as np
from scipy.signal import convolve

class Solver:
    """
    1D convolution using SciPy's optimized convolve (auto method).
    """

    def solve(self, problem, **kwargs):
        """
        :param problem: Tuple of two array-like (a, b)
        :return: Full convolution as np.ndarray
        """
        a, b = problem
        # direct call to SciPy's optimized convolution
        return convolve(a, b, mode='full')
        if next_fast_len:
            N = next_fast_len(total)
        else:
            N = 1 << (total - 1).bit_length()

        fa = np.fft.rfft(arr_a, n=N)
        fb = np.fft.rfft(arr_b, n=N)
        fc = fa * fb
        result = np.fft.irfft(fc, n=N)
        return result[:total]