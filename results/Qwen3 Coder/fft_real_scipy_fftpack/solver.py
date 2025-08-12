import scipy.fft as fft
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        return fft.fftn(problem)
        problem = np.asarray(problem)
        # Create output array
        output = pyfftw.empty_like(problem, dtype=complex)
        # Create FFTW object
        fft_obj = pyfftw.FFTW(problem, output, axes=range(problem.ndim))
        # Execute FFT
        return fft_obj()