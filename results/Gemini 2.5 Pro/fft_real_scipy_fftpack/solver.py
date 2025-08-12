import numpy as np
from typing import Any
import scipy.fft
import os

class Solver:
    def __init__(self):
        # Determine the number of available CPU cores for parallelization.
        # Using -1 in scipy.fft is also an option to use all cores, but
        # being explicit can be clearer.
        try:
            # os.sched_getaffinity is robust for containerized environments (e.g., Docker).
            self._workers = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for systems without sched_getaffinity (e.g., Windows).
            self._workers = os.cpu_count() if os.cpu_count() else 1

    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        """
        Computes the N-dimensional Fast Fourier Transform (FFT) of a real-valued matrix.
        
        This implementation uses scipy.fft.fftn, which is a high-performance
        FFT implementation built on the pocketfft library. It is parallelized 
        across multiple CPU cores for a significant speedup over the standard
        single-threaded numpy.fft.fftn.
        """
        # scipy.fft.fftn computes the full complex-to-complex FFT.
        # The 'workers' argument enables multi-threaded computation, providing the speedup.
        return scipy.fft.fftn(problem, workers=self._workers)