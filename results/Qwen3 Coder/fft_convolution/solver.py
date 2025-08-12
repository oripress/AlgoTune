from typing import Dict, Any
import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the convolution problem using the Fast Fourier Transform approach.

        Uses scipy.signal.fftconvolve to compute the convolution of signals x and y.

        :param problem: A dictionary representing the convolution problem.
        :return: A dictionary with key "result": a list representing the convolution result.
        """
        signal_x = np.array(problem["signal_x"])
        signal_y = np.array(problem["signal_y"])
        mode = problem.get("mode", "full")

        # Perform convolution using FFT
        convolution_result = signal.fftconvolve(signal_x, signal_y, mode=mode)

        return {"result": convolution_result.tolist()}