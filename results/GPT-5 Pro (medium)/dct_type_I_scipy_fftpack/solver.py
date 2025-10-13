from typing import Any

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the N-dimensional DCT Type I using scipy.fft (pocketfft backend),
        which is typically faster than scipy.fftpack.dctn. Uses multithreading.

        Parameters:
            problem: ndarray-like input array
            kwargs:
                workers (int): Number of worker threads for parallel computation. Default -1 (all cores).
                overwrite_x (bool): Allow overwrite of the input for speed. Default False.

        Returns:
            ndarray: DCT-I of the input array.
        """
        # Import locally to avoid import overhead when module is loaded
        from scipy.fft import dctn

        # Ensure input is an ndarray; keep dtype as-is
        x = np.asarray(problem)

        # Use all available threads by default
        workers = kwargs.get("workers", -1)
        overwrite_x = kwargs.get("overwrite_x", False)

        # Perform DCT Type-I across all axes
        # This matches scipy.fftpack.dctn(problem, type=1) scaling/convention
        result = dctn(x, type=1, workers=workers, overwrite_x=overwrite_x)

        return result