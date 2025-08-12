import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the N-dimensional FFT of a complex-valued matrix.

        Parameters
        ----------
        problem : array-like
            Input square matrix (list of lists or ndarray) containing complex numbers.

        Returns
        -------
        ndarray
            The N-dimensional FFT of the input matrix.
        """
        # Convert input to a NumPy array of complex dtype
        arr = np.array(problem, dtype=complex)
        # Use NumPy's FFT implementation, which is efficient and matches the reference.
        return np.fft.fftn(arr)