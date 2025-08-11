import numpy as np

class Solver:
    def __init__(self, mode: str = "full"):
        """
        Initialize the solver with a correlation mode.
        Supported modes are "full" (default) and "valid".
        """
        self.mode = mode

    def solve(self, problem, **kwargs):
        """
        Compute the 1‑D correlation for each pair in ``problem``.
        ``problem`` is an iterable of (a, b) where each is a
        1‑D array‑like of floats.

        Parameters
        ----------
        problem : list or iterable
            List of pairs (a, b). Each element can be a list,
            tuple, or numpy array.

        Returns
        -------
        list of np.ndarray
            Correlation results for each valid pair.
        """
        # Allow overriding mode via kwargs
        mode = kwargs.get("mode", self.mode)

        results = []
        for a, b in problem:
            # Convert to numpy arrays (ensure 1‑D)
            a_arr = np.asarray(a, dtype=np.float64).ravel()
            b_arr = np.asarray(b, dtype=np.float64).ravel()

            # Skip invalid pairs when mode is "valid"
            if mode == "valid" and b_arr.shape[0] > a_arr.shape[0]:
                continue

            # Compute correlation using FFT for speed
            n = a_arr.size + b_arr.size - 1
            n_fft = 1 << (n - 1).bit_length()
            A = np.fft.rfft(a_arr, n=n_fft)
            B = np.fft.rfft(b_arr[::-1], n=n_fft)
            C = A * B
            full_corr = np.fft.irfft(C, n=n_fft)[:n]
            if mode == "full":
                res = full_corr
            else:  # mode == "valid"
                valid_len = a_arr.size - b_arr.size + 1
                start = b_arr.size - 1
                res = full_corr[start:start + valid_len]
            results.append(res)
        return results