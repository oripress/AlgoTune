from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

def _upfirdn_1d(h: np.ndarray, x: np.ndarray, up: int, down: int) -> np.ndarray:
    """
    Efficient 1D upfirdn implementation using polyphase decomposition and NumPy.

    Computes y[n] = sum_k x[k] * h[n*down - k*up], with zero-extension.

    Parameters
    ----------
    h : 1D ndarray
        FIR filter coefficients.
    x : 1D ndarray
        Input signal.
    up : int
        Upsampling factor (>=1).
    down : int
        Downsampling factor (>=1).

    Returns
    -------
    y : 1D ndarray
        The result of upsample-filter-downsample.
    """
    # Ensure 1D contiguous float64 arrays for consistent behavior/performance
    h = np.asarray(h, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()

    # Handle trivial/edge cases
    if up <= 0 or down <= 0:
        raise ValueError("Upsampling and downsampling factors must be positive integers.")
    Lh = h.size
    Nx = x.size

    # If either x or h is empty, the output is determined by lengths:
    # length of z (conv of upsampled x with h) is Nx*up + Lh - 1
    # y is z[::down] => length = ((zlen - 1) // down) + 1 if zlen > 0 else 0
    if Nx == 0 or Lh == 0:
        zlen = Nx * up + Lh - 1
        if zlen <= 0:
            return np.zeros(0, dtype=np.float64)
        ylen = (zlen - 1) // down + 1
        return np.zeros(ylen, dtype=np.float64)

    # Fast paths
    if up == 1 and down == 1:
        # Just full convolution
        return np.convolve(x, h, mode="full")

    if up == 1:
        # Just convolution then decimate
        z = np.convolve(x, h, mode="full")
        return z[::down]

    if Lh == 1:
        # Single-tap filter: pure resample with scaling
        # zlen = Nx*up + 1 - 1 = Nx*up
        zlen = Nx * up
        ylen = (zlen - 1) // down + 1
        y = np.zeros(ylen, dtype=np.float64)
        # y[n] = h[0] * x[(n*down)/up] if divisible, else 0
        h0 = h[0]
        n = np.arange(ylen, dtype=np.int64)
        nd = n * down
        mask = (nd % up) == 0
        idx = (nd[mask] // up).astype(np.int64)
        # idx must be within [0, Nx-1], which it will be by construction
        y[mask] = h0 * x[idx]
        return y

    # General case: polyphase decomposition of h by modulo 'up':
    # For residue r, define h_r = h[r::up] (length Pr).
    # Then y[n] = (x * h_r)[n0] with r = (n*down) % up and n0 = (n*down - r) // up.
    # Output length:
    zlen = Nx * up + Lh - 1
    ylen = (zlen - 1) // down + 1
    y = np.zeros(ylen, dtype=np.float64)

    # Precompute residues and base indices
    n = np.arange(ylen, dtype=np.int64)
    nd = n * down
    r_arr = (nd % up).astype(np.int64)
    # n0 = floor((nd - r)/up) = (nd - r)//up
    n0_arr = (nd - r_arr) // up  # integer division

    # For each phase r with nonempty taps: compute convolution once and gather samples.
    max_r = min(up, Lh)  # residues beyond Lh-1 have zero taps
    # We iterate only residues that actually appear in r_arr for efficiency
    # but bounded to [0, max_r)
    unique_r = np.unique(r_arr)
    for r in unique_r:
        if r >= max_r:
            # No taps for this residue; contributions are zero
            continue
        h_r = h[r::up]
        if h_r.size == 0:
            continue
        w_r = np.convolve(x, h_r, mode="full")  # length Nx + Pr - 1
        # select indices where residue matches r
        idxs = np.nonzero(r_arr == r)[0]
        n0s = n0_arr[idxs]
        # Keep those within bounds of w_r
        # Valid indices: 0 <= n0 < w_r.size
        good = (n0s >= 0) & (n0s < w_r.size)
        if np.any(good):
            y[idxs[good]] = w_r[n0s[good]]

    return y

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the upfirdn operation for each problem instance.

        :param problem: A list-like of tuples (h, x, up, down).
        :return: A list of 1D numpy arrays representing the upfirdn results.
        """
        results: List[np.ndarray] = []
        # Iterate problems and compute results
        for item in problem:
            # Support input as tuple/list with 4 entries
            try:
                h, x, up, down = item  # type: ignore[misc]
            except Exception as exc:
                raise ValueError("Each problem item must be a tuple (h, x, up, down).") from exc

            # Convert up/down to ints
            up_i = int(up)
            down_i = int(down)

            res = _upfirdn_1d(h, x, up_i, down_i)
            results.append(res)
        return results