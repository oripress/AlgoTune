from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array using Simpson's rule.
        Vectorized implementation specialized for uniform spacing (dx) along the last axis.

        Matches SciPy's cumulative_simpson default behavior:
        - axis = -1
        - even = 'avg'
        - initial = None (returns shape with last axis length N-1)
        """
        y = np.asarray(problem["y2"])
        dx = problem["dx"]

        if y.ndim == 0:
            # Scalar input -> integral is zero-length when initial=None
            return np.empty((0,), dtype=y.dtype)

        n = y.shape[-1]
        # Prepare output buffer (same length as input initially)
        out = np.zeros_like(y)

        if n <= 1:
            # With initial=None, length is N-1 -> empty along last axis
            return out[..., 1:]

        # Slices of even and odd indices along the last axis
        y_even = y[..., 0::2]  # indices: 0, 2, 4, ...
        y_odd = y[..., 1::2]   # indices: 1, 3, 5, ...

        ne = y_even.shape[-1]  # ceil(n/2)
        no = y_odd.shape[-1]   # floor(n/2)

        # Cumulative sums along the last axis for even and odd indexed samples
        # Shape preservation: (..., ne) and (..., no)
        E = np.cumsum(y_even, axis=-1)
        O = np.cumsum(y_odd, axis=-1)

        # Fill even indices (k = 2m) for m >= 1 using vectorized Simpson blocks
        # I[2m] = dx/3 * ( -y0 + y_{2m} + 4*sum_{i=0}^{m-1} y_{2i+1} + 2*sum_{i=1}^{m-1} y_{2i} )
        if ne > 1:
            y0 = y_even[..., :1]  # broadcastable
            out_even_rest = (
                (dx / 3.0)
                * (
                    -y0
                    + y_even[..., 1:]
                    + 4.0 * O[..., : ne - 1]
                    + 2.0 * E[..., : ne - 1]
                )
            )
            # Place into even positions starting from index 2
            out[..., 0::2][..., 1:] = out_even_rest

        # Fill odd indices (k = 2m + 1) using 'avg' of 'last' and 'first' strategies
        # last: I_last[2m+1] = I[2m] + dx/2 * ( y_{2m} + y_{2m+1} )
        # first: I_first[2m+1] = dx/2*(y0+y1) + dx/3*( y1 + y_{2m+1}
        #                           + 4*sum_{i=1..m} y_{2i}
        #                           + 2*sum_{i=1..m-1} y_{2i+1} )
        if no > 0:
            I_even = out[..., 0::2]  # (..., ne)
            I_last_odd = I_even[..., :no] + (dx * 0.5) * (y_even[..., :no] + y_odd)

            y0 = y_even[..., :1]
            y1 = y_odd[..., :1]
            E_m = E[..., :no]
            if no > 1:
                O_m1 = np.concatenate([np.zeros_like(y1), O[..., : no - 1]], axis=-1)
            else:
                O_m1 = np.zeros_like(y1)

            I_first_odd = (dx * 0.5) * (y0 + y1) + (dx / 3.0) * (
                y1 + y_odd + 4.0 * (E_m - y0) + 2.0 * (O_m1 - y1)
            )

            out[..., 1::2] = 0.5 * (I_last_odd + I_first_odd)

        # Match SciPy's default initial=None -> drop the initial zero so output has length N-1
        return out[..., 1:]