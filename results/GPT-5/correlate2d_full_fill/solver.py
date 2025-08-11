from typing import Any, Tuple

import numpy as np

class Solver:
    def __init__(self) -> None:
        # Reference-compatible attributes
        self.mode = "full"
        self.boundary = "fill"

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute the 2D correlation of arrays a and b using "full" mode and "fill" boundary
        via FFTs for speed.

        correlate2d(a, b, full, fill) == conv2d(a, conj(rot90(b, 2)), full, fill)

        Using FFT-based convolution with padding sizes >= M+P-1 and N+Q-1.
        """
        a, b = problem
        a = np.asarray(a)
        b = np.asarray(b)

        # Promote to consistent high-precision dtype
        is_complex = np.iscomplexobj(a) or np.iscomplexobj(b)
        if is_complex:
            a = a.astype(np.complex128, copy=False)
            b = b.astype(np.complex128, copy=False)
            # Flip and conjugate b for correlation via convolution
            b_flip = np.conjugate(b[::-1, ::-1])
        else:
            a = a.astype(np.float64, copy=False)
            b = b.astype(np.float64, copy=False)
            b_flip = b[::-1, ::-1]

        m, n = a.shape
        p, q = b.shape
        s0 = m + p - 1
        s1 = n + q - 1

        # Choose FFT sizes (prefer next_fast_len if SciPy is available)
        use_scipy_fft = False
        L0 = s0
        L1 = s1
        try:
            from scipy.fft import (
                rfft2 as srfft2,
                irfft2 as sirfft2,
                fft2 as sfft2,
                ifft2 as sifft2,
                next_fast_len,
            )

            L0 = next_fast_len(s0)
            L1 = next_fast_len(s1)
            use_scipy_fft = True
        except Exception:
            # Fall back to NumPy FFTs without next-fast optimization
            pass

        if is_complex:
            if use_scipy_fft:
                Fa = sfft2(a, (L0, L1))
                Fb = sfft2(b_flip, (L0, L1))
                out = sifft2(Fa * Fb)
            else:
                Fa = np.fft.fft2(a, s=(L0, L1))
                Fb = np.fft.fft2(b_flip, s=(L0, L1))
                out = np.fft.ifft2(Fa * Fb)
            out = out[:s0, :s1]
            return out.astype(np.complex128, copy=False)
        else:
            if use_scipy_fft:
                Fa = srfft2(a, (L0, L1))
                Fb = srfft2(b_flip, (L0, L1))
                out = sirfft2(Fa * Fb, (L0, L1))
            else:
                Fa = np.fft.rfft2(a, s=(L0, L1))
                Fb = np.fft.rfft2(b_flip, s=(L0, L1))
                out = np.fft.irfft2(Fa * Fb, s=(L0, L1))
            out = out[:s0, :s1]
            return out.astype(np.float64, copy=False)