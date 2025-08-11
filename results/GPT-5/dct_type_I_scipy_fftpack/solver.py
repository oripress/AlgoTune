from typing import Any

import numpy as np

# Prefer pocketfft (scipy.fft) if available
try:
    from scipy.fft import dctn as _p_dctn  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _p_dctn = None  # type: ignore[assignment]

try:
    from scipy.fft import dct as _p_dct  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _p_dct = None  # type: ignore[assignment]

# Fallback to fftpack if needed
try:
    from scipy.fftpack import dctn as _f_dctn  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _f_dctn = None  # type: ignore[assignment]

try:
    from scipy.fftpack import dct as _f_dct  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _f_dct = None  # type: ignore[assignment]

# Detect supported kwargs once to avoid per-call try/except overhead
_KW_DCTN = {}
_KW_DCT = {}
if _p_dctn is not None:
    try:
        import inspect

        sig = inspect.signature(_p_dctn)
        if "workers" in sig.parameters:
            _KW_DCTN["workers"] = -1
        if "overwrite_x" in sig.parameters:
            _KW_DCTN["overwrite_x"] = True
    except Exception:
        pass

if _p_dct is not None:
    try:
        import inspect

        sig = inspect.signature(_p_dct)
        if "workers" in sig.parameters:
            _KW_DCT["workers"] = -1
        if "overwrite_x" in sig.parameters:
            _KW_DCT["overwrite_x"] = True
    except Exception:
        pass

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the N-dimensional DCT Type I of the input.

        Prefers scipy.fft.dctn with multithreading when available;
        falls back to separable 1D DCT-I or fftpack as needed.
        """
        a = np.asarray(problem)

        # Fast path: pocketfft dctn
        if _p_dctn is not None:
            return _p_dctn(a, type=1, **_KW_DCTN)

        # Next: separable with pocketfft dct
        if _p_dct is not None:
            x = a
            # Process along each axis; skip degenerate axes (size <= 1)
            for ax in range(x.ndim):
                if x.shape[ax] <= 1:
                    continue
                x = _p_dct(x, type=1, axis=ax, **_KW_DCT)
            return x

        # Fallback: fftpack dctn
        if _f_dctn is not None:
            return _f_dctn(a, type=1)

        # Last resort: separable with fftpack dct
        x = a
        for ax in range(x.ndim):
            if x.shape[ax] <= 1:
                continue
            x = _f_dct(x, type=1, axis=ax)  # type: ignore[operator]
        return x