from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

class Solver:
    """
    Fast upfirdn1d solver.

    Optimizations:
    - Cache SciPy's internal UpFIRDn(h, up, down) objects keyed by a cheap content
      signature of h (so reuse works even if h is passed as a new list each time).
    - Pre-import SciPy callables in __init__ (init cost not counted).
    - If UpFIRDn isn't available, try using scipy.signal._upfirdn.upfirdn directly
      (often lower overhead than scipy.signal.upfirdn wrapper).
    """

    __slots__ = (
        "_UpFIRDn",
        "_has_apply",
        "_upfirdn_pub",
        "_upfirdn_int",
        "_int_mode",
        "_ufd_cache",
        "_max_cache_keys",
    )

    def __init__(self) -> None:
        # Public upfirdn (always available with SciPy)
        from scipy import signal  # type: ignore

        self._upfirdn_pub: Callable[..., np.ndarray] = signal.upfirdn

        # Try internal UpFIRDn class
        self._UpFIRDn = None
        self._has_apply = False
        try:
            from scipy.signal._upfirdn import UpFIRDn as _UpFIRDn  # type: ignore

            self._UpFIRDn = _UpFIRDn
            self._has_apply = hasattr(_UpFIRDn, "apply")
        except Exception:
            self._UpFIRDn = None
            self._has_apply = False

        # Try internal upfirdn function
        self._upfirdn_int: Optional[Callable[..., np.ndarray]] = None
        self._int_mode = 0
        try:
            import scipy.signal._upfirdn as _m  # type: ignore

            f = getattr(_m, "upfirdn", None)
            if callable(f):
                h0 = np.array([1.0, 2.0], dtype=np.float64)
                x0 = np.array([3.0, 4.0, 5.0], dtype=np.float64)
                ref = self._upfirdn_pub(h0, x0, up=2, down=1)

                # Mode 0: f(h, x, up, down)
                ok = False
                try:
                    y = f(h0, x0, 2, 1)
                    if y.shape == ref.shape and np.allclose(y, ref):
                        self._upfirdn_int = f
                        self._int_mode = 0
                        ok = True
                except Exception:
                    pass

                if not ok:
                    # Mode 1: f(h, x, up=, down=)
                    try:
                        y = f(h0, x0, up=2, down=1)
                        if y.shape == ref.shape and np.allclose(y, ref):
                            self._upfirdn_int = f
                            self._int_mode = 1
                            ok = True
                    except Exception:
                        pass
        except Exception:
            self._upfirdn_int = None
            self._int_mode = 0

        # key -> list[(h_arr, UpFIRDn_instance)] (bucket to avoid signature collisions)
        self._ufd_cache: Dict[Tuple[Any, ...], List[Tuple[np.ndarray, Any]]] = {}
        self._max_cache_keys = 1024

    @staticmethod
    def _as_1d_contig(a: Any) -> np.ndarray:
        arr = np.asarray(a)
        if arr.ndim != 1:
            arr = np.ravel(arr)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    @staticmethod
    def _h_signature(h_arr: np.ndarray, up: int, down: int) -> Tuple[Any, ...]:
        lh = int(h_arr.size)
        dts = h_arr.dtype.str
        if lh <= 8:
            return (up, down, dts, lh, h_arr.tobytes())
        return (up, down, dts, lh, h_arr[:4].tobytes(), h_arr[-4:].tobytes())

    def solve(self, problem, **kwargs) -> Any:
        # Normalize input shape.
        if isinstance(problem, tuple) and len(problem) == 4:
            problem_list = [problem]
        else:
            problem_list = problem

        n = len(problem_list)
        results: List[Any] = [None] * n
        if n == 0:
            return results

        UpFIRDn = self._UpFIRDn

        # Fast path: use cached UpFIRDn when available.
        if UpFIRDn is not None:
            cache = self._ufd_cache
            max_cache_keys = self._max_cache_keys
            has_apply = self._has_apply

            for i in range(n):
                h, x, up, down = problem_list[i]
                up_i = int(up)
                down_i = int(down)

                x_arr = np.asarray(x)
                h_arr = self._as_1d_contig(h)

                key = self._h_signature(h_arr, up_i, down_i)
                bucket = cache.get(key)

                ufd = None
                if bucket is not None:
                    for h0, u0 in bucket:
                        if h0 is h_arr or np.array_equal(h0, h_arr):
                            ufd = u0
                            break

                if ufd is None:
                    ufd = UpFIRDn(h_arr, up=up_i, down=down_i)
                    if bucket is None:
                        cache[key] = [(h_arr, ufd)]
                    else:
                        bucket.append((h_arr, ufd))
                    if len(cache) > max_cache_keys:
                        cache.clear()

                if has_apply:
                    results[i] = ufd.apply(x_arr)
                else:
                    results[i] = ufd(x_arr)

            return results

        # Fallback: internal function if available, else public wrapper.
        f_int = self._upfirdn_int
        if f_int is not None:
            mode = self._int_mode
            if mode == 0:
                for i, (h, x, up, down) in enumerate(problem_list):
                    results[i] = f_int(self._as_1d_contig(h), np.asarray(x), int(up), int(down))
            else:
                for i, (h, x, up, down) in enumerate(problem_list):
                    results[i] = f_int(self._as_1d_contig(h), np.asarray(x), up=int(up), down=int(down))
            return results

        upfirdn = self._upfirdn_pub
        for i, (h, x, up, down) in enumerate(problem_list):
            results[i] = upfirdn(h, x, up=int(up), down=int(down))
        return results