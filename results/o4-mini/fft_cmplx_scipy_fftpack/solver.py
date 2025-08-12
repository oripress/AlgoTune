import os
# maximize threading for BLAS libs
_nt = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(_nt)
os.environ["OPENBLAS_NUM_THREADS"] = str(_nt)
os.environ["MKL_NUM_THREADS"] = str(_nt)

import numpy as np

# JAX FFT
try:
    import jax
    import jax.numpy as jnp
    _fft_jit = jax.jit(lambda x: jnp.fft.fftn(x))
    _has_jax = True
except Exception:
    _has_jax = False

# PyTorch FFT
try:
    import torch
    torch.set_num_threads(_nt)
    torch.set_num_interop_threads(_nt)
    import importlib
    _torch_fft_mod = importlib.import_module("torch.fft") if hasattr(torch, "fft") else None
    _torch_fftn = getattr(_torch_fft_mod, "fftn", None) if _torch_fft_mod else None
    _torch_cuda = torch.cuda.is_available()
    _has_torch = _torch_fftn is not None
except Exception:
    _has_torch = False

# SciPy FFT
try:
    from scipy.fft import fftn as _scipy_fftn
    _has_scipy = True
except Exception:
    _has_scipy = False

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute N-dimensional FFT using JAX (fast compiled), PyTorch, SciPy, or NumPy.
        """
        arr = np.asarray(problem, dtype=np.complex128)
        if _has_jax:
            jarr = jnp.asarray(arr)
            res = _fft_jit(jarr)
            return np.asarray(res)
        if _has_torch:
            x = torch.as_tensor(arr, dtype=torch.complex128)
            if _torch_cuda:
                x = x.cuda()
                y = _torch_fftn(x)
                y = y.cpu()
            else:
                y = _torch_fftn(x)
            return y.numpy()
        if _has_scipy:
            return _scipy_fftn(arr, workers=_nt)
        return np.fft.fftn(arr)