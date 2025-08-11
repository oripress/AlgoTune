import numpy as np

# Use SciPy's dstn when available for performance and exact matching.
try:
    from scipy.fftpack import dstn as _dstn_scipy  # type: ignore
except Exception:
    _dstn_scipy = None

# Cache sine transform matrices to avoid recomputing them.
_S_CACHE = {}

def _get_sine_matrix(n: int) -> np.ndarray:
    """Return the (n x n) DST-II sine matrix S with S[k, j] = sin(pi*(k+1)*(j+1)/(n+1))."""
    mat = _S_CACHE.get(n)
    if mat is not None:
        return mat
    i = np.arange(1, n + 1, dtype=np.float64)
    S = np.sin(np.pi * np.outer(i, i) / (n + 1.0))
    _S_CACHE[n] = S
    return S

def _dstn_via_sine_matrices(a: np.ndarray) -> np.ndarray:
    """
    Compute DST-II along each axis by multiplying with precomputed sine matrices.
    Efficient for small-to-moderate sizes because it uses BLAS-backed matmul.
    """
    res = np.asarray(a, dtype=np.float64)
    for axis in range(res.ndim):
        n = res.shape[axis]
        if n <= 1:
            continue
        S = _get_sine_matrix(n)
        moved = np.moveaxis(res, axis, 0)  # shape (n, ...)
        n0 = moved.shape[0]
        flat = moved.reshape(n0, -1)  # (n, K)
        out_flat = S @ flat
        moved_out = out_flat.reshape(moved.shape)
        res = np.moveaxis(moved_out, 0, axis)
    return res

def _dst_axis_rfft(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute 1-D DST-II along the given axis using an odd extension and rfft.

    For input x of length n along axis, build y of length L = 2*(n+1):
      y = [0, x[0], ..., x[n-1], 0, -x[n-1], ..., -x[0]]
    Then Y = rfft(y) and DST-II coefficients are -0.5 * imag(Y[1..n]).
    """
    n = a.shape[axis]
    if n <= 1:
        return np.asarray(a, dtype=np.float64)

    moved = np.moveaxis(a, axis, 0)
    n = moved.shape[0]
    orig_shape = moved.shape
    flat = moved.reshape(n, -1)  # (n, K)

    L = 2 * (n + 1)
    K = flat.shape[1]
    y = np.empty((L, K), dtype=np.float64)
    # Construct odd extension: [0, x0..x_{n-1}, 0, -x_{n-1}..-x0]
    y[0, :] = 0.0
    y[1 : n + 1, :] = flat
    y[n + 1, :] = 0.0
    y[n + 2 :, :] = -flat[::-1, :]

    Y = np.fft.rfft(y, axis=0)
    out_flat = -0.5 * Y[1 : n + 1, :].imag
    out_moved = out_flat.reshape(orig_shape)
    return np.moveaxis(out_moved, 0, axis)

def _dstn_via_rfft(a: np.ndarray) -> np.ndarray:
    """Apply DST-II along each axis using the rfft-based routine."""
    res = np.asarray(a, dtype=np.float64)
    for axis in range(res.ndim):
        res = _dst_axis_rfft(res, axis)
    return res

class Solver:
    def __init__(self):
        # Threshold for choosing matrix (BLAS) method vs FFT-based method.
        # Matrix method often helps for small-to-moderate lengths.
        self._matrix_threshold = 64

    def solve(self, problem, **kwargs):
        """
        Compute the N-dimensional DST Type II.

        Strategy:
        - Convert input to float64 ndarray.
        - For small-to-moderate axis sizes, use BLAS-backed sine-matrix multiplications.
        - For larger sizes prefer SciPy's dstn if available; otherwise fall back to rfft-based method.
        """
        a = np.asarray(problem, dtype=np.float64)

        # Trivial cases
        if a.size == 0:
            return a.astype(np.float64)
        if a.ndim == 0:
            return np.asarray(a, dtype=np.float64)

        maxdim = max(a.shape)
        if maxdim <= self._matrix_threshold:
            return _dstn_via_sine_matrices(a)
        if _dstn_scipy is not None:
            try:
                return _dstn_scipy(a, type=2)
            except Exception:
                return _dstn_via_rfft(a)
        return _dstn_via_rfft(a)