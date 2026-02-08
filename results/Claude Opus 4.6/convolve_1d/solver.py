import numpy as np
from scipy.fft import next_fast_len
import numba

try:
    from conv_cython import direct_convolve_cy
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

@numba.njit(cache=True, fastmath=True)
def direct_convolve(a, b):
    """Direct convolution."""
    na = a.shape[0]
    nb = b.shape[0]
    out_len = na + nb - 1
    out = np.empty(out_len)
    if nb <= na:
        for i in range(out_len):
            s = 0.0
            jlo = i - na + 1
            if jlo < 0:
                jlo = 0
            jhi = i + 1
            if jhi > nb:
                jhi = nb
            for j in range(jlo, jhi):
                s += b[j] * a[i - j]
            out[i] = s
    else:
        for i in range(out_len):
            s = 0.0
            jlo = i - nb + 1
            if jlo < 0:
                jlo = 0
            jhi = i + 1
            if jhi > na:
                jhi = na
            for j in range(jlo, jhi):
                s += a[j] * b[i - j]
            out[i] = s
    return out

# Pre-warm numba
_wa = np.array([1.0, 2.0, 3.0])
_wb = np.array([1.0, 2.0])
direct_convolve(_wa, _wb)

_rfft = np.fft.rfft
_irfft = np.fft.irfft
_f64 = np.float64
_nfl_cache = {}

def _fast_nfl(n):
    try:
        return _nfl_cache[n]
    except KeyError:
        v = next_fast_len(n)
        _nfl_cache[n] = v
        return v

def _oa_convolve(long_arr, short_arr, n_long, n_short):
    """Overlap-add convolution for asymmetric sizes."""
    out_len = n_long + n_short - 1
    
    # Choose block size to minimize total FFT work
    # Each block does FFT of size fft_len, need n_long/block_size blocks
    # Total work ~ (n_long/block_size + 1) * fft_len * log(fft_len)
    # We want block_size >> n_short for efficiency
    ideal_block = max(256, 4 * n_short)
    fft_len = _fast_nfl(ideal_block + n_short - 1)
    block_size = fft_len - n_short + 1
    
    # Pre-compute FFT of short array
    fb = _rfft(short_arr, fft_len)
    
    out = np.zeros(out_len)
    pos = 0
    while pos < n_long:
        end = min(pos + block_size, n_long)
        fc = _rfft(long_arr[pos:end], fft_len)
        fc *= fb
        conv_chunk = _irfft(fc, fft_len)
        
        chunk_out_len = min(end - pos + n_short - 1, out_len - pos)
        out[pos:pos + chunk_out_len] += conv_chunk[:chunk_out_len]
        pos += block_size
    
    return out

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        if type(a) is not np.ndarray or a.dtype != _f64:
            a = np.asarray(a, dtype=_f64)
        if type(b) is not np.ndarray or b.dtype != _f64:
            b = np.asarray(b, dtype=_f64)
        
        na = a.shape[0]
        nb = b.shape[0]
        mn = min(na, nb)
        mx = max(na, nb)
        
        # Direct convolution for small arrays
        if mn * mx <= 40000:
            if _HAS_CYTHON:
                return direct_convolve_cy(a, b)
            return direct_convolve(a, b)
        
        out_len = na + nb - 1
        
        # For highly asymmetric arrays, overlap-add is more efficient
        # Compare OA cost vs full FFT cost
        fft_full = _fast_nfl(out_len)
        
        if mn < 512 and mx > 16 * mn:
            # OA is likely better
            if na >= nb:
                return _oa_convolve(a, b, na, nb)
            else:
                return _oa_convolve(b, a, nb, na)
        
        # Standard FFT convolution
        fa = _rfft(a, fft_full)
        fa *= _rfft(b, fft_full)
        return _irfft(fa, fft_full)[:out_len]