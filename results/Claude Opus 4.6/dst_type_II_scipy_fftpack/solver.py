import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Compute the 2D DST Type II using FFT."""
        x = np.asarray(problem, dtype=np.float64)
        result = _dst2_axis_fft(x, 0)
        result = _dst2_axis_fft(result, 1)
        return result

def _dst2_axis_fft(x, axis):
    """
    Compute DST-II along given axis using a 2N-length FFT.
    
    DST-II_k = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(2n+1)*(k+1)/(2N))
    for k = 0, ..., N-1
    """
    N = x.shape[axis]
    if N == 0:
        return x.copy()
    
    # We'll work with the data along the specified axis.
    # Move the target axis to the last position for easier manipulation
    x_moved = np.moveaxis(x, axis, -1)  # shape: (..., N)
    shape = x_moved.shape
    M = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    x_flat = x_moved.reshape(M, N)
    
    # Method: Use a 4N-point real FFT
    # Create y of length 4N:
    # y = [0, x[0], 0, x[1], 0, x[2], ..., 0, x[N-1], 0, -x[N-1], 0, -x[N-2], ..., 0, -x[0]]
    # This has odd symmetry and the FFT gives pure imaginary values at odd harmonics.
    # DST-II_k = -Im(FFT(y)[k+1])
    # But 4N FFT might be wasteful.
    
    # Better: Use N-point FFT with twiddle factors (like DCT-II via FFT)
    # For DCT-II, we reorder: y[j] = x[2j] for j < ceil(N/2), y[N-1-j] = x[2j+1]
    # Then DCT-II_k = 2 * Re(FFT(y)[k] * exp(-j*pi*k/(2N)))
    
    # For DST-II, we can use the relationship:
    # DST-II_k(x[0], ..., x[N-1]) = DCT-II_{N-1-k}(x[N-1], ..., x[0]) with sign adjustment
    # Actually: DST-II_k(x) = (-1)^k * DCT-II_k(x_reversed, shifted)
    # Let me derive this properly.
    
    # DST-II_k = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(2n+1)*(k+1)/(2N))
    # Let m = N-1-n:
    # = 2 * sum_{m=0}^{N-1} x[N-1-m] * sin(pi*(2(N-1-m)+1)*(k+1)/(2N))
    # = 2 * sum_{m=0}^{N-1} x[N-1-m] * sin(pi*(2N-2m-1)*(k+1)/(2N))
    # sin(pi*(2N-2m-1)*(k+1)/(2N)) = sin(pi*(k+1) - pi*(2m+1)*(k+1)/(2N))
    # = sin(pi*(k+1))*cos(pi*(2m+1)*(k+1)/(2N)) - cos(pi*(k+1))*sin(pi*(2m+1)*(k+1)/(2N))
    # = 0 - (-1)^{k+1} * sin(pi*(2m+1)*(k+1)/(2N))
    # = (-1)^k * sin(pi*(2m+1)*(k+1)/(2N))
    
    # So DST-II_k(x) = (-1)^k * 2 * sum_{m=0}^{N-1} x_r[m] * sin(pi*(2m+1)*(k+1)/(2N))
    # = (-1)^k * DST-II_k(x_r)
    # where x_r is reversed x. This is a tautology!
    
    # Let me try a different approach. Use the relation between DST-II and DCT-II:
    # sin(pi*(2n+1)*(k+1)/(2N)) 
    # = cos(pi/2 - pi*(2n+1)*(k+1)/(2N))
    # Hmm, this doesn't simplify nicely.
    
    # Direct approach using 2N FFT:
    # Form v of length 2N:
    # v[n] = x[n] for n = 0, ..., N-1
    # v[n] = -x[2N-1-n] for n = N, ..., 2N-1  (so v[N] = -x[N-1], v[2N-1] = -x[0])
    
    v = np.zeros((M, 2*N), dtype=x.dtype)
    v[:, :N] = x_flat
    v[:, N:] = -x_flat[:, ::-1]
    
    # DFT(v)[k] for k=0..2N-1
    # Note v has odd symmetry: v[2N-1-n] = -v[n] (check: v[2N-1-n] when n < N: v[2N-1-n] = -x[2N-1-(2N-1-n)] = -x[n] = -v[n] ✓)
    # Wait: for n < N: v[n] = x[n], and v[2N-1-n] for n < N means 2N-1-n >= N, so v[2N-1-n] = -x[2N-1-(2N-1-n)] = -x[n] ✓
    
    # So DFT(v)[k] = sum_{n=0}^{2N-1} v[n] * exp(-j*2*pi*k*n/(2N))
    # Since v has odd anti-symmetry v[2N-n] = -v[n] (with v[0] and v[N]):
    # Actually v[0] = x[0] and v[2N-1] = -x[0], these are NOT v[2N-0] since index wraps.
    # The anti-symmetry is v[2N-1-n] = -v[n], not v[2N-n] = -v[n].
    
    # Let me just compute the FFT and extract what we need:
    V = np.fft.fft(v, axis=1)  # shape (M, 2N), complex
    
    # From the DFT of v, we need to extract DST-II coefficients.
    # V[k] = sum_{n=0}^{2N-1} v[n] * W^{kn} where W = exp(-j*2*pi/(2N))
    
    # v[n] = x[n] for n=0..N-1, v[n] = -x[2N-1-n] for n=N..2N-1
    # V[k] = sum_{n=0}^{N-1} x[n]*W^{kn} - sum_{n=N}^{2N-1} x[2N-1-n]*W^{kn}
    # In second sum, let m=2N-1-n, n=2N-1-m, m goes from 0 to N-1:
    # = sum_{n=0}^{N-1} x[n]*W^{kn} - sum_{m=0}^{N-1} x[m]*W^{k(2N-1-m)}
    # = sum x[n]*W^{kn} - W^{k(2N-1)} * sum x[m]*W^{-km}
    
    # W^{k(2N-1)} = exp(-j*2*pi*k*(2N-1)/(2N)) = exp(-j*2*pi*k) * exp(j*2*pi*k/(2N))
    # = exp(j*pi*k/N) (since exp(-j*2*pi*k) = 1)
    
    # Let A = sum x[n]*exp(-j*pi*k*n/N) (= DFT of x with freq halved)
    # Let B = sum x[m]*exp(j*pi*k*m/N) = conj(A) if x is real
    
    # V[k] = A - exp(j*pi*k/N) * B = A - exp(j*pi*k/N) * conj(A)
    
    # If A = a + jb, then V[k] = (a+jb) - exp(j*pi*k/N)*(a-jb)
    # Let c = cos(pi*k/N), s = sin(pi*k/N)
    # V[k] = (a+jb) - (c+js)(a-jb) = (a+jb) - (ca + jsb + j*sa - j^2*bs... wait let me be careful)
    # (c+js)(a-jb) = ca - cjb + jsa - j^2 sb = ca + sb + j(sa - cb)
    # V[k] = a + jb - ca - sb - j(sa - cb) = a(1-c) - sb + j(b - sa + cb)
    # = a(1-c) - sb + j(b(1+c) - sa) -- hmm, that doesn't simplify to DST.
    
    # Actually, wait. Let me reconsider what V[k] has to do with DST-II.
    # Maybe I should look at this differently.
    
    # The DST-II is defined as:
    # X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(k+1)*(2n+1)/(2N))  for k=0..N-1
    
    # Let me try computing with a half-sample shifted FFT approach.
    # Define z[n] = x[n] * exp(-j*pi*(n+0.5)/(something))... this is getting complicated.
    
    # Let me try a completely different approach. Use rfft and twiddle factors.
    
    # The key insight for DST-II via N-point FFT:
    # 1. Reverse the input: x_r[n] = x[N-1-n]
    # 2. Reorder like for DCT-II: y[j] = x_r[2j] for j = 0..ceil(N/2)-1, 
    #    y[N-1-j] = x_r[2j+1] for j = 0..floor(N/2)-1
    # 3. FFT of y -> Y
    # 4. Apply twiddle: DST-II_{N-1-k} = 2*Re(Y[k] * exp(-j*pi*k/(2N)))
    # Wait, I need to verify this.
    
    # Actually, looking at it again, let me use a simpler verified relationship:
    # DST-II(x)[k] for k=0..N-1 with the scipy convention
    # scipy.fftpack.dst(x, 2) = sum_{n=0}^{N-1} 2*x[n]*sin(pi*(2n+1)*(k+1)/(2N))
    
    # I'll use the verified implementation:
    # 1. Reverse x
    # 2. Compute DCT-II of reversed x
    # 3. DST-II[k] = (-1)^k * DCT-II[k] -- no, we showed this is a tautology
    
    # Let me try yet another path:
    # DST-II[k+1] relates to DCT-III... 
    
    # OK, I'll just directly implement it using the definition with vectorized numpy
    # For large N, this would be O(N^2) per row, which is too slow.
    
    # Let me just use scipy. The current approach is getting nowhere fast.
    import scipy.fftpack
    result = scipy.fftpack.dst(x_flat.T, type=2, axis=0).T  # apply along columns after transpose
    result = result.reshape(shape)
    return np.moveaxis(result, -1, axis)