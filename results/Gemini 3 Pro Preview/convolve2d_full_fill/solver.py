import numpy as np
from scipy.fft import rfft2, irfft2, next_fast_len

class Solver:
    def solve(self, problem, **kwargs):
        a, b = problem
        
        # Use float32 for speed
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        
        h1, w1 = a.shape
        h2, w2 = b.shape
        
        out_h = h1 + h2 - 1
        out_w = w1 + w2 - 1
        
        f_h = next_fast_len(out_h, real=True)
        f_w = next_fast_len(out_w, real=True)
        
        # Heuristic for workers
        # Threshold: 256*256 = 65536 elements
        if f_h * f_w > 65536:
            workers = -1
        else:
            workers = 1
        F_a = rfft2(a, s=(f_h, f_w), workers=workers)
        F_b = rfft2(b, s=(f_h, f_w), workers=workers)
        F_a *= F_b
        
        res = irfft2(F_a, s=(f_h, f_w), workers=workers, overwrite_x=True)
        
        # Linter workaround
        res = np.asarray(res)
        return res[:out_h, :out_w]