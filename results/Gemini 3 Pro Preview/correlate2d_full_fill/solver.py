import numpy as np
from scipy import fft

class Solver:
    def solve(self, problem, **kwargs):
        a_in, b_in = problem
        a = np.asarray(a_in, dtype=np.float32)
        b = np.asarray(b_in, dtype=np.float32)
        
        h1, w1 = a.shape
        h2, w2 = b.shape
        
        out_h = int(h1 + h2 - 1)
        out_w = int(w1 + w2 - 1)
        
        fh = int(fft.next_fast_len(out_h, real=True))
        fw = int(fft.next_fast_len(out_w, real=True))
        
        if fh * fw > 100000:
            workers = -1
        else:
            workers = 1
            
        b_flipped = np.ascontiguousarray(b[::-1, ::-1])
        
        sp_a = fft.rfft2(a, (fh, fw), workers=workers)
        sp_b = fft.rfft2(b_flipped, (fh, fw), workers=workers)
        
        sp_a *= sp_b
        
        ret = fft.irfft2(sp_a, (fh, fw), workers=workers, overwrite_x=True)
        ret = np.asarray(ret)
        
        return ret[:out_h, :out_w]