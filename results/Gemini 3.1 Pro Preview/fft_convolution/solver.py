import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

class Solver:
    def solve(self, problem, **kwargs):
        x = problem["signal_x"]
        y = problem["signal_y"]
        mode = problem.get("mode", "full")
        
        len_x = len(x)
        len_y = len(y)
        
        if len_x == 0 or len_y == 0:
            return {"convolution": []}
            
        shape = len_x + len_y - 1
        fshape = next_fast_len(shape, real=True)
        
        sp1 = rfft(x, fshape)
        sp2 = rfft(y, fshape)
        sp1 *= sp2
        
        ret = irfft(sp1, fshape)
        ret = ret[:shape]
        
        if mode == "full":
            pass
        elif mode == "same":
            start = (len_y - 1) // 2
            ret = ret[start:start + len_x]
        elif mode == "valid":
            start = min(len_x, len_y) - 1
            end = max(len_x, len_y)
            ret = ret[start:end]
            
        return {"convolution": ret}