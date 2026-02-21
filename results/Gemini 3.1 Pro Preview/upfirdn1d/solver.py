import numpy as np
from scipy.signal._upfirdn import _UpFIRDn

class Solver:
    def solve(self, problem: list) -> list:
        results = []
        for h, x, up, down in problem:
            h_arr = np.asarray(h)
            x_arr = np.asarray(x)
            
            dt = np.promote_types(h_arr.dtype, x_arr.dtype)
            if dt.char not in 'fdgFDG':
                dt = np.dtype(np.float64)
                
            upfirdn_obj = _UpFIRDn(h_arr, dt, up, down)
            res = upfirdn_obj.apply_filter(x_arr)
            results.append(res)
        return results