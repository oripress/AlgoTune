import numpy as np
import numba as nb
from scipy.ndimage import spline_filter


@nb.njit(parallel=True, cache=True)
def _affine_interp_cubic(coeffs, a00, a01, a10, a11, off0, off1, n0, n1):
    out = np.empty((n0, n1), dtype=np.float64)
    cn0 = coeffs.shape[0]
    cn1 = coeffs.shape[1]
    
    for i in nb.prange(n0):
        base_x = a00 * i + off0
        base_y = a10 * i + off1
        
        for j in range(n1):
            x = base_x + a01 * j
            y = base_y + a11 * j
            
            ix = int(np.floor(x))
            iy = int(np.floor(y))
            
            fx = x - ix
            fy = y - iy
            
            # B-spline weights for x direction
            fx2 = fx * fx
            fx3 = fx2 * fx
            omfx = 1.0 - fx
            wx0 = omfx * omfx * omfx / 6.0
            wx1 = (4.0 - 6.0 * fx2 + 3.0 * fx3) / 6.0
            wx2 = (1.0 + 3.0 * fx + 3.0 * fx2 - 3.0 * fx3) / 6.0
            wx3 = fx3 / 6.0
            
            # B-spline weights for y direction
            fy2 = fy * fy
            fy3 = fy2 * fy
            omfy = 1.0 - fy
            wy0 = omfy * omfy * omfy / 6.0
            wy1 = (4.0 - 6.0 * fy2 + 3.0 * fy3) / 6.0
            wy2 = (1.0 + 3.0 * fy + 3.0 * fy2 - 3.0 * fy3) / 6.0
            wy3 = fy3 / 6.0
            
            r0 = ix - 1
            c0 = iy - 1
            
            val = 0.0
            
            if r0 >= 0 and r0 + 3 < cn0 and c0 >= 0 and c0 + 3 < cn1:
                val = (wx0 * (wy0 * coeffs[r0, c0] + wy1 * coeffs[r0, c0+1] +
                              wy2 * coeffs[r0, c0+2] + wy3 * coeffs[r0, c0+3]) +
                       wx1 * (wy0 * coeffs[r0+1, c0] + wy1 * coeffs[r0+1, c0+1] +
                              wy2 * coeffs[r0+1, c0+2] + wy3 * coeffs[r0+1, c0+3]) +
                       wx2 * (wy0 * coeffs[r0+2, c0] + wy1 * coeffs[r0+2, c0+1] +
                              wy2 * coeffs[r0+2, c0+2] + wy3 * coeffs[r0+2, c0+3]) +
                       wx3 * (wy0 * coeffs[r0+3, c0] + wy1 * coeffs[r0+3, c0+1] +
                              wy2 * coeffs[r0+3, c0+2] + wy3 * coeffs[r0+3, c0+3]))
            else:
                if 0 <= r0 < cn0:
                    if 0 <= c0 < cn1:
                        val += wx0 * wy0 * coeffs[r0, c0]
                    if 0 <= c0 + 1 < cn1:
                        val += wx0 * wy1 * coeffs[r0, c0 + 1]
                    if 0 <= c0 + 2 < cn1:
                        val += wx0 * wy2 * coeffs[r0, c0 + 2]
                    if 0 <= c0 + 3 < cn1:
                        val += wx0 * wy3 * coeffs[r0, c0 + 3]
                if 0 <= r0 + 1 < cn0:
                    if 0 <= c0 < cn1:
                        val += wx1 * wy0 * coeffs[r0 + 1, c0]
                    if 0 <= c0 + 1 < cn1:
                        val += wx1 * wy1 * coeffs[r0 + 1, c0 + 1]
                    if 0 <= c0 + 2 < cn1:
                        val += wx1 * wy2 * coeffs[r0 + 1, c0 + 2]
                    if 0 <= c0 + 3 < cn1:
                        val += wx1 * wy3 * coeffs[r0 + 1, c0 + 3]
                if 0 <= r0 + 2 < cn0:
                    if 0 <= c0 < cn1:
                        val += wx2 * wy0 * coeffs[r0 + 2, c0]
                    if 0 <= c0 + 1 < cn1:
                        val += wx2 * wy1 * coeffs[r0 + 2, c0 + 1]
                    if 0 <= c0 + 2 < cn1:
                        val += wx2 * wy2 * coeffs[r0 + 2, c0 + 2]
                    if 0 <= c0 + 3 < cn1:
                        val += wx2 * wy3 * coeffs[r0 + 2, c0 + 3]
                if 0 <= r0 + 3 < cn0:
                    if 0 <= c0 < cn1:
                        val += wx3 * wy0 * coeffs[r0 + 3, c0]
                    if 0 <= c0 + 1 < cn1:
                        val += wx3 * wy1 * coeffs[r0 + 3, c0 + 1]
                    if 0 <= c0 + 2 < cn1:
                        val += wx3 * wy2 * coeffs[r0 + 3, c0 + 2]
                    if 0 <= c0 + 3 < cn1:
                        val += wx3 * wy3 * coeffs[r0 + 3, c0 + 3]
            
            out[i, j] = val
    
    return out


# Warm up JIT at import time
_dummy = np.zeros((4, 4), dtype=np.float64)
_affine_interp_cubic(_dummy, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4, 4)


class Solver:
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"], dtype=np.float64)
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        
        a00, a01, off0 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        a10, a11, off1 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        
        coeffs = spline_filter(image, order=3, mode='constant')
        n0, n1 = image.shape
        result = _affine_interp_cubic(coeffs, a00, a01, a10, a11, off0, off1, n0, n1)
        
        return {"transformed_image": result}