import numpy as np
import scipy.ndimage
from numba import njit, prange
import math

POLE = math.sqrt(3.0) - 2.0  # ≈ -0.26794919243112270647
GAIN = (1.0 - POLE) * (1.0 - 1.0/POLE)  # = 6.0

@njit(cache=True)
def _prefilter_1d_rows(data, pole, gain):
    """Apply B-spline prefilter along rows (axis=1) in-place"""
    nrows, ncols = data.shape
    for i in range(nrows):
        # Scale
        for j in range(ncols):
            data[i, j] *= gain
        # Forward (causal) pass: y[j] = x[j] + pole * y[j-1], y[-1] = 0
        for j in range(1, ncols):
            data[i, j] += pole * data[i, j-1]
        # Backward (anticausal) pass: y[j] = y[j] + pole * y[j+1], y[N] = 0
        for j in range(ncols - 2, -1, -1):
            data[i, j] += pole * data[i, j+1]

@njit(cache=True)
def _prefilter_1d_cols(data, pole, gain):
    """Apply B-spline prefilter along columns (axis=0) in-place"""
    nrows, ncols = data.shape
    for j in range(ncols):
        # Scale
        for i in range(nrows):
            data[i, j] *= gain
        # Forward pass
        for i in range(1, nrows):
            data[i, j] += pole * data[i-1, j]
        # Backward pass
        for i in range(nrows - 2, -1, -1):
            data[i, j] += pole * data[i+1, j]

@njit(parallel=True, cache=True)
def _interpolate_shift(coeffs, sr, sc, n_rows, n_cols):
    """Separable cubic B-spline interpolation for uniform shift"""
    cn_rows, cn_cols = coeffs.shape
    
    # Decompose row shift
    kr = int(math.floor(sr))
    fr = sr - kr
    # Decompose col shift
    kc = int(math.floor(sc))
    fc = sc - kc
    
    # Row weights
    wr0 = fr*fr*fr / 6.0
    f1r = 1.0 - fr
    wr1 = 2.0/3.0 - f1r*f1r + 0.5*f1r*f1r*f1r
    wr2 = 2.0/3.0 - fr*fr + 0.5*fr*fr*fr
    wr3 = f1r*f1r*f1r / 6.0
    
    # Col weights
    wc0 = fc*fc*fc / 6.0
    f1c = 1.0 - fc
    wc1 = 2.0/3.0 - f1c*f1c + 0.5*f1c*f1c*f1c
    wc2 = 2.0/3.0 - fc*fc + 0.5*fc*fc*fc
    wc3 = f1c*f1c*f1c / 6.0
    
    # Pass 1: interpolate along rows
    temp = np.zeros((n_rows, cn_cols), dtype=np.float64)
    for row in prange(n_rows):
        i0 = row - kr - 2
        i1 = row - kr - 1
        i2 = row - kr
        i3 = row - kr + 1
        for col in range(cn_cols):
            val = 0.0
            if 0 <= i0 < cn_rows:
                val += wr0 * coeffs[i0, col]
            if 0 <= i1 < cn_rows:
                val += wr1 * coeffs[i1, col]
            if 0 <= i2 < cn_rows:
                val += wr2 * coeffs[i2, col]
            if 0 <= i3 < cn_rows:
                val += wr3 * coeffs[i3, col]
            temp[row, col] = val
    
    # Pass 2: interpolate along columns
    result = np.zeros((n_rows, n_cols), dtype=np.float64)
    for row in prange(n_rows):
        for col in range(n_cols):
            j0 = col - kc - 2
            j1 = col - kc - 1
            j2 = col - kc
            j3 = col - kc + 1
            val = 0.0
            if 0 <= j0 < cn_cols:
                val += wc0 * temp[row, j0]
            if 0 <= j1 < cn_cols:
                val += wc1 * temp[row, j1]
            if 0 <= j2 < cn_cols:
                val += wc2 * temp[row, j2]
            if 0 <= j3 < cn_cols:
                val += wc3 * temp[row, j3]
            result[row, col] = val
    
    return result


class Solver:
    def __init__(self):
        # Warm up numba
        dummy = np.zeros((4, 4), dtype=np.float64)
        _prefilter_1d_rows(dummy.copy(), POLE, GAIN)
        _prefilter_1d_cols(dummy.copy(), POLE, GAIN)
        _interpolate_shift(dummy, 0.5, 0.5, 4, 4)
    
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"], dtype=np.float64)
        shift_vector = problem["shift"]
        sr, sc = float(shift_vector[0]), float(shift_vector[1])
        n_rows, n_cols = image.shape
        
        # Check: does our prefilter + interpolation match scipy's shift?
        # First try without prepadding
        coeffs = image.copy()
        _prefilter_1d_rows(coeffs, POLE, GAIN)
        _prefilter_1d_cols(coeffs, POLE, GAIN)
        result = _interpolate_shift(coeffs, sr, sc, n_rows, n_cols)
        
        # Compare with reference 
        ref = scipy.ndimage.shift(image, [sr, sc], order=3, mode='constant')
        if np.allclose(result, ref, rtol=1e-5, atol=1e-7):
            print("MATCH!")
            return {"shifted_image": result}
        else:
            max_err = np.max(np.abs(result - ref))
            print(f"MISMATCH! max_err={max_err:.6e}")
            
            # Also try with scipy's spline_filter
            coeffs2 = scipy.ndimage.spline_filter(image, order=3, mode='constant')
            result2 = _interpolate_shift(coeffs2, sr, sc, n_rows, n_cols)
            if np.allclose(result2, ref, rtol=1e-5, atol=1e-7):
                print("MATCH with scipy spline_filter!")
            else:
                max_err2 = np.max(np.abs(result2 - ref))
                print(f"MISMATCH with scipy spline_filter too! max_err={max_err2:.6e}")
            
            return {"shifted_image": ref}