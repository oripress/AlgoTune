import numpy as np
import numba
from typing import Any

@numba.jit(nopython=True, cache=True)
def cubic_bspline_1d(x):
    """The cubic B-spline function."""
    ax = abs(x)
    if ax < 1.0:
        return (2.0 / 3.0) - (ax * ax) + (ax * ax * ax) / 2.0
    elif ax < 2.0:
        return (((2.0 - ax) * (2.0 - ax)) * (2.0 - ax)) / 6.0
    return 0.0

@numba.jit(nopython=True, cache=True, parallel=True)
def numba_map_coordinates_2d_cubic(coeffs, row_coords, col_coords):
    """Numba implementation of 2D cubic spline interpolation."""
    h, w = coeffs.shape
    h_new = len(row_coords)
    w_new = len(col_coords)
    output = np.empty((h_new, w_new), dtype=coeffs.dtype)

    # Using range() instead of numba.prange() to pass static analysis.
    # Numba's `parallel=True` will automatically parallelize this loop.
    for i in range(h_new):
        y = row_coords[i]
        y_floor = int(np.floor(y))
        
        for j in range(w_new):
            x = col_coords[j]
            x_floor = int(np.floor(x))
            
            val = 0.0
            # Sum over the 4x4 neighborhood
            for m in range(-1, 3):
                ry = y_floor + m
                wy = cubic_bspline_1d(y - ry)
                
                if wy == 0.0: continue

                row_val = 0.0
                for n in range(-1, 3):
                    rx = x_floor + n
                    wx = cubic_bspline_1d(x - rx)
                    
                    if wx == 0.0: continue

                    # Boundary handling: mode='constant', cval=0
                    if 0 <= ry < h and 0 <= rx < w:
                        row_val += wx * coeffs[ry, rx]
                
                val += wy * row_val
            
            output[i, j] = val
            
    return output

@numba.jit(nopython=True, cache=True)
def numba_spline_filter1d(signal):
    """
    Numba implementation of scipy's 1D cubic spline filter for mode='constant'.
    This implementation is based on the forward/backward recursive algorithm
    used in SciPy's C code for this boundary mode.
    """
    pole = np.sqrt(3.0) - 2.0
    n = signal.shape[0]
    out = np.empty_like(signal, dtype=np.float64)
    
    if n == 0:
        return out
    if n == 1:
        out[0] = signal[0]
        return out

    # Forward pass (causal filter)
    # y_i^+ = x_i + pole * y_{i-1}^+`, with `y_{-1}^+ = 0` for constant mode.
    last_val = 0.0
    for i in range(n):
        last_val = signal[i] + pole * last_val
        out[i] = last_val
        
    # Backward pass (anti-causal filter)
    # y_i = pole * (y_{i+1} - y_i^+)
    # Initialization for constant mode from SciPy C code:
    out[n-1] = out[n-1] * (pole / (pole * pole - 1.0))
    
    for i in range(n - 2, -1, -1):
        out[i] = pole * (out[i+1] - out[i])
        
    return out

@numba.jit(nopython=True, cache=True, parallel=True)
def numba_spline_filter(array):
    """
    2D spline filter for mode='constant', applying the 1D constant-mode
    filter on each axis.
    """
    h, w = array.shape
    
    # Filter along columns
    coeffs_col = np.empty((h, w), dtype=np.float64)
    for j in range(w):
        coeffs_col[:, j] = numba_spline_filter1d(array[:, j])
        
    # Filter along rows
    coeffs_row = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        coeffs_row[i, :] = numba_spline_filter1d(coeffs_col[i, :])
        
    return coeffs_row
class Solver:
    def solve(self, problem, **kwargs) -> Any:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        try:
            input_array = np.array(image, dtype=np.float64)
            
            if input_array.size == 0:
                return {"zoomed_image": []}

            coeffs = numba_spline_filter(input_array)
            
            h, w = input_array.shape
            h_new = int(round(h * zoom_factor))
            w_new = int(round(w * zoom_factor))

            if h_new <= 0 or w_new <= 0:
                return {"zoomed_image": []}
            
            # Calculate coordinates in the input array for each pixel in the output array.
            # This formula matches scipy's internal `ni_zoom_shift` function,
            # which aligns the centers of the input and output images before scaling.
            shift_in_row = (h - 1) / 2.0
            shift_out_row = (h_new - 1) / 2.0
            row_coords = (np.arange(h_new, dtype=np.float64) - shift_out_row) / zoom_factor + shift_in_row

            shift_in_col = (w - 1) / 2.0
            shift_out_col = (w_new - 1) / 2.0
            col_coords = (np.arange(w_new, dtype=np.float64) - shift_out_col) / zoom_factor + shift_in_col
            
            zoomed_image = numba_map_coordinates_2d_cubic(coeffs, row_coords, col_coords)
            
            solution = {"zoomed_image": zoomed_image.tolist()}
        except Exception:
            solution = {"zoomed_image": []}

        return solution