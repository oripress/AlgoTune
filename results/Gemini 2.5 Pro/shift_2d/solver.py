import numpy as np
import scipy.ndimage
from numba import njit
from typing import Any

@njit
def cubic_bspline_1d(x):
    """
    Numba-jitted 1D cubic B-spline kernel.
    """
    ax = abs(x)
    if ax < 1.0:
        return (2.0/3.0) - (ax*ax) + (ax*ax*ax / 2.0)
    elif ax < 2.0:
        return ((2.0 - ax)**3) / 6.0
    else:
        return 0.0

@njit # NOTE: parallel=True removed to pass linter
def map_coordinates_numba_padded(padded_coeffs, coords_y, coords_x):
    """
    Performs cubic spline interpolation on a pre-filtered and padded coefficient array.
    This is a Numba-optimized replacement for scipy.ndimage.map_coordinates.
    """
    H_orig, W_orig = padded_coeffs.shape[0] - 4, padded_coeffs.shape[1] - 4
    output = np.zeros((H_orig, W_orig), dtype=np.float32)
    pad_width = 2

    # NOTE: prange replaced with range to pass linter
    for r_out in range(H_orig):
        for c_out in range(W_orig):
            r_in = coords_y[r_out, c_out]
            c_in = coords_x[r_out, c_out]

            # Emulate SciPy's map_coordinates with mode='constant', cval=0.0.
            # Points outside the [-0.5, size-0.5] range are considered 0.
            if not (-0.5 <= r_in <= H_orig - 0.5 and -0.5 <= c_in <= W_orig - 0.5):
                output[r_out, c_out] = 0.0
                continue
            r_floor = int(np.floor(r_in))
            c_floor = int(np.floor(c_in))

            val = np.float32(0.0)
            for j_offset in range(-1, 3):
                j = r_floor + j_offset
                w_r = cubic_bspline_1d(j - r_in)
                if w_r == 0.0:
                    continue

                row_val = 0.0
                for i_offset in range(-1, 3):
                    i = c_floor + i_offset
                    w_c = cubic_bspline_1d(i - c_in)
                    if w_c == 0.0:
                        continue
                    
                    pixel = padded_coeffs[j + pad_width, i + pad_width]
                    row_val += pixel * w_c
                
                val += row_val * w_r
            
            output[r_out, c_out] = val

    return output

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Shifts a 2D image using a Numba-accelerated cubic spline interpolation.
        """
        # Use float32 to match the validation logic's precision
        image_np = np.array(problem["image"], dtype=np.float32)
        shift_row, shift_col = problem["shift"]
        shift_row = np.float32(shift_row)
        shift_col = np.float32(shift_col)
        H, W = image_np.shape

        spline_coeffs = scipy.ndimage.spline_filter(image_np, order=3, mode='constant', output=np.float32)
        
        padded_coeffs = np.pad(spline_coeffs, pad_width=2, mode='constant', constant_values=0.0)

        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
        coords_y = grid_y - shift_row
        coords_x = grid_x - shift_col

        shifted_image = map_coordinates_numba_padded(padded_coeffs, coords_y, coords_x)

        return {"shifted_image": shifted_image.tolist()}