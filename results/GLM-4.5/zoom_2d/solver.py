import numpy as np
from typing import Any, Dict, List
import scipy.ndimage

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def cubic_interpolate_1d(p, x):
    """
    1D cubic interpolation using Catmull-Rom spline.
    p: array of 4 points
    x: interpolation position (between 0 and 1)
    """
    # Catmull-Rom spline coefficients
    a0 = p[1]
    a1 = -0.5 * p[0] + 0.5 * p[2]
    a2 = p[0] - 2.5 * p[1] + 2 * p[2] - 0.5 * p[3]
    a3 = -0.5 * p[0] + 1.5 * p[1] - 1.5 * p[2] + 0.5 * p[3]
    
    return a0 + a1 * x + a2 * x * x + a3 * x * x * x

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def cubic_interpolate_2d_numba(image, x_coords, y_coords, result):
        """
        2D cubic interpolation using separable 1D interpolation - numba optimized.
        """
        height, width = image.shape
        
        # For each coordinate
        for i in range(x_coords.shape[0]):
            for j in range(x_coords.shape[1]):
                x = x_coords[i, j] + 2  # Account for padding
                y = y_coords[i, j] + 2
                
                # Get integer and fractional parts
                x_int = int(x)
                y_int = int(y)
                x_frac = x - x_int
                y_frac = y - y_int
                
                # Get 4x4 neighborhood
                x_start = x_int - 1
                y_start = y_int - 1
                
                # Interpolate along rows first
                row_values = np.zeros(4)
                for k in range(4):
                    col_values = image[y_start + k, x_start:x_start + 4]
                    # Catmull-Rom spline coefficients
                    a0 = col_values[1]
                    a1 = -0.5 * col_values[0] + 0.5 * col_values[2]
                    a2 = col_values[0] - 2.5 * col_values[1] + 2 * col_values[2] - 0.5 * col_values[3]
                    a3 = -0.5 * col_values[0] + 1.5 * col_values[1] - 1.5 * col_values[2] + 0.5 * col_values[3]
                    row_values[k] = a0 + a1 * x_frac + a2 * x_frac * x_frac + a3 * x_frac * x_frac * x_frac
                
                # Interpolate along columns
                a0 = row_values[1]
                a1 = -0.5 * row_values[0] + 0.5 * row_values[2]
                a2 = row_values[0] - 2.5 * row_values[1] + 2 * row_values[2] - 0.5 * row_values[3]
                a3 = -0.5 * row_values[0] + 1.5 * row_values[1] - 1.5 * row_values[2] + 0.5 * row_values[3]
                result[i, j] = a0 + a1 * y_frac + a2 * y_frac * y_frac + a3 * y_frac * y_frac * y_frac

def cubic_interpolate_2d(image, x_coords, y_coords):
    """
    2D cubic interpolation using separable 1D interpolation.
    """
    height, width = image.shape
    
    # Pad image for boundary conditions
    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='constant', constant_values=0)
    
    # Initialize result array
    result = np.zeros_like(x_coords)
    
    if NUMBA_AVAILABLE:
        cubic_interpolate_2d_numba(padded_image, x_coords, y_coords, result)
    else:
        # For each coordinate
        for i in range(x_coords.shape[0]):
            for j in range(x_coords.shape[1]):
                x = x_coords[i, j] + 2  # Account for padding
                y = y_coords[i, j] + 2
                
                # Get integer and fractional parts
                x_int = int(x)
                y_int = int(y)
                x_frac = x - x_int
                y_frac = y - y_int
                
                # Get 4x4 neighborhood
                x_start = x_int - 1
                y_start = y_int - 1
                
                # Interpolate along rows first
                row_values = np.zeros(4)
                for k in range(4):
                    col_values = padded_image[y_start + k, x_start:x_start + 4]
                    row_values[k] = cubic_interpolate_1d(col_values, x_frac)
                
                # Interpolate along columns
                result[i, j] = cubic_interpolate_1d(row_values, y_frac)
    
    return result

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Solves the 2D zoom problem using manual cubic interpolation with numba acceleration.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "zoomed_image":
                 "zoomed_image": The zoomed image as a list of lists.
        """
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        try:
            # Convert to numpy array
            image_array = np.asarray(image, dtype=np.float64)
            input_shape = image_array.shape
            
            # Calculate output shape using scipy's method
            output_shape = scipy.ndimage.zoom._get_output_shape(image_array, zoom_factor)
            
            if output_shape[0] == 0 or output_shape[1] == 0:
                return {"zoomed_image": []}
            
            # Create coordinate grids for the output
            y_indices = np.arange(output_shape[0], dtype=np.float64)
            x_indices = np.arange(output_shape[1], dtype=np.float64)
            
            # Map output coordinates to input coordinates
            y_coords = (y_indices / zoom_factor).reshape(-1, 1)
            x_coords = (x_indices / zoom_factor).reshape(1, -1)
            
            # Create meshgrid using broadcasting
            coords_y = np.broadcast_to(y_coords, (output_shape[0], output_shape[1]))
            coords_x = np.broadcast_to(x_coords, (output_shape[0], output_shape[1]))
            
            # Use manual cubic interpolation
            zoomed_image = cubic_interpolate_2d(image_array, coords_x, coords_y)
            
            # Convert to list
            zoomed_image_list = zoomed_image.tolist()
        except Exception as e:
            # Fallback to scipy.zoom
            try:
                zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=3, mode='constant')
                zoomed_image_list = zoomed_image.tolist()
            except:
                return {"zoomed_image": []}

        solution = {"zoomed_image": zoomed_image_list}
        return solution