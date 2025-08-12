import numpy as np
import scipy.ndimage
from typing import Any, Dict
import numba
from numba import jit

class Solver:
    def __init__(self):
        # Pre-define parameters as instance variables
        self.order = 3  # cubic spline interpolation
        self.mode = 'constant'  # padding with 0
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solves the 2D zoom problem using scipy.ndimage.zoom.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "zoomed_image":
                 "zoomed_image": The zoomed image as a list of lists.
        """
        # Direct unpacking for faster access
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        
        # Convert to numpy array with specified dtype for better performance
        image_array = np.asarray(image, dtype=np.float64)
        
        try:
            # Apply zoom with direct parameters
            zoomed_image = scipy.ndimage.zoom(
                image_array,
                zoom_factor,
                order=self.order,
                mode=self.mode
            )
        except Exception:
            # Return empty list on failure
            return {"zoomed_image": []}
        
        # Convert to list for proper output format
        return {"zoomed_image": zoomed_image.tolist()}