import numpy as np
import scipy.ndimage
from typing import Any

class Solver:
    def __init__(self):
        # Pre-set parameters for the zoom operation
        self.order = 3  # Cubic spline interpolation
        self.mode = 'constant'  # Constant boundary mode (padding with 0)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the 2D zoom problem using scipy.ndimage.zoom.
        
        :param problem: A dictionary with keys "image" and "zoom_factor"
        :return: A dictionary with key "zoomed_image"
        """
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        
        try:
            # Convert to numpy array for processing
            image_array = np.asarray(image, dtype=np.float64)
            
            # Apply zoom
            zoomed_image = scipy.ndimage.zoom(
                image_array, 
                zoom_factor, 
                order=self.order, 
                mode=self.mode
            )
            
            # Convert back to list of lists
            solution = {"zoomed_image": zoomed_image.tolist()}
        except Exception:
            solution = {"zoomed_image": []}
        
        return solution