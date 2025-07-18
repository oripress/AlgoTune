import numpy as np
import scipy.ndimage
from typing import Any

class Solver:
    def __init__(self):
        self.order = 3  # cubic spline interpolation
        self.mode = 'constant'  # padding with 0
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the 2D zoom problem using scipy.ndimage.zoom.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "zoomed_image":
                 "zoomed_image": The zoomed image as a list of lists.
        """
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        
        try:
            zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=self.order, mode=self.mode)
        except Exception as e:
            return {"zoomed_image": []}  # Indicate failure
        
        solution = {"zoomed_image": zoomed_image.tolist()}
        return solution