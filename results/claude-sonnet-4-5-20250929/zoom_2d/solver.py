import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'
    
    def solve(self, problem, **kwargs):
        """
        Solves the 2D zoom problem using scipy.ndimage.zoom.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "zoomed_image"
        """
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        
        try:
            zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=self.order, mode=self.mode)
        except Exception as e:
            return {"zoomed_image": []}
        
        solution = {"zoomed_image": zoomed_image.tolist()}
        return solution