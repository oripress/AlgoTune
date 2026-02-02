import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'

    def solve(self, problem, **kwargs):
        image = np.array(problem["image"], dtype=float)
        zoom_factor = problem["zoom_factor"]
        
        # The reference implementation uses scipy.ndimage.zoom
        # We need to match its output exactly (or very closely)
        # order=3 is cubic spline interpolation
        # mode='constant' means padding with 0.0 (default cval is 0.0)
        
        zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=self.order, mode=self.mode)
        
        return {"zoomed_image": zoomed_image}