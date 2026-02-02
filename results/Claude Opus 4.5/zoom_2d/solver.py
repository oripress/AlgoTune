import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'
    
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"], dtype=np.float64)
        zoom_factor = problem["zoom_factor"]
        
        zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=self.order, mode=self.mode)
        
        return {"zoomed_image": zoomed_image}