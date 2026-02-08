import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"], dtype=np.float64)
        zoom_factor = float(problem["zoom_factor"])
        zoomed = scipy.ndimage.zoom(image, zoom_factor, order=3, mode='constant')
        return {"zoomed_image": zoomed}