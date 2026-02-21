import scipy.ndimage
import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        
        try:
            image_arr = np.asarray(image, dtype=np.float64)
            zoom_seq = (zoom_factor, zoom_factor)
            output_shape = tuple([int(round(s * z)) for s, z in zip(image_arr.shape, zoom_seq)])
            
            output = np.empty(output_shape, dtype=np.float64)
            scipy.ndimage.zoom(image_arr, zoom_factor, order=self.order, mode=self.mode, output=output)
        except Exception:
            return {"zoomed_image": []}
            
        return {"zoomed_image": output}