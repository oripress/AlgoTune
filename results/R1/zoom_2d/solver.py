import scipy.ndimage
from typing import Any

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'

    def solve(self, problem, **kwargs) -> Any:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        
        try:
            zoomed_image = scipy.ndimage.zoom(
                image, 
                zoom_factor, 
                order=self.order, 
                mode=self.mode
            )
            return {"zoomed_image": zoomed_image.tolist()}
        except Exception as e:
            return {"zoomed_image": []}