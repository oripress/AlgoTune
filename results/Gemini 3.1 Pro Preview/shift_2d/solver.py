import scipy.ndimage
import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'
        self.out = None

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        image = problem["image"]
        shift = problem["shift"]
        
        h = len(image)
        w = len(image[0]) if h > 0 else 0
        
        if self.out is None or self.out.shape != (h, w):
            self.out = np.empty((h, w), dtype=np.float64)
            
        res = scipy.ndimage.shift(image, shift, order=3, mode='constant', output=self.out)
        
        return {"shifted_image": res}