import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'
        self.cval = 0.0

    def solve(self, problem, **kwargs):
        image = np.array(problem["image"], dtype=float)
        shift_vector = problem["shift"]
        
        shifted_image = scipy.ndimage.shift(
            image, shift_vector, order=self.order, mode=self.mode, cval=self.cval
        )
        
        return {"shifted_image": shifted_image}