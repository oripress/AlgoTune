import numpy as np
from scipy.ndimage import shift

class Solver:
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"])
        shift_vector = problem["shift"]
        shifted_image = shift(image, shift_vector, order=3, mode='constant')
        return {"shifted_image": shifted_image}