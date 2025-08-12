import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3  # cubic spline interpolation
        self.mode = 'constant'  # padding with 0
    
    def solve(self, problem, **kwargs):
        """
        Solves the 2D shift problem using scipy.ndimage.shift.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "shifted_image"
        """
        image = np.array(problem["image"], dtype=np.float64)
        shift_vector = problem["shift"]
        
        try:
            shifted_image = scipy.ndimage.shift(
                image, shift_vector, order=self.order, mode=self.mode
            )
        except Exception as e:
            return {"shifted_image": []}
        
        solution = {"shifted_image": shifted_image.tolist()}
        return solution