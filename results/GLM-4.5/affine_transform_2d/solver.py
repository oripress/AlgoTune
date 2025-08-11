import numpy as np
import scipy.ndimage
from typing import Any

class Solver:
    def __init__(self):
        self.order = 3  # cubic spline interpolation
        self.mode = 'constant'  # padding with 0
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the 2D affine transformation problem using scipy.ndimage.affine_transform.

        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "transformed_image":
                 "transformed_image": The transformed image as a list of lists.
        """
        image = problem["image"]
        matrix = problem["matrix"]

        # Perform affine transformation
        try:
            # output_shape can be specified, default is same as input
            transformed_image = scipy.ndimage.affine_transform(
                image, matrix, order=self.order, mode=self.mode, cval=0.0, prefilter=True, output=np.float64
            )
        except Exception as e:
            # Return an empty list to indicate failure? Adjust based on benchmark policy.
            return {"transformed_image": []}

        solution = {"transformed_image": transformed_image.tolist()}
        return solution