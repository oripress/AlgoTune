import numpy as np
import scipy.ndimage
from typing import Any

class Solver:
    def __init__(self):
        self.reshape = False
        self.order = 3
        self.mode = 'constant'
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the 2D rotation problem using scipy.ndimage.rotate.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "rotated_image":
                 "rotated_image": The rotated image as a list of lists.
        """
        image = problem["image"]
        angle = problem["angle"]
        
        try:
            rotated_image = scipy.ndimage.rotate(
                image, angle, reshape=self.reshape, order=self.order, mode=self.mode
            )
        except Exception as e:
            return {"rotated_image": []}  # Indicate failure
        
        solution = {"rotated_image": rotated_image.tolist()}
        return solution