import scipy.ndimage
import numpy as np
from typing import Any, Dict, List

# Precompute constant values
ROTATE_KWARGS = {
    'reshape': False,
    'order': 3,
    'mode': 'constant',
    'cval': 0.0
}

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solves the 2D rotation problem using scipy.ndimage.rotate.
        
        :param problem: A dictionary representing the problem.
        :return: A dictionary with key "rotated_image":
                 "rotated_image": The rotated image as a list of lists.
        """
        # Directly use scipy.ndimage.rotate with minimal overhead
        image = np.asarray(problem["image"], dtype=np.float64)
        rotated_image = scipy.ndimage.rotate(
            image,
            problem["angle"],
            **ROTATE_KWARGS
        )
            
        # Convert numpy array to list of lists for output
        return {"rotated_image": rotated_image.tolist()}