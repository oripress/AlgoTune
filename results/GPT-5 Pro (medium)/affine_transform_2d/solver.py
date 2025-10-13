from typing import Any, Dict

import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # Match task specification
        self.order = 3
        self.mode = "constant"

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Apply a 2D affine transformation using scipy.ndimage.affine_transform
        with cubic interpolation (order=3) and constant padding.
        Returns a Python list for 'transformed_image' to match validator expectations.
        """
        image = problem["image"]
        matrix = problem["matrix"]

        # Ensure numpy arrays; scipy will handle list inputs but we normalize dtype/contiguity
        img_arr = np.asarray(image, dtype=float)

        try:
            transformed = scipy.ndimage.affine_transform(
                img_arr,
                matrix,
                order=self.order,
                mode=self.mode,
            )
        except Exception:
            # Mirror reference behavior: return empty list on failure
            return {"transformed_image": []}

        # Validator expects a list (not numpy array)
        return {"transformed_image": transformed.tolist()}