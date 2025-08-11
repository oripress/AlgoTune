import numpy as np
from scipy.ndimage import affine_transform
from typing import Dict, Any

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solves the 2D affine transformation problem using scipy.ndimage.affine_transform.

        :param problem: A dictionary representing the problem with 'image' and 'matrix' keys.
        :return: A dictionary with key "transformed_image": the transformed image as a list of lists.
        """
        # Extract image and transformation matrix from the problem
        image = np.asarray(problem["image"], dtype=np.float64, order='C')
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        
        # Pre-allocate output array for better performance
        output = np.empty_like(image, dtype=np.float64, order='C')
        
        # Perform affine transformation with pre-allocated output
        affine_transform(
            image,
            matrix,
            output=output,
            order=3,  # Cubic spline interpolation
            mode='constant',  # Padding with 0
            cval=0.0,
            prefilter=True  # Enable prefiltering for better accuracy
        )
        
        # Convert to list with optimized method
        return {"transformed_image": output.tolist()}