import numpy as np
from scipy.ndimage import affine_transform
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the 2D affine transformation problem by calling the same
        scipy function as the baseline, but with a pre-allocated output array.
        
        Previous attempts failed due to numerical differences:
        1. A JAX implementation failed because it lacks support for cubic
           interpolation (order=3).
        2. Using `np.float32` data types with the SciPy function introduced
           precision errors that failed the `np.allclose` check.

        This implementation uses the default `np.float64` data type to ensure
        numerical identity with the baseline. The sole optimization is
        pre-allocating the output array to avoid an internal memory allocation
        step within `affine_transform`, which provides a small speed boost.
        """
        # Convert input to numpy arrays. Using the default dtype (float64)
        # is crucial for passing the numerical tolerance checks.
        image = np.asarray(problem["image"])
        matrix = np.asarray(problem["matrix"])

        # Pre-allocate the output array. This will have the same dtype as image (float64).
        output_image = np.empty_like(image)

        # Call the same function as the baseline.
        # Using the default dtype and default prefilter=True ensures
        # the result is numerically identical to the baseline.
        affine_transform(
            image,
            matrix,
            output=output_image,
            order=3,
            mode='constant',
            cval=0.0
        )

        # Return a list of lists to avoid a potential validator bug and match output format.
        return {"transformed_image": output_image.tolist()}