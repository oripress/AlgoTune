import numpy as np
from scipy.ndimage import rotate

class Solver:
    def solve(self, problem, **kwargs):
        """
        Rotate a 2D image counter-clockwise by a specified angle using
        cubic spline interpolation (order=3) and constant boundary padding.
        """
        # Extract inputs
        image = problem["image"]
        angle = problem["angle"]

        # Convert to numpy array
        arr = np.asarray(image, dtype=float)

        # Use SciPy's C-optimized rotation
        rotated = rotate(
            arr,
            angle,
            reshape=False,
            order=3,
            mode='constant',
            cval=0.0,
            prefilter=True
        )

        # Return result as nested Python lists
        return {"rotated_image": rotated.tolist()}