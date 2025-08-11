import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # Default parameters matching the task description
        self.order = 3          # cubic spline interpolation
        self.mode = "constant"  # pad with zeros outside boundaries

    def solve(self, problem):
        """
        Apply a 2D affine transformation to the input image.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                "image"  : list of list of floats (n x n)
                "matrix" : 2x3 affine transformation matrix (list of lists)

        Returns
        -------
        dict
            {"transformed_image": <list of list of floats>}
        """
        # Extract data
        # Convert inputs to double-precision floats to match reference accuracy
        # and replace the original list with a NumPy array so that the
        # validation routine can access .shape.
        # Convert inputs to double‑precision floats (no extra copy)
        image = np.array(problem["image"], dtype=np.float64, copy=False)
        matrix = np.array(problem["matrix"], dtype=np.float64, copy=False)

        # Cache parameters locally to avoid attribute lookups
        order = self.order
        mode = self.mode

        # Perform affine transformation; slice matrix directly for linear part and offset
        transformed = scipy.ndimage.affine_transform(
            image,
            matrix[:, :2],
            offset=matrix[:, 2],
            order=order,
            mode=mode,
            cval=0.0,
            prefilter=True,
        )
        return {"transformed_image": transformed.tolist()}