import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # interpolation order (cubic) and constant mode for boundaries
        self.order = 3
        self.mode = 'constant'

    def solve(self, problem, **kwargs):
        """
        Shift a 2D image by a subpixel amount using cubic spline interpolation.
        Parameters
        ----------
        problem : dict
            Contains:
                "image": list of list of floats, shape (n, n)
                "shift": [shift_row, shift_col] as floats

        Returns
        -------
        dict
            {"shifted_image": list of list of floats}
        """
        # Extract data
        image = problem.get("image")
        shift_vector = problem.get("shift", [0.0, 0.0])

        # Convert to NumPy array for processing (use float64 for accuracy)
        img_arr = np.asarray(image, dtype=float)
        # Perform shift with cubic spline (order=3) and constant zero padding.
        # prefilter=True is required for correct cubic interpolation.
        shifted = scipy.ndimage.shift(
            img_arr,
            shift=shift_vector,
            order=self.order,
            mode=self.mode,
            cval=0.0,
            prefilter=True,
        )

        # Convert to plain Python list of lists for validation compatibility
        return {"shifted_image": shifted.tolist()}