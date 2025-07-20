import numpy as np
from scipy import ndimage

class Solver:
    def __init__(self):
        # using cubic spline interpolation and constant (zero) padding
        self.order = 3
        self.mode = 'constant'
        self.cval = 0.0

    def solve(self, problem, **kwargs):
        """
        Shift a 2D image by a subpixel vector using cubic spline interpolation.
        Expects problem = {"image": list[list[float]], "shift": [float, float]}
        Returns {"shifted_image": np.ndarray}
        """
        # Extract inputs
        image = problem.get("image")
        shift = problem.get("shift")
        if image is None or shift is None:
            return {"shifted_image": []}

        # Convert to numpy array
        arr = np.asarray(image, dtype=float)
        # Efficient shift using SciPy's C implementation
        shifted = ndimage.shift(
            arr,
            shift,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=True
        )
        return {"shifted_image": list(shifted)}