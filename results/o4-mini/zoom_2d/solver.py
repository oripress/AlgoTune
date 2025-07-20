import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem, **kwargs):
        """
        Zoom a 2D image using scipy.ndimage.zoom with cubic spline interpolation (order=3)
        and constant padding (cval=0.0).
        Returns list of numpy arrays (rows) for fast output conversion.
        """
        try:
            arr = np.asarray(problem["image"], dtype=float)
            zoomed = scipy.ndimage.zoom(
                arr, problem["zoom_factor"],
                order=3, mode='constant', cval=0.0
            )
            # Return list of rows (numpy arrays) to avoid deep Python float conversion
            return {"zoomed_image": list(zoomed)}
        except Exception:
            return {"zoomed_image": []}