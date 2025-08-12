import scipy.ndimage
import numpy as np
import json

class Solver:
    def __init__(self):
        # cubic spline (order=3) with constant padding
        self.order = 3
        self.mode = 'constant'

    def solve(self, problem, **kwargs):
        """
        Zoom a 2‑D image using cubic spline interpolation.

        Parameters
        ----------
        problem : dict
            Must contain:
            - "image": list of list of floats (shape n×n)
            - "zoom_factor": float scaling factor

        Returns
        -------
        dict
            {"zoomed_image": list of list of floats}
        """
        image = problem.get("image")
        zoom_factor = problem.get("zoom_factor")

        # Validate inputs quickly
        if image is None or zoom_factor is None:
            return {"zoomed_image": []}

        # Convert to a NumPy array (float64) for accurate cubic spline interpolation
        arr = np.asarray(image, dtype=np.float64)

        try:
            # Perform zoom with default prefilter (required for accurate cubic spline)
            zoomed = scipy.ndimage.zoom(
                arr,
                zoom_factor,
                order=self.order,
                mode=self.mode,
            )
        except Exception:
            # Any failure yields an empty result as per reference behaviour
            return {"zoomed_image": []}

        # Convert result back to plain Python lists
        return {"zoomed_image": zoomed.tolist()}