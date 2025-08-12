import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # match reference parameters
        self.order = 3
        self.mode = "constant"

    def solve(self, problem, **kwargs):
        """
        Zoom a 2D image using cubic spline interpolation (order=3) and constant boundary mode.
        Expects problem to be a dict with keys "image" (list of lists or array-like) and
        "zoom_factor" (float). Returns {"zoomed_image": list}.
        """
        # Basic validation / conversions
        if not isinstance(problem, dict):
            return {"zoomed_image": []}

        if "image" not in problem or "zoom_factor" not in problem:
            return {"zoomed_image": []}

        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        try:
            arr = np.asarray(image, dtype=float)
        except Exception:
            return {"zoomed_image": []}

        # Handle degenerate cases
        try:
            z = float(zoom_factor)
        except Exception:
            return {"zoomed_image": []}

        # If empty input or zero zoom, return empty list to indicate no output
        if arr.size == 0 or z == 0.0:
            return {"zoomed_image": []}

        try:
            zoomed = scipy.ndimage.zoom(arr, z, order=self.order, mode=self.mode)
        except Exception:
            return {"zoomed_image": []}

        # Convert to Python nested lists for validation compatibility
        try:
            zoom_list = zoomed.tolist()
        except Exception:
            # Fallback: ensure we return a list
            try:
                zoom_list = [float(zoomed)]
            except Exception:
                zoom_list = []

        return {"zoomed_image": zoom_list}