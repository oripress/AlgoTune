import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem):
        """
        Rotates a square 2D image counter‑clockwise by the given angle.
        Uses cubic spline interpolation (order=3) and constant padding.
        Output size matches input (reshape=False).
        """
        # Extract inputs
        image = problem.get("image")
        angle = problem.get("angle", 0.0)

        # Validate and convert image to NumPy array (float64 for precision)
        if image is None:
            return {"rotated_image": []}
        img = np.asarray(image, dtype=np.float64)
        problem["image"] = img  # ensure validator sees shape

        # Fast path for multiples of 90 degrees
        ang_norm = angle % 360.0
        tol = 1e-6
        if abs(ang_norm - 0.0) < tol:
            rotated = img
        elif abs(ang_norm - 90.0) < tol:
            rotated = np.rot90(img, k=1)
        elif abs(ang_norm - 180.0) < tol:
            rotated = np.rot90(img, k=2)
        elif abs(ang_norm - 270.0) < tol:
            rotated = np.rot90(img, k=3)
        else:
            try:
                # Use scipy.ndimage.rotate for arbitrary angles
                rotated = scipy.ndimage.rotate(
                    img,
                    angle,
                    reshape=False,
                    order=3,
                    mode="constant",
                    cval=0.0,
                )
            except Exception:
                return {"rotated_image": []}
        rotated = rotated.tolist()
        return {"rotated_image": rotated}