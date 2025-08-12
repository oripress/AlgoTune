from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self) -> None:
        self.order = 3
        self.mode = "constant"
        self.reshape = False
        self.cval = 0.0

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            image = problem["image"]
            angle = float(problem["angle"])
        except Exception:
            return {"rotated_image": []}

        # Ensure numpy array of floats
        img = np.asarray(image, dtype=float)

        # Expect a 2D image
        if img.ndim != 2:
            try:
                img = img.squeeze()
                if img.ndim != 2:
                    return {"rotated_image": []}
            except Exception:
                return {"rotated_image": []}

        if img.size == 0:
            return {"rotated_image": []}

        # Normalize angle and handle exact multiples of 90 degrees quickly
        norm = float(angle) % 360.0
        tol = 1e-12
        k = int(round(norm / 90.0)) % 4
        if abs(norm - (k * 90.0)) <= tol:
            rotated = np.rot90(img, k=k)
            # Return a list of row views (fast to create)
            return {"rotated_image": list(rotated)}

        if abs(norm) <= tol:
            return {"rotated_image": list(img)}

        # Preallocate output buffer and call scipy.ndimage.rotate with output to reduce allocations
        out = np.empty_like(img)
        rotate = scipy.ndimage.rotate
        try:
            rotate(img, angle, reshape=self.reshape, output=out, order=self.order, mode=self.mode, cval=self.cval)
            return {"rotated_image": list(out)}
        except Exception:
            # Fallback to standard call
            try:
                rotated = rotate(img, angle, reshape=self.reshape, order=self.order, mode=self.mode, cval=self.cval)
                return {"rotated_image": list(rotated)}
            except Exception:
                return {"rotated_image": []}