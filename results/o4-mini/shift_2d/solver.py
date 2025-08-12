import numpy as np
from scipy.ndimage import shift as nd_shift
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[List[float]]]:
        """
        Shift a 2D image using cubic spline interpolation (order=3)
        with constant padding (cval=0.0).
        """
        # Convert input to numpy array
        image = np.asarray(problem.get("image"), dtype=np.float64)
        shift_vector = problem.get("shift", [0.0, 0.0])
        try:
            # Use SciPy's optimized C implementation
            shifted = nd_shift(image, shift_vector, order=3, mode='constant', cval=0.0)
        except Exception:
            # Signal failure similarly to reference
            return {"shifted_image": []}
        # Return a list of row arrays (avoiding full .tolist() overhead)
        return {"shifted_image": list(shifted)}