import scipy.ndimage
import numpy as np
from typing import Any
import math

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        img = problem["image"]
        if isinstance(img, np.ndarray):
            if img.dtype != np.float64:
                image = img.astype(np.float64)
            else:
                image = img
        else:
            image = np.array(img, dtype=np.float64)
            
        angle_deg = problem["angle"]
        
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        matrix = np.array([[cos_a, sin_a],
                           [-sin_a, cos_a]], dtype=np.float64)
        
        h, w = image.shape
        c0 = (h - 1) * 0.5
        c1 = (w - 1) * 0.5
        
        offset0 = c0 - (cos_a * c0 + sin_a * c1)
        offset1 = c1 - (-sin_a * c0 + cos_a * c1)
        
        output = np.empty((h, w), dtype=np.float64)
        
        try:
            scipy.ndimage.affine_transform(
                image, matrix, offset=[offset0, offset1], output=output, order=3, mode='constant', prefilter=True
            )
        except Exception:
            return {"rotated_image": []}
            
        return {"rotated_image": output}