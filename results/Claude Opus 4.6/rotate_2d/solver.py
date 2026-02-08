import numpy as np
from typing import Any
from scipy.ndimage import affine_transform, spline_filter
import math

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem, **kwargs) -> Any:
        image = np.asarray(problem["image"], dtype=np.float64)
        angle = float(problem["angle"])
        
        # Handle trivial cases
        angle_mod = angle % 360
        if angle_mod == 0.0:
            return {"rotated_image": image.copy()}
        
        n = image.shape[0]
        m = image.shape[1]
        
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Prefilter the image for cubic spline
        filtered = spline_filter(image, order=3, mode='constant')
        
        # Rotation matrix for affine_transform (inverse mapping)
        rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        
        # Compute offset so rotation is around center
        center = np.array([(n - 1) / 2.0, (m - 1) / 2.0])
        offset = center - rot_matrix @ center
        
        rotated_image = affine_transform(
            filtered, rot_matrix, offset=offset, order=3, mode='constant',
            cval=0.0, prefilter=False
        )
        
        return {"rotated_image": rotated_image}