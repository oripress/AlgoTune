import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem, **kwargs):
        image = problem["image"]
        angle = problem["angle"]
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float64)
        
        # Handle empty image case
        if img_array.size == 0:
            return {"rotated_image": []}
        
        h, w = img_array.shape
        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0
        
        # Precompute rotation matrix components
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Create affine transformation matrix for rotation around center
        matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        offset = np.array([
            center_x - cos_a * center_x - sin_a * center_y,
            center_y + sin_a * center_x - cos_a * center_y
        ])
        
        # Apply affine transformation
        rotated_image = scipy.ndimage.affine_transform(
            img_array,
            matrix,
            offset=offset,
            output_shape=(h, w),
            order=3,
            mode='constant',
            cval=0.0
        )
        
        return {"rotated_image": rotated_image.tolist()}