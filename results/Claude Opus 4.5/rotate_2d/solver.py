import numpy as np
from scipy.ndimage import map_coordinates

class Solver:
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"], dtype=np.float64)
        angle = problem["angle"]
        
        # Normalize angle to [0, 360)
        angle_mod = angle % 360
        
        # Special case for no rotation
        if abs(angle_mod) < 1e-10 or abs(angle_mod - 360) < 1e-10:
            return {"rotated_image": image.copy()}
        
        # Special cases for 90, 180, 270 degrees
        if abs(angle_mod - 90) < 1e-10:
            return {"rotated_image": np.rot90(image, 1)}
        if abs(angle_mod - 180) < 1e-10:
            return {"rotated_image": np.rot90(image, 2)}
        if abs(angle_mod - 270) < 1e-10:
            return {"rotated_image": np.rot90(image, 3)}
        
        n, m = image.shape
        
        # Compute rotation matrix (counter-clockwise)
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Center of rotation
        cy, cx = (n - 1) / 2.0, (m - 1) / 2.0
        
        # Create output coordinate grid
        y_out, x_out = np.mgrid[0:n, 0:m].astype(np.float64)
        
        # Shift to center
        y_out_c = y_out - cy
        x_out_c = x_out - cx
        
        # Apply inverse rotation (to map output coords to input coords)
        # For counter-clockwise rotation by angle, inverse is clockwise rotation
        y_in = cos_a * y_out_c + sin_a * x_out_c + cy
        x_in = -sin_a * y_out_c + cos_a * x_out_c + cx
        
        # Stack coordinates for map_coordinates
        coords = np.array([y_in, x_in])
        
        # Use cubic spline interpolation
        rotated_image = map_coordinates(image, coords, order=3, mode='constant', cval=0.0)
        
        return {"rotated_image": rotated_image}