import numpy as np
from scipy.ndimage import affine_transform

class Solver:
    def solve(self, problem: dict, **kwargs):
        # Load image and transform
        image = np.asarray(problem["image"], dtype=np.float64)
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        H, W = image.shape

        # If matrix has shape (2,3), split into linear + offset
        if matrix.ndim == 2 and matrix.shape[1] == image.ndim + 1:
            linear = matrix[:, :image.ndim]
            offset = matrix[:, image.ndim]
        else:
            linear = matrix
            offset = np.zeros(linear.shape[0], dtype=np.float64)

        # Perform affine transform in C
        transformed = affine_transform(
            image,
            linear,
            offset=offset,
            output_shape=(H, W),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True
        )
        return {"transformed_image": transformed.tolist()}