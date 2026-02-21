import scipy.ndimage
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        transformed_image = scipy.ndimage.affine_transform(
            problem["image"], problem["matrix"], order=3, mode='constant', cval=0.0
        )
        return {"transformed_image": transformed_image}