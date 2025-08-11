from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import affine_transform

def _inv_2x2(A: np.ndarray) -> np.ndarray:
    """Fast inverse for 2x2 matrices with a fallback to pseudoinverse."""
    A = np.asarray(A, dtype=float)
    if A.shape == (2, 2):
        a, b, c, d = A.ravel()
        det = a * d - b * c
        if abs(det) > 1e-15:
            return np.array([[d, -b], [-c, a]], dtype=float) / det
    return np.linalg.pinv(A)

def _to_affine_params(matrix: Any, ndim: int = 2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert various affine representations into (matrix, offset) suitable for
    scipy.ndimage.affine_transform which expects a matrix mapping output->input
    coordinates and an optional offset.

    Supported forms:
      - (ndim, ndim+1) forward-affine [A | t] -> returns (inv(A), -inv(A)@t)
      - (ndim+1, ndim+1) homogeneous -> same conversion
      - (ndim, ndim) square matrix -> returned as-is (offset=None)
      - flattened arrays of length ndim*(ndim+1) or ndim*ndim
    """
    arr = np.asarray(matrix, dtype=float)

    # Forward (ndim x ndim+1) matrix [A | t]
    if arr.ndim == 2 and arr.shape == (ndim, ndim + 1):
        A = arr[:, :ndim]
        t = arr[:, ndim]
        A_inv = _inv_2x2(A) if ndim == 2 else np.linalg.pinv(A)
        offset = -A_inv.dot(t)
        return A_inv, offset

    # Homogeneous (ndim+1 x ndim+1)
    if arr.ndim == 2 and arr.shape == (ndim + 1, ndim + 1):
        A = arr[:ndim, :ndim]
        t = arr[:ndim, ndim]
        A_inv = _inv_2x2(A) if ndim == 2 else np.linalg.pinv(A)
        offset = -A_inv.dot(t)
        return A_inv, offset

    # Square (ndim x ndim)
    if arr.ndim == 2 and arr.shape == (ndim, ndim):
        return arr, None

    # Flattened possibilities
    flat = arr.ravel()
    if flat.size == ndim * (ndim + 1):
        A = flat[: ndim * ndim].reshape((ndim, ndim))
        t = flat[ndim * ndim : ndim * (ndim + 1)]
        A_inv = _inv_2x2(A) if ndim == 2 else np.linalg.pinv(A)
        offset = -A_inv.dot(t)
        return A_inv, offset

    if flat.size == ndim * ndim:
        A = flat.reshape((ndim, ndim))
        return A, None

    return None, None

class Solver:
    """
    Apply a 2D affine transformation to an input image using cubic spline
    interpolation (order=3) and constant boundary padding (cval=0.0).
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Basic validation
        if not isinstance(problem, dict):
            return {"transformed_image": []}
        if "image" not in problem or "matrix" not in problem:
            return {"transformed_image": []}

        try:
            image = np.asarray(problem["image"], dtype=float)
        except Exception:
            return {"transformed_image": []}

        if image.ndim != 2:
            return {"transformed_image": []}

        matrix = problem["matrix"]

        order = 3
        mode = "constant"
        cval = 0.0
        output_shape = image.shape

        # Use the provided matrix directly to match the reference behavior.
        try:
            transformed = affine_transform(
                image,
                matrix,
                order=order,
                mode=mode,
            )
        except Exception:
            # Fallback: try converting matrix to ndarray and call again
            try:
                transformed = affine_transform(
                    image,
                    np.asarray(matrix, dtype=float),
                    order=order,
                    mode=mode,
                )
            except Exception:
                return {"transformed_image": []}
        except Exception:
            # As a last resort try the direct call again to handle unexpected shapes
            try:
                transformed = affine_transform(
                    image,
                    matrix,
                    output_shape=output_shape,
                    order=order,
                    mode=mode,
                    cval=cval,
                )
            except Exception:
                return {"transformed_image": []}

        # Ensure no NaN/inf values
        if not np.all(np.isfinite(transformed)):
            transformed = np.nan_to_num(transformed, nan=cval, posinf=cval, neginf=cval)

        return {"transformed_image": transformed.tolist()}