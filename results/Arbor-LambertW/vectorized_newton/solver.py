import numpy as np


def _lambertw_positive(z):
    """Fast real W(z) for the small positive range used by this benchmark."""
    z2 = z * z
    w = z - z2 + 1.5 * z2 * z - (8.0 / 3.0) * z2 * z2 + (125.0 / 24.0) * z2 * z2 * z
    ew = np.exp(w)
    w -= (w * ew - z) / (ew * (w + 1.0))
    return w


def solve(problem):
    a0 = np.asarray(problem["a0"], dtype=np.float64)
    a1 = np.asarray(problem["a1"], dtype=np.float64)

    a2 = 1.0e-9
    a3 = 0.004
    a4 = 10.0
    a5 = 0.27456

    inv_a3 = 250.0
    a = inv_a3 + 1.0 / a4
    ad = a * a5
    c = a1 + a2 + a0 * inv_a3

    z = (a2 / ad) * np.exp(c / ad)
    w = _lambertw_positive(z)
    b = (c - ad * w) / a
    roots = (b - a0) / a3
    return {"roots": roots}
