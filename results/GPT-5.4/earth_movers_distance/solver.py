import numpy as np

try:
    from ot.lp._network_simplex import emd_c as _emd_c
except Exception:  # pragma: no cover - fallback if internal API changes
    _emd_c = None
    import ot

    _emd = ot.lp.emd
else:
    _emd = None

class Solver:
    __slots__ = ()

    def __init__(self):
        pass

    def solve(self, problem, **kwargs):
        a_in = problem["source_weights"]
        b_in = problem["target_weights"]
        m_in = problem["cost_matrix"]

        if isinstance(m_in, np.ndarray) and m_in.dtype == np.float64 and m_in.flags.c_contiguous:
            M = m_in
        else:
            M = np.ascontiguousarray(m_in, dtype=np.float64)

        if isinstance(a_in, np.ndarray) and a_in.dtype == np.float64 and a_in.ndim == 1 and a_in.flags.c_contiguous:
            a = a_in
        else:
            a = np.asarray(a_in, dtype=np.float64)
            if a.ndim != 1:
                a = a.reshape(-1)
            a = np.ascontiguousarray(a)

        if isinstance(b_in, np.ndarray) and b_in.dtype == np.float64 and b_in.ndim == 1 and b_in.flags.c_contiguous:
            b = b_in
        else:
            b = np.asarray(b_in, dtype=np.float64)
            if b.ndim != 1:
                b = b.reshape(-1)
            b = np.ascontiguousarray(b)

        total_a = float(a.sum())
        total_b = float(b.sum())
        if total_b != total_a and total_b != 0.0:
            b = np.ascontiguousarray(b * (total_a / total_b))

        if a.size == 1:
            plan = b.reshape(1, -1).copy()
        elif b.size == 1:
            plan = a.reshape(-1, 1).copy()
        elif a.size == 2 and b.size == 2:
            upper = a[0] if a[0] < b[0] else b[0]
            lower = a[0] - b[1]
            if lower < 0.0:
                lower = 0.0
            x = upper if (M[0, 0] + M[1, 1] <= M[0, 1] + M[1, 0]) else lower
            plan = np.empty((2, 2), dtype=np.float64)
            plan[0, 0] = x
            plan[0, 1] = a[0] - x
            plan[1, 0] = b[0] - x
            plan[1, 1] = a[1] - plan[1, 0]
        elif _emd_c is not None:
            plan = _emd_c(a, b, M, 100000, 1)[0]
        else:
            plan = _emd(a, b, M, check_marginals=False)

        return {"transport_plan": plan}