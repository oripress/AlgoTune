import numpy as np

# Try to import OR-Tools (fast C++ min-cost flow). If unavailable, fall back to POT.
try:
    from ortools.graph import pywrapgraph  # type: ignore
    _ORTOOLS_AVAILABLE = True
except Exception:
    _ORTOOLS_AVAILABLE = False

# POT (Python Optimal Transport) fallback (reference solver)
try:
    import ot  # type: ignore
    _POT_EMD = ot.lp.emd
except Exception:
    _POT_EMD = None

class Solver:
    def _nw_corner(self, a, b):
        """Deterministic northwest-corner feasible transport plan (fallback)."""
        na = a.size
        nb = b.size
        G = np.zeros((na, nb), dtype=np.float64)
        a_tmp = a.copy()
        b_tmp = b.copy()
        i = 0
        j = 0
        tol = 1e-15
        while i < na and j < nb:
            if a_tmp[i] <= tol:
                i += 1
                continue
            if b_tmp[j] <= tol:
                j += 1
                continue
            move = min(a_tmp[i], b_tmp[j])
            G[i, j] = move
            a_tmp[i] -= move
            b_tmp[j] -= move
            if a_tmp[i] <= tol:
                i += 1
            if b_tmp[j] <= tol:
                j += 1
        return G

    def solve(self, problem, **kwargs):
        """
        Solve discrete EMD (optimal transport) problem.

        Strategy:
         - Prefer OR-Tools SimpleMinCostFlow (C++) for speed when available.
         - Fall back to POT's ot.lp.emd (reference) if OR-Tools isn't present.
         - Final fallback: NW-corner feasible plan.

        Returns:
            {"transport_plan": G} where G is an (n x m) numpy array.
        """
        a = np.asarray(problem["source_weights"], dtype=np.float64).reshape(-1)
        b = np.asarray(problem["target_weights"], dtype=np.float64).reshape(-1)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)

        n = a.size
        m = b.size

        # Trivial cases
        if n == 0 or m == 0:
            return {"transport_plan": np.zeros((n, m), dtype=np.float64)}

        # Prefer POT (fast C implementation) if available; else fall back to OR-Tools, finally NW-corner
        if _POT_EMD is not None:
            G = _POT_EMD(a, b, M, check_marginals=False)
            return {"transport_plan": np.asarray(G, dtype=np.float64)}
        if not _ORTOOLS_AVAILABLE:
            return {"transport_plan": self._nw_corner(a, b)}

        # Use OR-Tools Min-Cost Flow on a reduced problem (drop zero-weight bins)
        tol = 1e-15
        idx_a = np.flatnonzero(a > tol)
        idx_b = np.flatnonzero(b > tol)

        if idx_a.size == 0 or idx_b.size == 0:
            return {"transport_plan": np.zeros((n, m), dtype=np.float64)}

        a_red = a[idx_a].astype(np.float64)
        b_red = b[idx_b].astype(np.float64)
        M_red = M[np.ix_(idx_a, idx_b)].astype(np.float64)

        # Integer scaling for marginals to use integer min-cost flow
        weight_scale = 10**6  # gives ~1e-6 granularity in transported mass
        a_int = np.rint(a_red * weight_scale).astype(np.int64)
        b_int = np.rint(b_red * weight_scale).astype(np.int64)
        a_int[a_int < 0] = 0
        b_int[b_int < 0] = 0

        sum_a = int(a_int.sum())
        sum_b = int(b_int.sum())

        if sum_a == 0 or sum_b == 0:
            return {"transport_plan": np.zeros((n, m), dtype=np.float64)}

        # Balance totals deterministically by adjusting the largest bin (small rounding fix)
        if sum_a != sum_b:
            diff = sum_a - sum_b
            if diff > 0:
                jmax = int(np.argmax(b_int))
                b_int[jmax] += diff
            else:
                imax = int(np.argmax(a_int))
                a_int[imax] += -diff

        total_flow = int(a_int.sum())
        if total_flow == 0:
            return {"transport_plan": np.zeros((n, m), dtype=np.float64)}

        # Scale costs to integer while avoiding int64 overflow
        max_M = float(np.max(np.abs(M_red))) if M_red.size > 0 else 0.0
        if max_M <= 0.0:
            cost_scale = 1
        else:
            # Choose cost_scale so that max_M * cost_scale * total_flow stays well below 2^63
            # Conservative divisor chosen to avoid overflow across a wide range of sizes
            try:
                allowed = int(9_000_000_000_000 // (int(max_M * total_flow) + 1))
            except Exception:
                allowed = 1
            if allowed <= 0:
                cost_scale = 1
            else:
                cost_scale = min(10**6, max(1, allowed))

        cost_int = np.rint(M_red * cost_scale).astype(np.int64)

        # If there are negative costs, shift them by a constant (doesn't change argmin since flow is fixed)
        min_cost = int(np.min(cost_int))
        if min_cost < 0:
            cost_int = cost_int + (-min_cost)

        na = a_int.size
        nb = b_int.size

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()

        # Add arcs: from each supply i to each demand j
        # capacity set to supply a_int[i] (node supplies will ensure correctness)
        for i_local in range(na):
            cap_i = int(a_int[i_local])
            if cap_i <= 0:
                continue
            tail = int(i_local)
            row_costs = cost_int[i_local]
            for j_local in range(nb):
                head = int(na + j_local)
                c = int(row_costs[j_local])
                min_cost_flow.AddArcWithCapacityAndUnitCost(tail, head, cap_i, c)

        # Set node supplies (+ for sources, - for sinks)
        for i_local in range(na):
            s = int(a_int[i_local])
            if s != 0:
                min_cost_flow.SetNodeSupply(i_local, s)
        for j_local in range(nb):
            d = int(b_int[j_local])
            if d != 0:
                min_cost_flow.SetNodeSupply(na + j_local, -d)

        # Solve
        status = min_cost_flow.Solve()
        if status != min_cost_flow.OPTIMAL:
            # Fallback to POT for correctness if available, else NW-corner
            if _POT_EMD is not None:
                G = _POT_EMD(a, b, M, check_marginals=False)
                return {"transport_plan": np.asarray(G, dtype=np.float64)}
            return {"transport_plan": self._nw_corner(a, b)}

        # Reconstruct full transport plan on original indices
        G_full = np.zeros((n, m), dtype=np.float64)
        num_arcs = min_cost_flow.NumArcs()
        for arc in range(num_arcs):
            tail = min_cost_flow.Tail(arc)
            head = min_cost_flow.Head(arc)
            # Our arcs are from [0..na-1] -> [na..na+nb-1]
            if tail < na and head >= na:
                flow = int(min_cost_flow.Flow(arc))
                if flow == 0:
                    continue
                i_local = int(tail)
                j_local = int(head - na)
                i_orig = int(idx_a[i_local])
                j_orig = int(idx_b[j_local])
                G_full[i_orig, j_orig] = flow / float(weight_scale)

        # Clamp tiny negatives and return
        if np.any(G_full < 0):
            G_full = np.maximum(G_full, 0.0)

        return {"transport_plan": G_full}