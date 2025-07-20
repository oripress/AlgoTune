import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Solve the discrete optimal transport (EMD) problem by
        reducing to an integer min-cost flow and using OR-Tools.
        """
        # Load inputs as numpy arrays
        a = np.asarray(problem["source_weights"], dtype=np.float64)
        b = np.asarray(problem["target_weights"], dtype=np.float64)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64)
        n, m = a.size, b.size

        # Scale supplies and demands to integers
        FLOW_SCALE = 10**6
        supply = np.rint(a * FLOW_SCALE).astype(np.int64)
        demand = np.rint(b * FLOW_SCALE).astype(np.int64)
        # Fix rounding mismatch
        diff = int(supply.sum() - demand.sum())
        if diff > 0:
            supply[int(np.argmax(supply))] -= diff
        elif diff < 0:
            demand[int(np.argmax(demand))] -= -diff

        # Scale costs to integers
        COST_SCALE = 10**6
        cost_int = np.rint(M * COST_SCALE).astype(np.int64)

        # Try OR-Tools min-cost flow
        try:
            pywrap = __import__('ortools.graph.pywrapgraph', fromlist=['SimpleMinCostFlow'])
            mcf = pywrap.SimpleMinCostFlow()

            # Add arcs: from source i to sink (n+j)
            for i in range(n):
                si = int(supply[i])
                if si <= 0:
                    continue
                for j in range(m):
                    dj = int(demand[j])
                    if dj <= 0:
                        continue
                    cap = si if si < dj else dj
                    if cap <= 0:
                        continue
                    mcf.AddArcWithCapacityAndUnitCost(
                        i, n + j, cap, int(cost_int[i, j])
                    )

            # Set supplies: positive for sources, negative for sinks
            for i in range(n):
                mcf.SetNodeSupply(int(i), int(supply[i]))
            for j in range(m):
                mcf.SetNodeSupply(int(n + j), -int(demand[j]))

            status = mcf.Solve()
            if status == mcf.OPTIMAL:
                G = np.zeros((n, m), dtype=np.float64)
                for arc in range(mcf.NumArcs()):
                    f = mcf.Flow(arc)
                    if f > 0:
                        u = mcf.Tail(arc)
                        v = mcf.Head(arc)
                        if u < n and v >= n:
                            G[int(u), int(v - n)] = f / FLOW_SCALE
                return {"transport_plan": G}
        except Exception:
            pass

        # Fallback to POT LP solver if OR-Tools is unavailable or fails
        import ot
        G = ot.lp.emd(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            np.ascontiguousarray(M, dtype=np.float64),
            check_marginals=False,
        )
        return {"transport_plan": G}