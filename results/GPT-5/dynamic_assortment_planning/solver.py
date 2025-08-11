from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[int]:
        """
        Solve Dynamic Assortment Planning exactly via a linear program.

        Maximize sum_{t,i} w[t,i] * x[t,i]
        s.t.
            - For each period t: sum_i x[t,i] <= 1
            - For each product i: sum_t x[t,i] <= capacities[i]
            - 0 <= x[t,i] <= 1
        The constraint matrix is totally unimodular, so the LP relaxation yields an integral optimum.
        """
        T = int(problem["T"])
        N = int(problem["N"])
        prices = problem["prices"]
        capacities_in = problem["capacities"]
        probs = problem["probs"]

        # Quick exits
        if T <= 0 or N <= 0:
            return []

        # Normalize capacities: ensure non-negative ints and not exceeding T
        caps = [0] * N
        all_zero_caps = True
        for i in range(N):
            ci = int(capacities_in[i])
            if ci < 0:
                ci = 0
            if ci > T:
                ci = T
            caps[i] = ci
            if ci > 0:
                all_zero_caps = False

        # If no capacities available, we must stay idle every period
        if all_zero_caps:
            return [-1] * T

        # Early exit: N == 1
        if N == 1:
            # Choose up to caps[0] best periods with positive weight
            weights = []
            for t in range(T):
                p = probs[t][0] if len(probs[t]) > 0 else 0.0
                w = prices[0] * p
                if w > 0.0:
                    weights.append((w, t))
            if not weights:
                return [-1] * T
            weights.sort(reverse=True)
            offer = [-1] * T
            for _, t in weights[: caps[0]]:
                offer[t] = 0
            return offer

        # Vectorized preparation
        try:
            import numpy as np  # type: ignore

            # Build dense probability matrix P (T x N) robustly
            P_in = np.asarray(probs, dtype=float)
            if P_in.ndim != 2:
                # Fallback to safe zeros if malformed
                P = np.zeros((T, N), dtype=float)
            else:
                t0 = min(T, P_in.shape[0])
                n0 = min(N, P_in.shape[1])
                if (t0, n0) == (T, N):
                    P = P_in
                else:
                    P = np.zeros((T, N), dtype=float)
                    if t0 > 0 and n0 > 0:
                        P[:t0, :n0] = P_in[:t0, :n0]

            prices_arr = np.asarray(prices, dtype=float)
            caps_arr = np.asarray(caps, dtype=int)

            # Compute weights W = P * prices (broadcast across columns)
            W = P * prices_arr

            # Mask out products with zero capacity
            zero_cap_mask = caps_arr <= 0
            if zero_cap_mask.any():
                W[:, zero_cap_mask] = 0.0

            # Early exit: per-period best if capacities not violated
            W_mask = W.copy()
            if zero_cap_mask.any():
                W_mask[:, zero_cap_mask] = -np.inf
            best_i = np.argmax(W_mask, axis=1)
            best_w = W_mask[np.arange(T), best_i]
            offer_candidate = [-1] * T
            valid_rows = best_w > 0.0
            if np.any(valid_rows):
                # Count selections per product
                chosen_i = best_i[valid_rows]
                counts = np.bincount(chosen_i, minlength=N)
                if np.all(counts <= caps_arr):
                    # Valid: construct offer
                    for t in range(T):
                        offer_candidate[t] = int(best_i[t]) if best_w[t] > 0.0 else -1
                    return offer_candidate

            # Hungarian algorithm branch for small instances
            try:
                from scipy.optimize import linear_sum_assignment  # type: ignore

                sum_cap = int(caps_arr.sum())
                # Thresholds chosen to avoid large dense matrices
                if 0 < sum_cap <= 250 and T <= 250:
                    # Build cost matrix: columns = replicated product slots + T idle cols
                    total_cols = sum_cap + T
                    C = np.zeros((T, total_cols), dtype=float)
                    col = 0
                    for i in range(N):
                        ci = int(caps_arr[i])
                        if ci <= 0:
                            continue
                        if ci == 1:
                            C[:, col] = -W[:, i]
                            col += 1
                        else:
                            # Replicate this product column ci times
                            C[:, col : col + ci] = -W[:, i, None]
                            col += ci
                    # Idle columns already zero cost
                    # Solve assignment
                    row_ind, col_ind = linear_sum_assignment(C)
                    offer = [-1] * T
                    # Decode: if assigned column < sum_cap -> product; else idle
                    col_to_prod = []
                    for i in range(N):
                        ci = int(caps_arr[i])
                        if ci > 0:
                            col_to_prod.extend([i] * ci)
                    for r, c in zip(row_ind, col_ind):
                        if c < sum_cap:
                            offer[r] = int(col_to_prod[c])
                        else:
                            offer[r] = -1
                    return offer
            except Exception:
                pass

            # Build variables for positive-weight edges using vectorization
            pos_mask = W > 0.0
            if not np.any(pos_mask):
                return [-1] * T

            var_t, var_i = np.nonzero(pos_mask)
            var_w = W[pos_mask].astype(float)
            num_vars = var_w.size

            # Min-Cost Flow branch for medium sparse instances
            # Add idle arcs to ensure feasibility with total supply = T.
            try:
                if num_vars <= 300_000 and T <= 100_000 and (T + N) <= 150_000:
                    from ortools.graph import pywrapgraph  # type: ignore

                    source = 0
                    first_period = 1
                    first_product = first_period + T
                    sink = first_product + N
                    num_nodes = sink + 1

                    mcf = pywrapgraph.SimpleMinCostFlow()

                    # Add source -> period arcs and period -> sink (idle) arcs
                    for t in range(T):
                        u = source
                        v = first_period + t
                        mcf.AddArcWithCapacityAndUnitCost(u, v, 1, 0)
                        # Idle arc
                        mcf.AddArcWithCapacityAndUnitCost(v, sink, 1, 0)

                    # Add product -> sink arcs
                    for i in range(N):
                        ci = int(caps_arr[i])
                        if ci > 0:
                            u = first_product + i
                            mcf.AddArcWithCapacityAndUnitCost(u, sink, ci, 0)

                    # Scaling for integer costs
                    w_max = float(np.max(var_w)) if num_vars > 0 else 0.0
                    if not np.isfinite(w_max) or w_max <= 0.0:
                        scale = 1
                    else:
                        # Keep costs within ~1e12 to avoid overflow, maximize precision otherwise
                        scale = int(max(1.0, min(1e9, 1e12 / w_max)))

                    # Add period -> product arcs with negative costs (maximize revenue)
                    for idx in range(num_vars):
                        t = int(var_t[idx])
                        i = int(var_i[idx])
                        u = first_period + t
                        v = first_product + i
                        cost = -int(round(var_w[idx] * scale))
                        # Capacity 1
                        mcf.AddArcWithCapacityAndUnitCost(u, v, 1, cost)

                    # Supplies
                    mcf.SetNodeSupply(source, T)
                    mcf.SetNodeSupply(sink, -T)
                    # Other nodes default to 0 supply

                    status = mcf.Solve()
                    if status == mcf.OPTIMAL:
                        offer = [-1] * T
                        # Read flows on arcs exiting period nodes
                        num_arcs = mcf.NumArcs()
                        for a in range(num_arcs):
                            tail = mcf.Tail(a)
                            head = mcf.Head(a)
                            if first_period <= tail < first_period + T and mcf.Flow(a) > 0:
                                t = tail - first_period
                                if head == sink:
                                    offer[t] = -1
                                elif first_product <= head < first_product + N:
                                    offer[t] = head - first_product
                        return offer
                    # else fall through to LP
            except Exception:
                pass

            # Build and solve LP using SciPy's HiGHS (dual simplex tends to be fast)
            try:
                from scipy.optimize import linprog  # type: ignore
                from scipy.sparse import coo_matrix  # type: ignore

                # Objective: maximize sum w * x -> minimize sum (-w) * x
                c = -var_w  # numpy array

                # Constraints A_ub x <= b_ub
                # Rows: first T period-rows, then N product-rows
                num_rows = T + N

                # Each variable contributes 1 to its period row and 1 to its product row
                idxs = np.arange(num_vars, dtype=int)
                rows = np.empty(2 * num_vars, dtype=int)
                cols = np.empty(2 * num_vars, dtype=int)
                # Period entries
                rows[0::2] = var_t
                cols[0::2] = idxs
                # Product entries
                rows[1::2] = T + var_i
                cols[1::2] = idxs
                data = np.ones(2 * num_vars, dtype=float)

                A_ub = coo_matrix((data, (rows, cols)), shape=(num_rows, num_vars)).tocsr()

                # RHS: periods <= 1, products <= cap
                b_ub = np.concatenate((np.ones(T, dtype=float), caps_arr.astype(float)))

                res = linprog(
                    c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=(0.0, 1.0),
                    method="highs-ds",
                    options={"presolve": True},
                )

                if not res.success or res.x is None:
                    raise RuntimeError("LP solver failed")

                x = res.x  # numpy array length num_vars

                # Decode solution: due to total unimodularity, x entries are {0,1}
                offer = [-1] * T
                chosen_idxs = np.nonzero(x > 0.5)[0]
                for idx in chosen_idxs:
                    offer[int(var_t[idx])] = int(var_i[idx])
                return offer
            except Exception:
                # Fallback: exact CP-SAT solver (reference approach)
                try:
                    from ortools.sat.python import cp_model  # type: ignore

                    model = cp_model.CpModel()
                    x = {}
                    for t in range(T):
                        for i in range(N):
                            x[(t, i)] = model.NewBoolVar(f"x_{t}_{i}")

                    # Each period at most one product
                    for t in range(T):
                        model.Add(sum(x[(t, i)] for i in range(N)) <= 1)

                    # Capacity limits
                    for i in range(N):
                        model.Add(sum(x[(t, i)] for t in range(T)) <= caps[i])

                    # Objective
                    model.Maximize(
                        sum(
                            prices[i] * probs[t][i] * x[(t, i)]
                            for t in range(T)
                            for i in range(N)
                        )
                    )

                    solver = cp_model.CpSolver()
                    status = solver.Solve(model)
                    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        return [-1] * T

                    offer = []
                    for t in range(T):
                        chosen = -1
                        for i in range(N):
                            if solver.Value(x[(t, i)]) == 1:
                                chosen = i
                                break
                        offer.append(chosen)
                    return offer
                except Exception:
                    # Final fallback: idle all periods
                    return [-1] * T

        except Exception:
            # If NumPy isn't available for some reason, fall back to previous implementation path
            # Build variables only for positive-weight edges (weights >= 0, skip zero)
            var_t: List[int] = []
            var_i: List[int] = []
            var_w: List[float] = []
            vars_by_period: List[List[int]] = [[] for _ in range(T)]

            k = 0
            for t in range(T):
                row = probs[t]
                # Defensive: ensure row length
                if len(row) != N:
                    # Truncate or pad with zeros to length N
                    if len(row) > N:
                        row = row[:N]
                    else:
                        row = row + [0.0] * (N - len(row))
                for i in range(N):
                    if caps[i] == 0:
                        continue
                    w = prices[i] * row[i]
                    if w > 0.0:
                        var_t.append(t)
                        var_i.append(i)
                        var_w.append(float(w))
                        vars_by_period[t].append(k)
                        k += 1

            num_vars = len(var_w)
            if num_vars == 0:
                return [-1] * T

            # Build and solve LP using SciPy's HiGHS
            try:
                from scipy.optimize import linprog  # type: ignore
                from scipy.sparse import coo_matrix  # type: ignore

                # Objective: maximize sum w * x -> minimize sum (-w) * x
                c = [-w for w in var_w]

                # Constraints A_ub x <= b_ub
                # Rows: first T period-rows, then N product-rows
                num_rows = T + N

                # Build A in COO: each variable contributes:
                #  - 1 in its period row
                #  - 1 in its product row
                nnz = 2 * num_vars
                rows = [0] * nnz
                cols = [0] * nnz
                data = [1.0] * nnz
                for idx in range(num_vars):
                    # Period constraint row
                    rows[2 * idx] = var_t[idx]
                    cols[2 * idx] = idx
                    # Product constraint row
                    rows[2 * idx + 1] = T + var_i[idx]
                    cols[2 * idx + 1] = idx

                A_ub = coo_matrix((data, (rows, cols)), shape=(num_rows, num_vars)).tocsr()

                # RHS: periods <= 1, products <= cap
                b_ub = [1.0] * T + [float(caps[i]) for i in range(N)]

                res = linprog(
                    c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=(0.0, 1.0),
                    method="highs-ds",
                    options={"presolve": True},
                )

                if not res.success or res.x is None:
                    # Fallback if LP doesn't solve
                    raise RuntimeError("LP solver failed")

                x = res.x
                # Construct offer decision
                offer = [-1] * T
                # Threshold to determine selected edge; due to total unimodularity, x should be in {0,1}
                thr = 0.5
                for t in range(T):
                    chosen = -1
                    for idx in vars_by_period[t]:
                        if x[idx] > thr:
                            chosen = var_i[idx]
                            break
                    offer[t] = chosen
                return offer
            except Exception:
                # Fallback: exact CP-SAT solver (reference approach)
                try:
                    from ortools.sat.python import cp_model  # type: ignore

                    model = cp_model.CpModel()
                    x = {}
                    for t in range(T):
                        for i in range(N):
                            x[(t, i)] = model.NewBoolVar(f"x_{t}_{i}")

                    # Each period at most one product
                    for t in range(T):
                        model.Add(sum(x[(t, i)] for i in range(N)) <= 1)

                    # Capacity limits
                    for i in range(N):
                        model.Add(sum(x[(t, i)] for t in range(T)) <= caps[i])

                    # Objective
                    model.Maximize(
                        sum(prices[i] * probs[t][i] * x[(t, i)] for t in range(T) for i in range(N))
                    )

                    solver = cp_model.CpSolver()
                    status = solver.Solve(model)
                    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                        return [-1] * T

                    offer = []
                    for t in range(T):
                        chosen = -1
                        for i in range(N):
                            if solver.Value(x[(t, i)]) == 1:
                                chosen = i
                                break
                        offer.append(chosen)
                    return offer
                except Exception:
                    # Final fallback: idle all periods
                    return [-1] * T