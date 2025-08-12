from ortools.sat.python import cp_model
class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        # trivial: no edges
        have_edge = False
        for i in range(n):
            for j in range(i+1, n):
                if problem[i][j]:
                    have_edge = True
                    break
            if have_edge:
                break
        if not have_edge:
            return []
        # choose method: bitset for moderate size
        if n <= 50:
            return self._solve_bitset(problem)
        else:
            return self._solve_cpsat(problem)

    def _solve_cpsat(self, problem):
        n = len(problem)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x{i}') for i in range(n)]
        for i in range(n):
            row = problem[i]
            for j in range(i+1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)
        model.Maximize(sum(x))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(x[i]) == 0]
        # fallback to full cover
        return list(range(n))

    def _solve_bitset(self, problem):
        n = len(problem)
        all_mask = (1 << n) - 1
        # adjacency bitsets
        neighbor = [0] * n
        for i in range(n):
            m = 0
            row = problem[i]
            for j in range(n):
                if row[j]:
                    m |= 1 << j
            neighbor[i] = m
        # complement neighbors (exclude self)
        neighborC = [(all_mask ^ neighbor[i]) & ~(1 << i) for i in range(n)]
        # initial greedy MIS bound
        greedy_mask = 0
        greedy_size = 0
        P_mask_g = all_mask
        while P_mask_g:
            u = None
            min_deg = n + 1
            m2 = P_mask_g
            while m2:
                vbit2 = m2 & -m2
                m2 ^= vbit2
                v2 = vbit2.bit_length() - 1
                deg = (neighbor[v2] & P_mask_g).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    u = v2
            greedy_mask |= 1 << u
            greedy_size += 1
            P_mask_g &= ~(neighbor[u] | (1 << u))
        best_size = greedy_size
        best_mask = greedy_mask
        # branch and bound with pivot (Tomita)
        def dfs(P_mask, R_size, curr_mask):
            nonlocal best_size, best_mask
            if P_mask == 0:
                if R_size > best_size:
                    best_size = R_size
                    best_mask = curr_mask
                return
            # bounding
            if R_size + P_mask.bit_count() <= best_size:
                return
            # pivot selection
            mask_u = P_mask
            u = None
            max_common = -1
            while mask_u:
                vbit = mask_u & -mask_u
                mask_u ^= vbit
                v = vbit.bit_length() - 1
                common = (P_mask & neighborC[v]).bit_count()
                if common > max_common:
                    max_common = common
                    u = v
            # branch on vertices not adjacent to pivot
            P_without = P_mask & ~neighborC[u]
            m = P_without
            while m:
                vbit = m & -m
                m ^= vbit
                dfs(P_mask & neighborC[vbit.bit_length() - 1],
                    R_size + 1,
                    curr_mask | vbit)
                # remove vbit from P_mask to reduce search
                P_mask &= ~vbit
        dfs(all_mask, 0, 0)
        # MIS vertices indicated by bits in best_mask; return complement as vertex cover
        return [i for i in range(n) if not ((best_mask >> i) & 1)]
        return [i for i in range(n) if not ((best_mask >> i) & 1)]