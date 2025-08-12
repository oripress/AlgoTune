import sys
import time
from ortools.sat.python import cp_model

sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem, **kwargs):
        start = time.time()
        m = len(problem)
        if m == 0:
            return []
        # Convert to sets and list unique elements
        subsets = [set(s) for s in problem]
        elements = sorted({e for subset in subsets for e in subset})
        n = len(elements)
        # Map each element to a bit position
        idx_map = {e: i for i, e in enumerate(elements)}
        # Build bitmask for each subset
        subset_mask = [0] * m
        for i, subset in enumerate(subsets):
            mask = 0
            for e in subset:
                mask |= 1 << idx_map[e]
            subset_mask[i] = mask
        # Map each element‐index to covering subsets
        element_to_subsets = [[] for _ in range(n)]
        for i, mask in enumerate(subset_mask):
            tmp = mask
            while tmp:
                b = tmp & -tmp
                j = b.bit_length() - 1
                element_to_subsets[j].append(i)
                tmp ^= b
        # Precompute coverage sizes
        cover_counts = [mask.bit_count() for mask in subset_mask]
        # Sort covers descending to speed up B&B and hints
        for lst in element_to_subsets:
            lst.sort(key=lambda i: cover_counts[i], reverse=True)
        # Greedy initial solution (upper bound)
        full_universe = (1 << n) - 1
        uncovered = full_universe
        greedy_sol = []
        while uncovered:
            best_i = max(range(m), key=lambda i: bin(subset_mask[i] & uncovered).count('1'))
            greedy_sol.append(best_i)
            uncovered &= ~subset_mask[best_i]
        best_solution = greedy_sol.copy()
        best_len = len(best_solution)
        # Lower‐bound estimate: ceil(remain_elements / max_cover)
        max_cover = max(cover_counts) if cover_counts else 1
        def lb(remain):
            return (bin(remain).count('1') + max_cover - 1) // max_cover
        # Branch‐and‐bound on small instances
        threshold_m = kwargs.get('branch_threshold_m', kwargs.get('branch_threshold', 40))
        threshold_n = kwargs.get('branch_threshold_n', 60)
        if m <= threshold_m or n <= threshold_n:
            # Recursive DFS with pruning
            def dfs(uncovered, current):
                nonlocal best_len, best_solution
                d = len(current)
                # prune by LB
                if d + lb(uncovered) >= best_len:
                    return
                # if fully covered, record solution
                if uncovered == 0:
                    best_len = d
                    best_solution = current.copy()
                    return
                # pick uncovered element with fewest choices
                tmp = uncovered
                min_c, e_sel = None, None
                while tmp:
                    b = tmp & -tmp
                    j = b.bit_length() - 1
                    c = len(element_to_subsets[j])
                    if min_c is None or c < min_c:
                        min_c, e_sel = c, j
                        if c <= 1:
                            break
                    tmp ^= b
                # branch on each subset covering e_sel
                for s in element_to_subsets[e_sel]:
                    current.append(s)
                    dfs(uncovered & ~subset_mask[s], current)
                    current.pop()
            dfs(full_universe, [])
            sol = sorted(best_solution)
            return [i + 1 for i in sol]
        # Fallback to CP‐SAT for larger instances
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x{i}') for i in range(m)]
        for j in range(n):
            model.AddBoolOr([x[i] for i in element_to_subsets[j]])
        model.Minimize(sum(x))
        # Hint with greedy solution
        for i in range(m):
            model.AddHint(x[i], 1 if i in greedy_sol else 0)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = kwargs.get('num_search_workers', 8)
        solver.parameters.max_time_in_seconds = kwargs.get('time_limit', 9.5)
        solver.parameters.log_search_progress = False
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol = [i for i in range(m) if solver.Value(x[i])]
            return [i + 1 for i in sorted(sol)]
        # fallback if SAT fails
        return [i + 1 for i in sorted(greedy_sol)]