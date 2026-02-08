from pysat.solvers import Solver as SATSolver
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

class Solver:
    def solve(self, problem, **kwargs):
        m = len(problem)
        if m == 0:
            return []
        
        # Determine the universe
        universe = set()
        for subset in problem:
            universe.update(subset)
        n = max(universe) if universe else 0
        if n == 0:
            return []
        
        # Build element-to-sets mapping
        elem_to_sets = [[] for _ in range(n + 1)]
        for i, subset in enumerate(problem):
            for e in subset:
                elem_to_sets[e].append(i + 1)  # 1-based variable
        
        # Build coverage clauses
        clauses = []
        for e in range(1, n + 1):
            if elem_to_sets[e]:
                clauses.append(elem_to_sets[e])
        
        # Greedy upper bound (optimized with sorted approach)
        sets_as_sets = [set(s) for s in problem]
        uncovered = set(range(1, n + 1))
        greedy_selected = []
        remaining = list(range(m))
        
        while uncovered:
            best_i = -1
            best_count = -1
            for i in remaining:
                count = len(sets_as_sets[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_i = i
            if best_i == -1 or best_count == 0:
                break
            greedy_selected.append(best_i)
            uncovered -= sets_as_sets[best_i]
            remaining.remove(best_i)
        
        upper = len(greedy_selected)
        best_solution = [i + 1 for i in greedy_selected]
        
        if upper <= 1:
            return best_solution
        
        # LP relaxation for lower bound
        rows = []
        cols = []
        for i, subset in enumerate(problem):
            for e in subset:
                rows.append(e - 1)
                cols.append(i)
        data = np.ones(len(rows))
        A = csc_matrix((data, (rows, cols)), shape=(n, m))
        
        c = np.ones(m)
        # A @ x >= 1  =>  -A @ x <= -1
        res = linprog(c, A_ub=-A, b_ub=-np.ones(n), bounds=[(0, 1)] * m, method='highs')
        if res.success:
            lb = int(np.ceil(res.fun - 1e-9))
        else:
            lb = 1
        
        left = max(lb, 1)
        right = upper
        
        # Binary search with Minicard (native cardinality)
        lits = list(range(1, m + 1))
        
        while left < right:
            mid = (left + right) // 2
            with SATSolver(name="Minicard", bootstrap_with=clauses) as solver:
                solver.add_atmost(lits, mid)
                sat = solver.solve()
                if sat:
                    model = solver.get_model()
                    selected = [v for v in model if v > 0 and v <= m]
                    best_solution = selected
                    right = len(selected)
                else:
                    left = mid + 1
        
        return best_solution