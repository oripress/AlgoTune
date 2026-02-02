import networkx as nx
from pysat.solvers import Solver as SATSolver
from pysat.card import CardEnc
import bisect

class Solver:
    def solve(self, problem: list[list[float]], **kwargs) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0] if n == 1 else []
            
        # Extract edges and weights
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((problem[i][j], i, j))
        
        edges.sort(key=lambda x: x[0])
        unique_weights = sorted(list(set(e[0] for e in edges)))
        weight_to_idx = {w: i for i, w in enumerate(unique_weights)}
        
        # Lower bound
        min_bottleneck = 0
        for i in range(n):
            row = sorted([problem[i][j] for j in range(n) if i != j])
            if len(row) >= 2:
                if row[1] > min_bottleneck:
                    min_bottleneck = row[1]
        
        low_idx = bisect.bisect_left(unique_weights, min_bottleneck)
        high_idx = len(unique_weights) - 1
        
        # Nearest Neighbor for Upper Bound
        current_node = 0
        visited = {0}
        tour = [0]
        nn_bottleneck = 0
        possible = True
        for _ in range(n - 1):
            best_dist = float('inf')
            next_node = -1
            for j in range(n):
                if j not in visited:
                    if problem[current_node][j] < best_dist:
                        best_dist = problem[current_node][j]
                        next_node = j
            if next_node != -1:
                visited.add(next_node)
                tour.append(next_node)
                nn_bottleneck = max(nn_bottleneck, best_dist)
                current_node = next_node
            else:
                possible = False
                break
        
        best_tour = []
        if possible:
            nn_bottleneck = max(nn_bottleneck, problem[current_node][0])
            tour.append(0)
            ub_idx = bisect.bisect_right(unique_weights, nn_bottleneck) - 1
            if ub_idx < high_idx:
                high_idx = ub_idx
            best_tour = tour

        # SAT Solver Setup
        with SATSolver(name='g4') as sat:
            # Variables
            # x_{p, c}: city c at position p. p in 1..n-1, c in 1..n-1
            # Base vars: 1 .. (n-1)^2
            num_pos_vars = (n - 1) ** 2
            
            def get_var(p, c):
                return (p - 1) * (n - 1) + (c - 1) + 1
            
            # Weight variables: one per unique weight
            # Indices: num_pos_vars + 1 .. num_pos_vars + len(unique_weights)
            w_vars_start = num_pos_vars + 1
            
            def get_w_var(w_idx):
                return w_vars_start + w_idx

            # 1. Permutation Constraints
            # Each position has exactly one city
            for p in range(1, n):
                vars_p = [get_var(p, c) for c in range(1, n)]
                top = max(sat.nof_vars(), w_vars_start + len(unique_weights))
                sat.append_formula(CardEnc.equals(vars_p, bound=1, top_id=top))

            # Each city is at exactly one position
            for c in range(1, n):
                vars_c = [get_var(p, c) for p in range(1, n)]
                top = max(sat.nof_vars(), w_vars_start + len(unique_weights))
                sat.append_formula(CardEnc.equals(vars_c, bound=1, top_id=top))
            
            # 2. Symmetry Breaking: city at pos 1 < city at pos n-1
            for c1 in range(1, n):
                for c2 in range(1, c1):
                    sat.add_clause([-get_var(1, c1), -get_var(n - 1, c2)])
            
            # 3. Transition Constraints linked to Weight Variables
            # Step 0 -> Step 1
            for c in range(1, n):
                w = problem[0][c]
                w_idx = weight_to_idx[w]
                if w_idx <= low_idx:
                    pass
                elif w_idx > high_idx:
                    sat.add_clause([-get_var(1, c)])
                else:
                    sat.add_clause([-get_var(1, c), get_w_var(w_idx)])
            
            # Step n-1 -> Step 0
            for c in range(1, n):
                w = problem[c][0]
                w_idx = weight_to_idx[w]
                if w_idx <= low_idx:
                    pass
                elif w_idx > high_idx:
                    sat.add_clause([-get_var(n - 1, c)])
                else:
                    sat.add_clause([-get_var(n - 1, c), get_w_var(w_idx)])
            
            # Step p -> Step p+1
            for p in range(1, n - 1):
                for c1 in range(1, n):
                    for c2 in range(1, n):
                        if c1 == c2: continue
                        w = problem[c1][c2]
                        w_idx = weight_to_idx[w]
                        if w_idx <= low_idx:
                            pass
                        elif w_idx > high_idx:
                            sat.add_clause([-get_var(p, c1), -get_var(p + 1, c2)])
                        else:
                            sat.add_clause([-get_var(p, c1), -get_var(p + 1, c2), get_w_var(w_idx)])
            
            # Binary Search with Assumptions
            while low_idx <= high_idx:
                mid_idx = (low_idx + high_idx) // 2
                
                # Assume all weights > mid_idx are forbidden
                assumptions = [-get_w_var(i) for i in range(mid_idx + 1, len(unique_weights))]
                
                if sat.solve(assumptions=assumptions):
                    # Found a solution
                    model = sat.get_model()
                    res_tour = [0] * n
                    true_vars = set(v for v in model if v > 0)
                    for p in range(1, n):
                        for c in range(1, n):
                            if get_var(p, c) in true_vars:
                                res_tour[p] = c
                                break
                    res_tour.append(0)
                    best_tour = res_tour
                    high_idx = mid_idx - 1
                else:
                    low_idx = mid_idx + 1
                    
        return best_tour