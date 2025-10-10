from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Glucose4

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Optimized set cover using SAT with Glucose4 solver.
        """
        if not problem:
            return []
        
        m = len(problem)
        
        # Build element -> subset mapping once
        element_to_subsets = {}
        for i, subset in enumerate(problem):
            for e in subset:
                if e not in element_to_subsets:
                    element_to_subsets[e] = []
                element_to_subsets[e].append(i + 1)  # 1-indexed
        
        if not element_to_subsets:
            return []
        
        # Quick greedy upper bound
        subsets_sets = [set(s) for s in problem]
        uncovered = set(element_to_subsets.keys())
        greedy_sol = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(m):
                if i + 1 not in greedy_sol:
                    count = len(subsets_sets[i] & uncovered)
                    if count > best_count:
                        best_count = count
                        best_idx = i
            
            if best_idx == -1:
                break
            greedy_sol.append(best_idx + 1)
            uncovered -= subsets_sets[best_idx]
        
        # Binary search for minimum k
        left = 1
        right = len(greedy_sol)
        best_solution = greedy_sol
        
        # Precompute coverage constraints (reusable)
        coverage_clauses = [list(covers) for covers in element_to_subsets.values()]
        
        while left < right:
            mid = (left + right) // 2
            
            # Build CNF efficiently
            cnf = CNF()
            
            # Add coverage constraints
            for clause in coverage_clauses:
                cnf.append(clause)
            
            # Cardinality: at most mid subsets
            lits = list(range(1, m + 1))
            atmost = CardEnc.atmost(lits=lits, bound=mid, encoding=EncType.seqcounter)
            cnf.extend(atmost.clauses)
            
            # Solve
            with Glucose4(bootstrap_with=cnf) as solver:
                if solver.solve():
                    model = solver.get_model()
                    selected = [i for i in range(1, m + 1) if i in model]
                    best_solution = selected
                    right = len(selected)
                else:
                    left = mid + 1
        
        return best_solution