from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver

class Solver:
    def solve(self, problem):
        """
        Finds the minimum vertex cover using SAT solver with binary search.
        """
        n = len(problem)
        if n == 0:
            return []
        
        # Check if there are any edges
        has_edge = False
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    has_edge = True
                    break
            if has_edge:
                break
        
        if not has_edge:
            return []
        
        # Create CNF for vertex cover with at most k vertices
        def vertex_cover_sat(adj_matrix, k):
            n = len(adj_matrix)
            cnf = CNF()
            
            # For each edge (i,j), at least one endpoint must be in cover: x_i ∨ x_j
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_matrix[i][j] == 1:
                        cnf.append([i + 1, j + 1])  # Variables are 1-indexed in SAT
            
            # At most k vertices in the cover
            atmost_k = CardEnc.atmost(
                lits=[i + 1 for i in range(n)], 
                bound=k, 
                encoding=EncType.seqcounter
            )
            cnf.extend(atmost_k.clauses)
            
            return cnf
        
        # Binary search for minimum vertex cover size
        left, right = 0, n
        best_solution = list(range(n))  # Worst case: all vertices
        
        while left < right:
            mid = (left + right) // 2
            cnf = vertex_cover_sat(problem, mid)
            
            with SATSolver(name="Minicard") as solver:
                solver.append_formula(cnf)
                if solver.solve():
                    model = solver.get_model()
                    # Extract vertices in cover (positive literals)
                    solution = [i for i in range(n) if model[i] > 0]
                    best_solution = solution
                    right = mid
                else:
                    left = mid + 1
        
        return sorted(best_solution)