from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as PySatSolver
from typing import List

class Solver:
    def solve(self, problem, **kwargs) -> List[int]:
        """
        Vertex cover solver using direct SAT encoding with optimized search.
        """
        
        try:
            n = len(problem)
            if n == 0:
                return []
            
            # Quick check for no edges
            has_edge = any(problem[i][j] == 1 for i in range(n) for j in range(i + 1, n))
            if not has_edge:
                return []
            
            # Pre-compute edges and literals
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if problem[i][j] == 1]
            literals = [i + 1 for i in range(n)]
            
            # Helper function to create CNF for vertex cover
            def create_vc_cnf(k):
                cnf = CNF()
                # Cover constraints: For all edges, at least one endpoint in cover
                cnf.extend([[i + 1, j + 1] for i, j in edges])
                # Cardinality constraint: at most k vertices in cover
                atmost_k = CardEnc.atmost(lits=literals, bound=k, encoding=EncType.seqcounter)
                cnf.extend(atmost_k.clauses)
                return cnf
            
            # Binary search for minimum vertex cover
            best_vc = list(range(n))
            left, right = 0, n
            
            while left <= right:
                mid = (left + right) // 2
                
                # Skip if we already have a better solution
                if mid >= len(best_vc):
                    right = mid - 1
                    continue
                
                cnf = create_vc_cnf(mid)
                with PySatSolver(name="Minicard") as solver:
                    solver.append_formula(cnf)
                    if solver.solve():
                        model = solver.get_model()
                        best_vc = [i for i in range(n) if model[i] > 0]
                        right = mid - 1
                    else:
                        left = mid + 1
            
            return sorted(best_vc)
            
        except Exception as e:
            return list(range(len(problem)))