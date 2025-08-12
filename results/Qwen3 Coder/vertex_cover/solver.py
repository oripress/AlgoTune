import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve Vertex Cover problem by finding Maximum Independent Set
        and taking its complement.
        """
        def mis_to_sat(adj_matrix, k):
            """Encode maximum independent set problem as SAT with at least k vertices"""
            n = len(adj_matrix)
            cnf = CNF()

            # Independence constraints: For all (i, j) with edge, add NOT(x_i) ∨ NOT(x_j)
            # This means at most one of connected vertices can be in independent set
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_matrix[i][j] == 1:
                        cnf.append([-(i + 1), -(j + 1)])  # NOT(x_i) ∨ NOT(x_j)

            # Cardinality constraint: at least k vertices in independent set
            # Use a more efficient encoding for small k values
            if k <= n // 2:
                # For small k, use at least encoding
                at_least_k = CardEnc.atleast(
                    lits=[i + 1 for i in range(n)], bound=k, encoding=EncType.seqcounter
                )
            else:
                # For large k, use at most encoding on the complement
                at_most_complement = CardEnc.atmost(
                    lits=[i + 1 for i in range(n)], bound=n - k, encoding=EncType.seqcounter
                )
                # This is equivalent to "at least k" on the original variables
                cnf.extend(at_most_complement.clauses)
                return cnf
                
            cnf.extend(at_least_k.clauses)
            return cnf

        try:
            n = len(problem)
            n = len(problem)
            # Early return for empty graph
            if n == 0:
                return []
                return []
            
            # Quick check for trivial cases
            if n == 1:
                return [] if problem[0][0] == 0 else [0]
            
            # Convert to numpy array for faster access
            adj_matrix = np.array(problem)
            
            # Check if there are any edges
            if np.sum(adj_matrix) == 0:
                return []
            
            # Precompute edges to avoid repeated access
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_matrix[i, j] == 1]
            
            # If no edges, return empty set
            if not edges:
                return []
            
            # Better initial bounds: use greedy approximation for upper bound
            # Greedy vertex cover approximation: 2-approximation
            vertex_cover_approx = set()
            edge_copy = edges[:]
            while edge_copy:
                # Pick an arbitrary edge
                u, v = edge_copy[0]
                # Add both endpoints to cover
                vertex_cover_approx.add(u)
                vertex_cover_approx.add(v)
                # Remove all edges incident to u or v
                edge_copy = [(x, y) for (x, y) in edge_copy if x != u and y != u and x != v and y != v]
            
            # Binary search bounds
            left, right = 0, min(len(vertex_cover_approx), n)
            vertex_cover = list(range(n))  # Default to all vertices
            
            # Use incremental SAT solving to reduce overhead
            cnf_base = mis_to_sat(problem, 0)  # Base constraints only
            
            with SATSolver(bootstrap_with=cnf_base.clauses, name="Minicard") as solver:
                while left < right:
                    mid = (left + right + 1) // 2
                    
                    # Add cardinality constraint incrementally
                    cnf_card = CardEnc.atleast(
                        lits=[i + 1 for i in range(n)], bound=mid, encoding=EncType.seqcounter
                    )
                    
                    # Temporarily add cardinality constraints
                    for clause in cnf_card.clauses:
                        solver.add_clause(clause)
                    
                    sat = solver.solve()
                    
                    if sat:
                        # Found an independent set of size >= mid
                        model = solver.get_model()
                        # Extract vertices in independent set
                        independent_set = [i for i in range(n) if model[i] > 0]
                        # Vertex cover is complement of maximum independent set
                        independent_set_set = set(independent_set)
                        vertex_cover = [i for i in range(n) if i not in independent_set_set]
                        left = len(independent_set)
                    else:
                        right = mid - 1
                    
                    # Remove cardinality constraints for next iteration
                    # Note: This is a simplification - in practice we'd need to track added clauses
                    # For now, we'll recreate the solver for each iteration to avoid complexity
                    solver.delete()
                    solver = SATSolver(bootstrap_with=cnf_base.clauses, name="Minicard")
                        
            return vertex_cover
        except Exception:
            # Fallback to all vertices
            return list(range(len(problem)))