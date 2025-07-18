from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem):
        """
        Solves the Vertex Cover problem using Integer Linear Programming.
        
        :param problem: A 2d array (adjacency matrix) with 0/1 values
        :return: A list showing the indices of selected nodes
        """
        n = len(problem)
        
        # Trivial case: empty graph
        if n == 0:
            return []
        
        # Build edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    edges.append((i, j))
        
        # If no edges, return empty set
        if not edges:
            return []
        
        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # Fallback to greedy if solver not available
            return self._greedy_fallback(edges, n)
        
        # Create variables: x[i] = 1 if vertex i is in the cover
        x = {}
        for i in range(n):
            x[i] = solver.IntVar(0, 1, f'x_{i}')
        
        # Constraints: for each edge (i,j), at least one endpoint must be in cover
        for i, j in edges:
            solver.Add(x[i] + x[j] >= 1)
        
        # Objective: minimize the number of vertices in cover
        solver.Minimize(solver.Sum([x[i] for i in range(n)]))
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            # Extract solution
            cover = []
            for i in range(n):
                if x[i].solution_value() > 0.5:
                    cover.append(i)
            return sorted(cover)
        else:
            # Fallback to greedy if no optimal solution found
            return self._greedy_fallback(edges, n)
    
    def _greedy_fallback(self, edges, n):
        """Greedy algorithm as fallback."""
        cover = set()
        remaining_edges = list(edges)
        
        while remaining_edges:
            # Count degrees
            degrees = [0] * n
            for i, j in remaining_edges:
                if i not in cover:
                    degrees[i] += 1
                if j not in cover:
                    degrees[j] += 1
            
            # Pick highest degree vertex
            max_v = max(range(n), key=lambda v: degrees[v])
            if degrees[max_v] == 0:
                break
                
            cover.add(max_v)
            
            # Remove covered edges
            remaining_edges = [(i, j) for i, j in remaining_edges 
                              if i != max_v and j != max_v]
        
        return sorted(list(cover))