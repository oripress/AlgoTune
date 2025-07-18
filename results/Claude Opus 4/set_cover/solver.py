from ortools.linear_solver import pywraplp
import itertools

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the set cover problem using a hybrid approach.
        
        :param problem: A list of subsets (each subset is a list of integers).
        :return: A list of indices (1-indexed) of the selected subsets.
        """
        if not problem:
            return []
            
        # Find universe
        universe = set()
        for subset in problem:
            universe.update(subset)
        
        if not universe:
            return []
        
        n = len(problem)
        
        # For small problems, use exhaustive search
        if n <= 20:
            return self._exhaustive_search(problem, universe)
        
        # For medium problems, try greedy first with optimality check
        greedy_sol = self._greedy_solve(problem, universe)
        
        # Check if greedy is likely optimal using a lower bound
        lower_bound = self._compute_lower_bound(problem, universe)
        if len(greedy_sol) == lower_bound:
            return greedy_sol
        
        # Otherwise use ILP with CBC (faster than SCIP)
        return self._ilp_solve(problem, universe)
    
    def _exhaustive_search(self, problem: list[list[int]], universe: set) -> list[int]:
        """Exhaustive search for small instances."""
        n = len(problem)
        best_size = n + 1
        best_solution = []
        
        # Try all possible combinations
        for r in range(1, n + 1):
            if r >= best_size:
                break
            for combo in itertools.combinations(range(n), r):
                # Check if this combination covers the universe
                covered = set()
                for idx in combo:
                    covered.update(problem[idx])
                if covered == universe:
                    if r < best_size:
                        best_size = r
                        best_solution = [i + 1 for i in combo]  # 1-indexed
                    break  # Found solution of size r, no need to check other combos of same size
        
        return best_solution
    
    def _compute_lower_bound(self, problem: list[list[int]], universe: set) -> int:
        """Compute a simple lower bound on the optimal solution size."""
        # Maximum element frequency gives a lower bound
        max_freq = 0
        for elem in universe:
            freq = sum(1 for subset in problem if elem in subset)
            max_freq = max(max_freq, freq)
        
        # Lower bound: universe size divided by max subset size
        max_subset_size = max(len(subset) for subset in problem) if problem else 1
        return max(1, len(universe) // max_subset_size, len(universe) // max_freq if max_freq > 0 else 1)
    
    def _ilp_solve(self, problem: list[list[int]], universe: set) -> list[int]:
        """Solve using Integer Linear Programming with CBC solver."""
        # Create the CBC solver directly
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            # Fallback to any available solver
            solver = pywraplp.Solver.CreateSolver('SCIP')
        
        # Variables: x[i] = 1 if subset i is selected
        x = []
        for i in range(len(problem)):
            x.append(solver.IntVar(0, 1, f'x[{i}]'))
        
        # Constraints: each element must be covered
        for element in universe:
            constraint = solver.RowConstraint(1, solver.infinity(), f'cover_{element}')
            for i, subset in enumerate(problem):
                if element in subset:
                    constraint.SetCoefficient(x[i], 1)
        
        # Objective: minimize the number of selected subsets
        objective = solver.Objective()
        for i in range(len(problem)):
            objective.SetCoefficient(x[i], 1)
        objective.SetMinimization()
        
        # Set time limit to avoid long runs
        solver.SetTimeLimit(5000)  # 5 seconds
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            selected = []
            for i in range(len(problem)):
                if x[i].solution_value() > 0.5:
                    selected.append(i + 1)  # 1-indexed
            return selected
        else:
            # Fallback to greedy if solver fails
            return self._greedy_solve(problem, universe)
    
    def _greedy_solve(self, problem: list[list[int]], universe: set) -> list[int]:
        """Greedy algorithm that selects sets covering most uncovered elements."""
        subsets = [set(s) for s in problem]
        uncovered = universe.copy()
        selected = []
        
        while uncovered:
            best_idx = -1
            best_count = 0
            
            for i, subset in enumerate(subsets):
                if i + 1 in selected:
                    continue
                    
                count = len(subset & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx + 1)
            uncovered -= subsets[best_idx]
        
        return selected