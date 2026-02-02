from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        if not problem:
            return []

        # Universe size
        max_elem = 0
        for s in problem:
            if s:
                m_val = max(s)
                if m_val > max_elem:
                    max_elem = m_val
        n = max_elem
        m = len(problem)

        wcnf = WCNF()
        
        # element_to_subsets[e] = list of subset indices (1-based)
        element_to_subsets = [[] for _ in range(n + 1)]
        for i, subset in enumerate(problem):
            var = i + 1
            for elem in subset:
                element_to_subsets[elem].append(var)
        
        # Hard constraints: Cover every element
        for e in range(1, n + 1):
            if element_to_subsets[e]:
                wcnf.append(element_to_subsets[e])
            else:
                return []

        # Soft constraints: Minimize selected subsets
        # We add soft clause [-x_i] with weight 1.
        # Maximizing satisfied soft clauses => Maximizing number of FALSE x_i => Minimizing number of TRUE x_i.
        for i in range(1, m + 1):
            wcnf.append([-i], weight=1)
            
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            
        if model:
            return [i for i in model if i > 0]
        return []
        # Run MILP
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if res.success:
            # Extract indices
            return [i + 1 for i, val in enumerate(res.x) if val > 0.5]
        else:
            return []
        
        solver = cp_model.CpSolver()
        # Use multiple workers for potential speedup
        solver.parameters.num_search_workers = 4
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Return 1-based indices
            return [i + 1 for i in range(num_subsets) if solver.Value(x[i])]
        else:
            return []