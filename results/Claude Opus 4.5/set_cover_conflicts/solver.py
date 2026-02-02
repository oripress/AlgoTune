from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

class Solver:
    def solve(self, problem, **kwargs):
        if isinstance(problem, tuple):
            n, sets, conflicts = problem
        else:
            n, sets, conflicts = problem.n, problem.sets, problem.conflicts
        
        num_sets = len(sets)
        
        # Precompute which sets cover each object
        obj_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(i)
        
        wcnf = WCNF()
        
        # Hard clauses: each object must be covered (at least one set)
        for obj in range(n):
            wcnf.append([i + 1 for i in obj_to_sets[obj]])
        
        # Hard clauses: conflict constraints (at most one from each conflict)
        for conflict in conflicts:
            clen = len(conflict)
            for i in range(clen):
                ci = conflict[i] + 1
                for j in range(i + 1, clen):
                    wcnf.append([-ci, -(conflict[j] + 1)])
        
        # Soft clauses: prefer not selecting each set (minimize count)
        for i in range(num_sets):
            wcnf.append([-(i + 1)], weight=1)
        
        # Try Cadical solver which is often faster
        try:
            with RC2(wcnf, solver='cd15', exhaust=True) as solver:
                model = solver.compute()
        except:
            with RC2(wcnf, solver='g4', exhaust=True) as solver:
                model = solver.compute()
        
        if model:
            model_set = set(model)
            return [i for i in range(num_sets) if (i + 1) in model_set]
        raise ValueError("No feasible solution found.")