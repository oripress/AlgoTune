from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the set cover with conflicts problem using MaxSAT.
        
        Args:
            problem: A tuple (n, sets, conflicts) where:
                - n is the number of objects
                - sets is a list of sets (each set is a list of integers)
                - conflicts is a list of conflicts (each conflict is a list of set indices)
        
        Returns:
            A list of set indices that form a valid cover
        """
        n, sets, conflicts = problem
        
        # Create WCNF formula
        wcnf = WCNF()
        
        # Preprocess: Build object to sets mapping
        obj_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(i + 1)
        
        # Hard clauses: coverage constraints
        for obj_sets in obj_to_sets:
            wcnf.append(obj_sets)
        
        # Build conflict pairs (avoid duplicates)
        conflict_pairs = set()
        for conflict in conflicts:
            for i in range(len(conflict)):
                for j in range(i + 1, len(conflict)):
                    conflict_pairs.add((conflict[i], conflict[j]))
        
        # Hard clauses: conflict constraints
        for i, j in conflict_pairs:
            wcnf.append([-(i + 1), -(j + 1)])
        
        # Soft clauses: minimize number of selected sets
        for i in range(len(sets)):
            wcnf.append([-(i + 1)], weight=1)
        
        # Solve using RC2 with cadical backend (often faster)
        solver = RC2(wcnf, solver='cd')
        model = solver.compute()
        
        if model:
            solution = [abs(v) - 1 for v in model if v > 0]
            return solution
        else:
            raise ValueError("No feasible solution found.")