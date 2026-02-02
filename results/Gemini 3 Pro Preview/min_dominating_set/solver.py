from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the minimum dominating set problem using PySAT (RC2).
        """
        n = len(problem)
        wcnf = WCNF()
        
        # Variables are 1..n. x_i corresponds to variable i+1
        
        # Hard clauses: Domination
        # For each node i, at least one neighbor (or itself) must be selected
        for i in range(n):
            clause = [i + 1] # Self
            for j, val in enumerate(problem[i]):
                if val:
                    clause.append(j + 1)
            wcnf.append(clause) # Hard clause
        
        # Soft clauses: Minimize selected nodes
        # Add soft clause (not x_i) with weight 1.
        # If x_i is selected (True), clause is False, incurring cost 1.
        for i in range(n):
            wcnf.append([-(i + 1)], weight=1)
            
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            
        if model:
            # Filter positive literals which correspond to selected nodes
            return [i - 1 for i in model if i > 0]
        return []