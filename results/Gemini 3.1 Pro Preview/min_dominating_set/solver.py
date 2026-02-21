from typing import Any
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        n = len(problem)
        wcnf = WCNF()
        
        for i, row in enumerate(problem):
            clause = [i + 1] + [j + 1 for j, val in enumerate(row) if val]
            wcnf.append(clause)
            wcnf.append([-(i + 1)], weight=1)
            
        with RC2(wcnf, solver='m22') as rc2:
            model = rc2.compute()
            
        if model:
            return [v - 1 for v in model if v > 0]
        return []