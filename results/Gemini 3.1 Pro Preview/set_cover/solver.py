from typing import Any
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        wcnf = WCNF()
        num_subsets = len(problem)
        
        max_e = 0
        for subset in problem:
            for e in subset:
                if e > max_e:
                    max_e = e
                    
        element_to_subsets = [[] for _ in range(max_e + 1)]
        for i, subset in enumerate(problem, 1):
            for e in subset:
                element_to_subsets[e].append(i)
                
        for subsets in element_to_subsets:
            if subsets:
                wcnf.append(subsets)
                
        for i in range(1, num_subsets + 1):
            wcnf.append([-i], weight=1)
            
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            
        if model:
            return [i for i in model if i > 0]
        return []