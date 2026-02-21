from typing import Any
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF, IDPool
from pysat.card import CardEnc
from itertools import combinations

class Solver:
    def solve(self, problem: Any, **kwargs) -> list[int]:
        if hasattr(problem, "n"):
            n = problem.n
            sets = problem.sets
            conflicts = problem.conflicts
        else:
            n, sets, conflicts = problem

        hard = []
        soft = []
        wght = []
        
        num_sets = len(sets)
        vpool = IDPool(start_from=num_sets + 1)
        
        obj_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            var = i + 1
            for obj in s:
                obj_to_sets[obj].append(var)
            soft.append([-var])
            wght.append(1)
            
        hard.extend(obj_to_sets)
            
        for conflict in conflicts:
            if len(conflict) <= 4:
                hard.extend([-(u + 1), -(v + 1)] for u, v in combinations(conflict, 2))
            else:
                lits = [i + 1 for i in conflict]
                enc = CardEnc.atmost(lits=lits, bound=1, vpool=vpool)
                hard.extend(enc.clauses)
                
        wcnf = WCNF()
        wcnf.hard = hard
        wcnf.soft = soft
        wcnf.wght = wght
        wcnf.nv = vpool.top
        with RC2(wcnf, solver='cd') as rc2:
            model = rc2.compute()
            if model:
                return [v - 1 for v in model if v > 0 and v <= num_sets]
            
        raise ValueError("No feasible solution found.")