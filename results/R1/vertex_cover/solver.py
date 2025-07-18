from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as PySatSolver
import time
import numpy as np
from numba import jit

@jit(nopython=True)
def kernelize(problem):
    n = len(problem)
    cover = set()
    removed = np.zeros(n, dtype=np.bool_)
    degree = problem.sum(axis=1)
    
    # Process isolated vertices
    for i in range(n):
        if degree[i] == 0:
            removed[i] = True
    
    # Process degree-1 vertices
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not removed[i] and degree[i] == 1:
                # Find neighbor
                for j in range(n):
                    if not removed[j] and problem[i, j]:
                        break
                else:
                    continue
                
                # Add neighbor to cover
                cover.add(j)
                removed[j] = True
                removed[i] = True
                
                # Update degrees
                for k in range(n):
                    if problem[j, k] and not removed[k]:
                        degree[k] -= 1
                        if degree[k] == 1:
                            changed = True
                break
    
    remaining = [i for i in range(n) if not removed[i]]
    return cover, remaining

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        
        # Convert to numpy for faster processing
        problem_np = np.array(problem, dtype=np.int8)
        
        # Apply kernelization
        cover, remaining = kernelize(problem_np)
        m = len(remaining)
        
        # Solve reduced problem
        if m == 0:
            return list(cover)
        
        # Build reduced adjacency matrix
        reduced_problem = problem_np[np.ix_(remaining, remaining)]
        
        # Solve reduced problem
        reduced_cover = self._solve_sat(reduced_problem)
        
        # Combine results
        full_cover = cover | {remaining[i] for i in reduced_cover}
        return list(full_cover)
    
    def _solve_sat(self, problem):
        n = len(problem)
        if n == 0:
            return []
        
        # Build base CNF with edge constraints
        base_cnf = CNF()
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    base_cnf.append([i + 1, j + 1])
        
        # Binary search for minimal cover
        left = 0
        right = n
        best_cover = list(range(n))
        start_time = time.time()
        
        try:
            while left < right:
                if time.time() - start_time > 1.5:  # Tighter timeout
                    break
                    
                mid = (left + right) // 2
                
                # Use Glucose4 which is generally faster
                solver = PySatSolver(name="glucose4", bootstrap_with=base_cnf)
                
                # Add cardinality constraint
                enc = CardEnc.atmost(
                    lits=[i + 1 for i in range(n)], 
                    bound=mid, 
                    encoding=EncType.seqcounter
                )
                solver.append_formula(enc.clauses)
                
                # Solve with current constraints
                if solver.solve():
                    model = solver.get_model()
                    cover = [i for i in range(n) if model[i] > 0]
                    if len(cover) < len(best_cover):
                        best_cover = cover
                        # Early termination if we found a small cover
                        if len(cover) <= 1:
                            solver.delete()
                            break
                    right = len(cover)
                else:
                    left = mid + 1
                
                solver.delete()
                
            return best_cover
        except Exception:
            return list(range(n))