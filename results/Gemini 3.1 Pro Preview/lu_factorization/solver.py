import numpy as np
from scipy.linalg import lu
import inspect

with open("lu_source.py", "w") as f:
    f.write(inspect.getsource(lu))

class Solver:
    def solve(self, problem, **kwargs):
        return {"LU": dict(zip(("P", "L", "U"), lu(problem["matrix"], overwrite_a=True, check_finite=False)))}