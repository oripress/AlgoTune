from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        return {"exponential": expm(problem["matrix"]).tolist()}