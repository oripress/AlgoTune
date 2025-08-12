import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        X = scipy.linalg.sqrtm(problem["matrix"])
        return {"sqrtm": {"X": X.tolist()}}