from scipy.linalg import lu

class Solver:
    def solve(self, problem, **kwargs):
        P, L, U = lu(problem["matrix"])
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}