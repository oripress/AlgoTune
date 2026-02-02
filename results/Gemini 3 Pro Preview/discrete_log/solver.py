from sympy.ntheory.residue_ntheory import discrete_log

class Solver:
    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        return {"x": discrete_log(p, h, g)}