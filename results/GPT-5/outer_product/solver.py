from numpy import multiply as _mul
_outer = _mul.outer

class Solver:
    __slots__ = ()
    def solve(self, problem, _outer=_outer, **kwargs):
        a, b = problem
        return _outer(a, b)