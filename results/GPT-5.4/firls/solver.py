import inspect

from scipy import signal

_PRINTED = False

class Solver:
    def solve(self, problem, **kwargs):
        global _PRINTED
        if not _PRINTED:
            src = inspect.getsource(signal.firls).splitlines()
            for i, line in enumerate(src, 1):
                print(f"{i:03d}: {line}")
            _PRINTED = True
        n, edges = problem
        n = 2 * int(n) + 1
        return signal.firls(n, (0.0, *tuple(edges), 1.0), [1, 1, 0, 0])