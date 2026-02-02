import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        import ortools
        print(f"ortools dir: {dir(ortools)}")
        try:
            import ortools.graph
            print(f"ortools.graph dir: {dir(ortools.graph)}")
        except ImportError:
            print("ortools.graph import failed")
            
        n = len(problem["capacity"])
        return [[0]*n for _ in range(n)]