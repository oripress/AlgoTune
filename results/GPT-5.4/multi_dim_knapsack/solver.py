from typing import Any

try:
    from ortools.algorithms.python import knapsack_solver as _new_knapsack_solver
except Exception:  # pragma: no cover
    _new_knapsack_solver = None

try:
    from ortools.algorithms import pywrapknapsack_solver as _old_knapsack_solver
except Exception:  # pragma: no cover
    _old_knapsack_solver = None

try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover
    cp_model = None

class Solver:
    def __init__(self) -> None:
        self._backend = 0  # 0 none, 1 new snake, 2 new camel, 3 old camel
        self._solvers: dict[Any, Any] = {}

        self._new_multi = None
        self._new_bf = None
        self._new_64 = None
        self._new_dc = None
        self._new_dp = None

        self._old_multi = None
        self._old_bf = None
        self._old_64 = None
        self._old_dc = None
        self._old_dp = None

        if _new_knapsack_solver is not None:
            enum = getattr(_new_knapsack_solver, "SolverType", None)
            if enum is not None:
                self._new_multi = getattr(
                    enum, "KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER", None
                )
                self._new_bf = getattr(enum, "KNAPSACK_BRUTE_FORCE_SOLVER", None)
                self._new_64 = getattr(enum, "KNAPSACK_64ITEMS_SOLVER", None)
                self._new_dc = getattr(enum, "KNAPSACK_DIVIDE_AND_CONQUER_SOLVER", None)
                self._new_dp = getattr(enum, "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER", None)

        if _old_knapsack_solver is not None:
            cls = getattr(_old_knapsack_solver, "KnapsackSolver", None)
            if cls is not None:
                self._old_multi = getattr(
                    cls, "KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER", None
                )
                self._old_bf = getattr(cls, "KNAPSACK_BRUTE_FORCE_SOLVER", None)
                self._old_64 = getattr(cls, "KNAPSACK_64ITEMS_SOLVER", None)
                self._old_dc = getattr(cls, "KNAPSACK_DIVIDE_AND_CONQUER_SOLVER", None)
                self._old_dp = getattr(cls, "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER", None)

        if self._new_multi is not None:
            try:
                s = _new_knapsack_solver.KnapsackSolver(self._new_multi, "p")
                try:
                    s.init([1], [[1]], [1])
                    s.solve()
                    s.best_solution_contains(0)
                    self._backend = 1
                    return
                except Exception:
                    s.Init([1], [[1]], [1])
                    s.Solve()
                    s.BestSolutionContains(0)
                    self._backend = 2
                    return
            except Exception:
                pass

        if self._old_multi is not None:
            try:
                s = _old_knapsack_solver.KnapsackSolver(self._old_multi, "p")
                s.Init([1], [[1]], [1])
                s.Solve()
                s.BestSolutionContains(0)
                self._backend = 3
            except Exception:
                pass

    def _pick_type(self, k: int, n: int):
        if self._backend in (1, 2):
            if k == 1:
                if n <= 30 and self._new_bf is not None:
                    return self._new_bf
                if n <= 64 and self._new_64 is not None:
                    return self._new_64
                if self._new_dc is not None:
                    return self._new_dc
                if self._new_dp is not None:
                    return self._new_dp
            return self._new_multi

        if self._backend == 3:
            if k == 1:
                if n <= 30 and self._old_bf is not None:
                    return self._old_bf
                if n <= 64 and self._old_64 is not None:
                    return self._old_64
                if self._old_dc is not None:
                    return self._old_dc
                if self._old_dp is not None:
                    return self._old_dp
            return self._old_multi

        return None

    def _get_solver(self, solver_type):
        solver = self._solvers.get(solver_type)
        if solver is not None:
            return solver
        if self._backend in (1, 2):
            solver = _new_knapsack_solver.KnapsackSolver(solver_type, "k")
        else:
            solver = _old_knapsack_solver.KnapsackSolver(solver_type, "k")
        self._solvers[solver_type] = solver
        return solver

    def _solve_dedicated(self, values, demands, supplies):
        k = len(supplies)
        mandatory: list[int] = []
        kept_idx: list[int] = []
        kept_values: list[int] = []

        if k == 0:
            for i, v in enumerate(values):
                if v > 0:
                    mandatory.append(i)
            return mandatory

        if k == 1:
            cap0 = supplies[0]
            w0: list[int] = []
            total0 = 0
            for i, v in enumerate(values):
                if v <= 0:
                    continue
                d0 = demands[i][0]
                if d0 > cap0:
                    continue
                if d0 == 0:
                    mandatory.append(i)
                else:
                    kept_idx.append(i)
                    kept_values.append(v)
                    w0.append(d0)
                    total0 += d0
            if not kept_values:
                return mandatory
            if total0 <= cap0:
                mandatory.extend(kept_idx)
                return mandatory
            weights = [w0]

        elif k == 2:
            cap0 = supplies[0]
            cap1 = supplies[1]
            w0: list[int] = []
            w1: list[int] = []
            total0 = 0
            total1 = 0
            for i, v in enumerate(values):
                if v <= 0:
                    continue
                row = demands[i]
                d0 = row[0]
                d1 = row[1]
                if d0 > cap0 or d1 > cap1:
                    continue
                if d0 == 0 and d1 == 0:
                    mandatory.append(i)
                else:
                    kept_idx.append(i)
                    kept_values.append(v)
                    w0.append(d0)
                    w1.append(d1)
                    total0 += d0
                    total1 += d1
            if not kept_values:
                return mandatory
            if total0 <= cap0 and total1 <= cap1:
                mandatory.extend(kept_idx)
                return mandatory
            weights = [w0, w1]

        elif k == 3:
            cap0 = supplies[0]
            cap1 = supplies[1]
            cap2 = supplies[2]
            w0: list[int] = []
            w1: list[int] = []
            w2: list[int] = []
            total0 = 0
            total1 = 0
            total2 = 0
            for i, v in enumerate(values):
                if v <= 0:
                    continue
                row = demands[i]
                d0 = row[0]
                d1 = row[1]
                d2 = row[2]
                if d0 > cap0 or d1 > cap1 or d2 > cap2:
                    continue
                if d0 == 0 and d1 == 0 and d2 == 0:
                    mandatory.append(i)
                else:
                    kept_idx.append(i)
                    kept_values.append(v)
                    w0.append(d0)
                    w1.append(d1)
                    w2.append(d2)
                    total0 += d0
                    total1 += d1
                    total2 += d2
            if not kept_values:
                return mandatory
            if total0 <= cap0 and total1 <= cap1 and total2 <= cap2:
                mandatory.extend(kept_idx)
                return mandatory
            weights = [w0, w1, w2]

        elif k == 4:
            cap0 = supplies[0]
            cap1 = supplies[1]
            cap2 = supplies[2]
            cap3 = supplies[3]
            w0: list[int] = []
            w1: list[int] = []
            w2: list[int] = []
            w3: list[int] = []
            total0 = 0
            total1 = 0
            total2 = 0
            total3 = 0
            for i, v in enumerate(values):
                if v <= 0:
                    continue
                row = demands[i]
                d0 = row[0]
                d1 = row[1]
                d2 = row[2]
                d3 = row[3]
                if d0 > cap0 or d1 > cap1 or d2 > cap2 or d3 > cap3:
                    continue
                if d0 == 0 and d1 == 0 and d2 == 0 and d3 == 0:
                    mandatory.append(i)
                else:
                    kept_idx.append(i)
                    kept_values.append(v)
                    w0.append(d0)
                    w1.append(d1)
                    w2.append(d2)
                    w3.append(d3)
                    total0 += d0
                    total1 += d1
                    total2 += d2
                    total3 += d3
            if not kept_values:
                return mandatory
            if total0 <= cap0 and total1 <= cap1 and total2 <= cap2 and total3 <= cap3:
                mandatory.extend(kept_idx)
                return mandatory
            weights = [w0, w1, w2, w3]

        else:
            totals = [0] * k
            weights = [[] for _ in range(k)]
            for i, v in enumerate(values):
                if v <= 0:
                    continue
                row = demands[i]
                ok = True
                nonzero = False
                for r in range(k):
                    x = row[r]
                    if x > supplies[r]:
                        ok = False
                        break
                    if x:
                        nonzero = True
                if not ok:
                    continue
                if not nonzero:
                    mandatory.append(i)
                else:
                    kept_idx.append(i)
                    kept_values.append(v)
                    for r in range(k):
                        x = row[r]
                        weights[r].append(x)
                        totals[r] += x
            if not kept_values:
                return mandatory
            if all(totals[r] <= supplies[r] for r in range(k)):
                mandatory.extend(kept_idx)
                return mandatory

        n = len(kept_values)
        solver_type = self._pick_type(k, n)
        if solver_type is None:
            return None

        try:
            solver = self._get_solver(solver_type)
            if self._backend == 1:
                solver.init(kept_values, weights, supplies)
                solver.solve()
                contains = solver.best_solution_contains
            elif self._backend == 2:
                solver.Init(kept_values, weights, supplies)
                solver.Solve()
                contains = solver.BestSolutionContains
            else:
                solver.Init(kept_values, weights, supplies)
                solver.Solve()
                contains = solver.BestSolutionContains

            for i in range(n):
                if contains(i):
                    mandatory.append(kept_idx[i])
            return mandatory
        except Exception:
            self._solvers.pop(solver_type, None)
            return None

    def _solve_cpsat(self, values, demands, supplies):
        if cp_model is None:
            return []
        try:
            n = len(values)
            k = len(supplies)
            model = cp_model.CpModel()
            x = [model.NewBoolVar("x") for _ in range(n)]
            for r in range(k):
                model.Add(sum(x[i] * int(demands[i][r]) for i in range(n)) <= int(supplies[r]))
            model.Maximize(sum(x[i] * int(values[i]) for i in range(n)))
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return [i for i in range(n) if solver.Value(x[i])]
        except Exception:
            return []
        return []

    def solve(self, problem, **kwargs) -> Any:
        try:
            if (
                hasattr(problem, "value")
                and hasattr(problem, "demand")
                and hasattr(problem, "supply")
            ):
                values = problem.value
                demands = problem.demand
                supplies = problem.supply
            elif isinstance(problem, (list, tuple)) and len(problem) == 3:
                values, demands, supplies = problem
            else:
                return []
        except Exception:
            return []

        if not values:
            return []

        if self._backend:
            result = self._solve_dedicated(values, demands, supplies)
            if result is not None:
                return result

        return self._solve_cpsat(values, demands, supplies)