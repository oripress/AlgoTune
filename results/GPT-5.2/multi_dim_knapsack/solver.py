from __future__ import annotations

from typing import Any, Sequence

# -----------------------------
# OR-Tools imports (module-level)
# -----------------------------
_cp_model = None
_HAVE_CP = False
try:
    from ortools.sat.python import cp_model as _cp_model  # type: ignore

    _HAVE_CP = True
except Exception:  # pragma: no cover
    _cp_model = None
    _HAVE_CP = False

# Knapsack backend detection (varies across OR-Tools versions).
_pks = None
_HAVE_KS = False
_KS_SOLVER_CLS = None
_KS_TYPE_BB = None
_KS_TYPE_DP = None
_KS_INIT_NAME = None
_KS_SOLVE_NAME = None
_KS_CONTAINS_NAME = None

try:  # Prefer legacy wrapper if present.
    from ortools.algorithms import pywrapknapsack_solver as _pks  # type: ignore

    _KS_SOLVER_CLS = _pks.KnapsackSolver
    _KS_TYPE_BB = _pks.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER
    _KS_TYPE_DP = _pks.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
    _KS_INIT_NAME = "Init"
    _KS_SOLVE_NAME = "Solve"
    _KS_CONTAINS_NAME = "BestSolutionContains"
    _HAVE_KS = True
except Exception:  # pragma: no cover
    try:  # Newer python wrapper
        from ortools.algorithms.python import knapsack_solver as _pks  # type: ignore

        _KS_SOLVER_CLS = getattr(_pks, "KnapsackSolver", None) or getattr(_pks, "Knapsack", None)

        # Solver types may be in an enum SolverType or as module constants.
        if hasattr(_pks, "SolverType"):
            st = _pks.SolverType
            _KS_TYPE_BB = getattr(st, "KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER", None)
            _KS_TYPE_DP = getattr(st, "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER", None)
        else:
            _KS_TYPE_BB = getattr(_pks, "KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER", None)
            _KS_TYPE_DP = getattr(_pks, "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER", None)

        if _KS_SOLVER_CLS is not None:
            _KS_INIT_NAME = "Init" if hasattr(_KS_SOLVER_CLS, "Init") else "init"
            _KS_SOLVE_NAME = "Solve" if hasattr(_KS_SOLVER_CLS, "Solve") else "solve"
            if hasattr(_KS_SOLVER_CLS, "BestSolutionContains"):
                _KS_CONTAINS_NAME = "BestSolutionContains"
            elif hasattr(_KS_SOLVER_CLS, "best_solution_contains"):
                _KS_CONTAINS_NAME = "best_solution_contains"
            else:
                _KS_CONTAINS_NAME = None

        _HAVE_KS = _KS_SOLVER_CLS is not None and _KS_TYPE_BB is not None and _KS_CONTAINS_NAME is not None
    except Exception:  # pragma: no cover
        _pks = None
        _HAVE_KS = False

# Reuse solver objects across calls (Init()/init resets the instance).
_KS_BB = None
_KS_DP = None

class _Instance:
    __slots__ = ("value", "demand", "supply")

    def __init__(self, value: Sequence[int], demand: Any, supply: Sequence[int]) -> None:
        self.value = value
        self.demand = demand
        self.supply = supply

def _parse_problem(problem: Any) -> _Instance | None:
    if problem is None:
        return None
    if hasattr(problem, "value") and hasattr(problem, "demand") and hasattr(problem, "supply"):
        return _Instance(problem.value, problem.demand, problem.supply)
    if isinstance(problem, (tuple, list)) and len(problem) == 3:
        try:
            v, d, s = problem
            return _Instance(v, d, s)
        except Exception:
            return None
    return None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        inst = _parse_problem(problem)
        if inst is None:
            return []

        value = inst.value
        demand = inst.demand
        supply = inst.supply

        try:
            n = len(value)
            if n == 0:
                return []
            k = len(supply)
        except Exception:
            return []

        if k == 0:
            # No constraints: take all strictly positive items.
            return [i for i, vi in enumerate(value) if vi > 0]

        # Keep caps as list-like (avoid per-element int() unless needed).
        caps = supply if isinstance(supply, list) else list(supply)

        # Build weights/profits, filter:
        # - non-positive profit items
        # - items infeasible alone
        weights = [[] for _ in range(k)]
        profits: list[Any] = []
        kept_idx: list[int] = []
        total_w = [0] * k

        w_lists = weights
        tw = total_w
        cap0 = caps
        p_append = profits.append
        idx_append = kept_idx.append
        pop = list.pop

        for i in range(n):
            vi = value[i]
            if vi <= 0:
                continue
            di = demand[i]
            # single pass over dimensions; rollback appends on failure
            for r in range(k):
                wr = di[r]
                if wr > cap0[r]:
                    for rr in range(r):
                        w = pop(w_lists[rr])
                        tw[rr] -= w
                    break
                w_lists[r].append(wr)
                tw[r] += wr
            else:
                p_append(vi)
                idx_append(i)

        if not profits:
            return []

        # Take-all shortcut
        for r in range(k):
            if tw[r] > cap0[r]:
                break
        else:
            return kept_idx

        # Fast exact knapsack backend (C++).
        if _HAVE_KS:
            global _KS_BB, _KS_DP

            # Prefer DP for 1D if available and capacity not huge.
            if k == 1 and _KS_TYPE_DP is not None and cap0[0] <= 200_000:
                if _KS_DP is None:
                    _KS_DP = _KS_SOLVER_CLS(_KS_TYPE_DP, "kp_dp")  # type: ignore[misc]
                ks = _KS_DP
            else:
                if _KS_BB is None:
                    _KS_BB = _KS_SOLVER_CLS(_KS_TYPE_BB, "kp_bb")  # type: ignore[misc]
                ks = _KS_BB

            init = getattr(ks, _KS_INIT_NAME)
            solve = getattr(ks, _KS_SOLVE_NAME)
            contains = getattr(ks, _KS_CONTAINS_NAME)

            try:
                init(profits, weights, caps)
            except Exception:
                # Rare fallback: cast everything to plain Python ints.
                caps_i = [int(c) for c in caps]
                profits_i = [int(p) for p in profits]
                weights_i = [[int(w) for w in wr] for wr in weights]
                init(profits_i, weights_i, caps_i)
                caps = caps_i  # keep for consistency

            solve()
            kept = kept_idx
            m = len(kept)
            return [kept[j] for j in range(m) if contains(j)]

        # Fallback exact method (should be rare).
        if _HAVE_CP:
            model = _cp_model.CpModel()
            x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
            for r in range(k):
                model.Add(sum(x[i] * int(demand[i][r]) for i in range(n)) <= int(caps[r]))
            model.Maximize(sum(x[i] * int(value[i]) for i in range(n)))
            solver = _cp_model.CpSolver()
            status = solver.Solve(model)
            if status in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
                return [i for i in range(n) if solver.Value(x[i])]

        return []

        # Fallback exact method (should be rare).
        if _HAVE_CP:
            model = _cp_model.CpModel()
            x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
            for r in range(k):
                model.Add(sum(x[i] * int(demand[i][r]) for i in range(n)) <= caps[r])
            model.Maximize(sum(x[i] * int(value[i]) for i in range(n)))
            solver = _cp_model.CpSolver()
            status = solver.Solve(model)
            if status in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
                return [i for i in range(n) if solver.Value(x[i])]

        return []