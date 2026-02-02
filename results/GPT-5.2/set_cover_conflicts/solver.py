from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model

def _parse(problem: Any) -> tuple[int, list[list[int]], list[list[int]]]:
    """Accept either a raw tuple (n, sets, conflicts) or an Instance-like object."""
    if isinstance(problem, tuple) and len(problem) == 3:
        n, sets, conflicts = problem
        return int(n), list(sets), list(conflicts)
    # NamedTuple-like / Instance-like
    n = int(getattr(problem, "n"))
    sets = list(getattr(problem, "sets"))
    conflicts = list(getattr(problem, "conflicts"))
    return n, sets, conflicts

def _singleton_indices(n: int, sets: list[list[int]]) -> list[int]:
    """Indices of singleton sets [obj]. Guaranteed to exist and be conflict-free."""
    single = [-1] * n
    for i, s in enumerate(sets):
        if len(s) == 1:
            o = s[0]
            if 0 <= o < n and single[o] == -1:
                single[o] = i
    for o in range(n):
        if single[o] == -1:
            for i, s in enumerate(sets):
                if len(s) == 1 and s[0] == o:
                    single[o] = i
                    break
            if single[o] == -1:
                raise ValueError("Missing required singleton set.")
    return single

def _greedy_hint(
    n: int,
    sets: list[list[int]],
    conflicts: list[list[int]],
    singleton_idx: list[int],
) -> list[int]:
    """
    Fast greedy cover respecting conflict groups (AtMostOne per conflict list).
    Only used as a feasible hint for CP-SAT.
    """
    m = len(sets)

    # set -> conflict groups membership
    mem: list[list[int]] = [[] for _ in range(m)]
    for g, conf in enumerate(conflicts):
        for si in conf:
            if 0 <= si < m:
                mem[si].append(g)
    group_sel = [-1] * len(conflicts)

    if n <= 1024:
        masks = [0] * m
        sizes = [0] * m
        for i, s in enumerate(sets):
            mask = 0
            for o in s:
                if 0 <= o < n:
                    mask |= 1 << o
            masks[i] = mask
            sizes[i] = mask.bit_count()

        uncovered = (1 << n) - 1
        selected: list[int] = []

        order = sorted(range(m), key=lambda i: (sizes[i] <= 1, -sizes[i]))
        for i in order:
            if uncovered == 0:
                break
            ok = True
            for g in mem[i]:
                sel = group_sel[g]
                if sel != -1 and sel != i:
                    ok = False
                    break
            if not ok:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain <= 0:
                continue
            selected.append(i)
            for g in mem[i]:
                group_sel[g] = i
            uncovered &= ~masks[i]

        while uncovered:
            lsb = uncovered & -uncovered
            o = lsb.bit_length() - 1
            selected.append(singleton_idx[o])
            uncovered ^= lsb
        return selected

    uncovered_set = set(range(n))
    selected2: list[int] = []
    order2 = sorted(range(m), key=lambda i: (len(sets[i]) <= 1, -len(sets[i])))
    for i in order2:
        if not uncovered_set:
            break
        ok = True
        for g in mem[i]:
            sel = group_sel[g]
            if sel != -1 and sel != i:
                ok = False
                break
        if not ok:
            continue
        gain = 0
        for o in sets[i]:
            if o in uncovered_set:
                gain += 1
                if gain >= 2:
                    break
        if gain <= 0:
            continue
        selected2.append(i)
        for g in mem[i]:
            group_sel[g] = i
        for o in sets[i]:
            uncovered_set.discard(o)

    for o in uncovered_set:
        selected2.append(singleton_idx[o])
    return selected2

class Solver:
    def solve(self, problem: Any, **kwargs: Any) -> list[int]:
        n, sets, conflicts = _parse(problem)
        m = len(sets)

        # object -> sets covering it
        obj_to_sets: list[list[int]] = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for o in s:
                if 0 <= o < n:
                    obj_to_sets[o].append(i)

        singleton_idx = _singleton_indices(n, sets)

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"s{i}") for i in range(m)]

        # Coverage constraints (BoolOr is typically faster than linear sum>=1)
        for o in range(n):
            lits = obj_to_sets[o]
            if len(lits) == 1:
                model.Add(set_vars[lits[0]] == 1)
            else:
                model.AddBoolOr([set_vars[i] for i in lits])

        # Conflicts: AtMostOne for each conflict group (matches reference)
        for conf in conflicts:
            k = len(conf)
            if k <= 1:
                continue
            if k == 2:
                a, b = conf
                if 0 <= a < m and 0 <= b < m:
                    model.AddBoolOr([set_vars[a].Not(), set_vars[b].Not()])
            else:
                vars_conf = [set_vars[i] for i in conf if 0 <= i < m]
                if len(vars_conf) >= 2:
                    model.AddAtMostOne(vars_conf)

        model.Minimize(cp_model.LinearExpr.Sum(set_vars))

        # Feasible hint to accelerate optimization.
        if m <= 5000:
            hint = _greedy_hint(n, sets, conflicts, singleton_idx)
            hint_set = set(hint)
            for i, v in enumerate(set_vars):
                model.AddHint(v, 1 if i in hint_set else 0)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = int(kwargs.get("num_search_workers", 8))
        solver.parameters.random_seed = int(kwargs.get("random_seed", 0))
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        solver.parameters.symmetry_level = 2

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("No feasible solution found.")
        return [i for i in range(m) if solver.Value(set_vars[i]) == 1]