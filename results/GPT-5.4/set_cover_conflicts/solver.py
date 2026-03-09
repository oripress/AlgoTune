from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n, sets, conflicts = problem

        if n <= 0:
            return []

        singleton_idx = [-1] * n
        multi_orig: list[int] = []
        obj_to_multi: list[list[int]] = [[] for _ in range(n)]
        orig_to_multi: dict[int, int] = {}

        for i, s in enumerate(sets):
            if len(s) == 1:
                obj = s[0]
                if 0 <= obj < n and singleton_idx[obj] == -1:
                    singleton_idx[obj] = i
                continue

            seen_mask = 0
            uniq_objs = []
            for obj in s:
                bit = 1 << obj
                if (seen_mask & bit) == 0:
                    seen_mask |= bit
                    uniq_objs.append(obj)

            if len(uniq_objs) <= 1:
                continue

            j = len(multi_orig)
            multi_orig.append(i)
            orig_to_multi[i] = j
            for obj in uniq_objs:
                obj_to_multi[obj].append(j)

        m = len(multi_orig)
        if m == 0:
            return [singleton_idx[obj] for obj in range(n)]

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"s{j}") for j in range(m)]
        u = [model.NewBoolVar(f"u{o}") for o in range(n)]

        for obj in range(n):
            covers = obj_to_multi[obj]
            if covers:
                model.Add(sum(x[j] for j in covers) + u[obj] >= 1)
            else:
                model.Add(u[obj] == 1)

        for conflict in conflicts:
            mapped = []
            seen = set()
            for idx in conflict:
                j = orig_to_multi.get(idx)
                if j is not None and j not in seen:
                    seen.add(j)
                    mapped.append(j)
            if len(mapped) > 1:
                model.AddAtMostOne(x[j] for j in mapped)

        model.Minimize(sum(x) + sum(u))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            raise ValueError("No feasible solution found.")

        out = [multi_orig[j] for j in range(m) if solver.Value(x[j])]
        for obj in range(n):
            if solver.Value(u[obj]):
                idx = singleton_idx[obj]
                if idx < 0:
                    raise ValueError("No feasible solution found.")
                out.append(idx)
        return out