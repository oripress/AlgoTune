import collections
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: tuple, **kwargs) -> Any:
        n, sets, conflicts = problem
        num_sets = len(sets)

        # --- Faster Dominated Set Reduction ---
        is_dominated = [False] * num_sets
        set_fs = [frozenset(s) for s in sets]

        conflicts_per_set = collections.defaultdict(set)
        for c_idx, conflict in enumerate(conflicts):
            for s_idx in conflict:
                conflicts_per_set[s_idx].add(c_idx)
        
        obj_to_sets_full = collections.defaultdict(list)
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets_full[obj].append(i)

        for i in range(num_sets):
            if is_dominated[i]: continue
            if not sets[i]:
                is_dominated[i] = True
                continue

            # Find potential dominators for set i
            potential_dominators = set(obj_to_sets_full[sets[i][0]])
            for obj_idx in range(1, len(sets[i])):
                potential_dominators.intersection_update(obj_to_sets_full[sets[i][obj_idx]])
            
            for j in potential_dominators:
                if i == j: continue

                if not conflicts_per_set[j].issubset(conflicts_per_set[i]):
                    continue

                if len(set_fs[i]) < len(set_fs[j]) or \
                   (len(set_fs[i]) == len(set_fs[j]) and len(conflicts_per_set[j]) < len(conflicts_per_set[i])) or \
                   (len(set_fs[i]) == len(set_fs[j]) and conflicts_per_set[j] == conflicts_per_set[i] and j < i):
                    is_dominated[i] = True
                    break
        
        active_sets_indices = [i for i, d in enumerate(is_dominated) if not d]
        if not active_sets_indices:
             active_sets_indices = list(range(num_sets))
             is_dominated = [False] * num_sets

        map_orig_to_active = {orig_idx: active_idx for active_idx, orig_idx in enumerate(active_sets_indices)}

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"set_{i}") for i in active_sets_indices]

        obj_to_sets = collections.defaultdict(list)
        for i in active_sets_indices:
            for obj in sets[i]:
                obj_to_sets[obj].append(i)

        for obj in range(n):
            if obj in obj_to_sets and len(obj_to_sets[obj]) == 1:
                set_idx = obj_to_sets[obj][0]
                model.Add(set_vars[map_orig_to_active[set_idx]] == 1)

        for obj in range(n):
            if obj in obj_to_sets:
                vars_for_obj = [set_vars[map_orig_to_active[s_idx]] for s_idx in obj_to_sets[obj]]
                if vars_for_obj:
                    model.Add(sum(vars_for_obj) >= 1)

        for conflict in conflicts:
            vars_for_conflict = [set_vars[map_orig_to_active[s_idx]] for s_idx in conflict if not is_dominated[s_idx]]
            if len(vars_for_conflict) > 1:
                model.Add(sum(vars_for_conflict) < len(vars_for_conflict))

        model.Minimize(sum(set_vars))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [active_sets_indices[i] for i, var in enumerate(set_vars) if solver.Value(var) == 1]
            return solution
        else:
            return self.solve_no_reduction(problem)

    def solve_no_reduction(self, problem: tuple) -> list[int]:
        n, sets, conflicts = problem
        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(len(sets))]
        obj_to_sets = collections.defaultdict(list)
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(i)
        for obj in range(n):
            model.Add(sum(set_vars[i] for i in obj_to_sets[obj]) >= 1)
        for c in conflicts:
            if len(c) > 1: model.Add(sum(set_vars[i] for i in c) < len(c))
        model.Minimize(sum(set_vars))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, v in enumerate(set_vars) if solver.Value(v) == 1]
        return []