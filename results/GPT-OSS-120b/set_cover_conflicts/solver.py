from __future__ import annotations
from typing import Any, List, Tuple
import itertools
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Tuple[int, List[List[int]], List[List[int]]], **kwargs) -> List[int]:
        """
        Solve the set cover with conflicts problem using OR‑Tools CP‑SAT.

        Parameters
        ----------
        problem : tuple
            (n, sets, conflicts)
            - n : number of objects (0 .. n‑1)
            - sets : list of sets, each a list of object indices.
                     Trivial singleton sets are guaranteed to be present and never in conflicts.
            - conflicts : list of conflicts, each a list of set indices that cannot be all selected.

        Returns
        -------
        list[int]
            Indices of selected sets forming an optimal cover.
        """
        # Unpack problem; support both raw tuple and Instance‑like objects
        if isinstance(problem, tuple):
            n, sets, conflicts = problem
        else:  # pragma: no cover
            n, sets, conflicts = problem.n, problem.sets, problem.conflicts

        model = cp_model.CpModel()

        # Boolean variable for each set
        set_vars = [model.NewBoolVar(f"s{i}") for i in range(len(sets))]

        # Coverage constraints: each object must be covered by at least one selected set
        for obj in range(n):
            covering = [set_vars[i] for i, s in enumerate(sets) if obj in s]
            # At least one covering set
            model.Add(sum(covering) >= 1)

        # Conflict constraints: at most one set from each conflict group may be chosen
        for conflict in conflicts:
            # conflict is a list of set indices that cannot be all selected together
            # The requirement is that not all of them are 1 simultaneously.
            # This is equivalent to AtMostOne for the whole group.
            model.AddAtMostOne([set_vars[i] for i in conflict])

        # Objective: minimize total number of selected sets
        model.Minimize(sum(set_vars))

        # Solve
        solver = cp_model.CpSolver()
        # A modest time limit (e.g., 10 seconds) to stay within the 10× budget
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = [i for i, var in enumerate(set_vars) if solver.Value(var) == 1]
            return solution
        else:
            # Fallback: trivial singleton cover (always feasible)
            return list(range(n))