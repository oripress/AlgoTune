from typing import Any, List

# PySAT (python-sat) imports
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Set Cover with Conflicts problem.

        Args:
            problem: A tuple (n, sets, conflicts) where:
                - n: number of objects
                - sets: list of lists, each inner list is the set of objects covered by that set
                - conflicts: list of lists, each inner list contains indices of sets that cannot be chosen together

        Returns:
            A list of selected set indices forming an optimal cover.
        """
        # Extract problem data
        if isinstance(problem, tuple):
            n, sets_list, conflicts = problem
        else:
            # Fallback for potential NamedTuple-like Instance
            n, sets_list, conflicts = problem  # type: ignore

        m = len(sets_list)
        if m == 0:
            return []

        # Build mapping: for each object, which set variables cover it
        # Variable ids for SAT are 1-based: var_id = set_index + 1
        covered_by: List[List[int]] = [[] for _ in range(n)]
        for si, s in enumerate(sets_list):
            vid = si + 1
            for obj in s:
                if 0 <= obj < n:
                    covered_by[obj].append(vid)

        # Ensure every object is coverable (should be guaranteed)
        for obj in range(n):
            if not covered_by[obj]:
                # No set covers this object; as a safe fallback select its trivial set if exists
                # Search for a singleton set [obj]
                for si, s in enumerate(sets_list):
                    if len(s) == 1 and s[0] == obj:
                        return [si]
                # Otherwise, return trivial cover using any set that contains obj (none here) -> infeasible
                # Per problem statement this shouldn't happen
                raise ValueError("Object {} is not covered by any set.".format(obj))

        # Build WCNF: hard constraints for coverage and conflicts; soft clauses (¬x_i) to minimize number of sets
        wcnf = WCNF()

        # Coverage constraints: for each object, at least one covering set must be chosen
        for obj in range(n):
            lits = covered_by[obj]
            # Add as hard clause (OR of all lits)
            wcnf.append(lits)

        # Conflict constraints: at most one set from each conflict group
        # Use pairwise encoding for small groups; for larger, use ladder encoding to keep it compact.
        top_id = m  # highest variable id so far
        for group in conflicts:
            # Map set indices to SAT variable ids (1-based)
            lits = [idx + 1 for idx in group if 0 <= idx < m]
            if len(lits) <= 1:
                continue
            if len(lits) <= 8:
                # Pairwise at-most-one: for all i < j: (¬xi ∨ ¬xj)
                for i in range(len(lits)):
                    li = lits[i]
                    for j in range(i + 1, len(lits)):
                        lj = lits[j]
                        wcnf.append([-li, -lj])
            else:
                # Ladder encoding for at-most-one
                cnf = CardEnc.atmost(lits=lits, bound=1, top_id=top_id, encoding=EncType.ladder)
                top_id = max(top_id, cnf.nv)
                for clause in cnf.clauses:
                    wcnf.append(clause)

        # Objective: minimize number of chosen sets => add soft clauses (¬x_i) with weight 1
        for si in range(m):
            wcnf.append([-(si + 1)], weight=1)

        # Solve using RC2 MaxSAT solver
        with RC2(wcnf) as rc2:
            model = rc2.compute()

        # Extract chosen sets: variables x_i that are True in the model
        model_set = set(model)
        solution = [i for i in range(m) if (i + 1) in model_set]

        return solution