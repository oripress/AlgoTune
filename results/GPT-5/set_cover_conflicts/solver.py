from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        # Unpack problem whether it's a tuple or a NamedTuple-like object
        try:
            n, sets_list, conflicts = problem
        except Exception:
            n = getattr(problem, "n")
            sets_list = getattr(problem, "sets")
            conflicts = getattr(problem, "conflicts")

        m = len(sets_list)

        # Identify singleton (trivial) sets [o] for each object o
        singleton_index = [-1] * n
        non_singleton_indices = []
        for idx, s in enumerate(sets_list):
            if len(s) == 1:
                o = s[0]
                if isinstance(o, int) and 0 <= o < n:
                    if singleton_index[o] == -1:
                        singleton_index[o] = idx
                    else:
                        # duplicate trivial set; ignore for selection purposes
                        pass
                    continue
            non_singleton_indices.append(idx)

        have_all_singletons = all(si != -1 for si in singleton_index)

        # Helper to reconstruct the final solution by adding necessary singletons
        def finalize_solution(chosen_non_singleton: list[int]) -> list[int]:
            covered = set()
            for i in chosen_non_singleton:
                covered.update(sets_list[i])
            result = list(chosen_non_singleton)
            # Add singleton for each uncovered object
            for o in range(n):
                if o not in covered:
                    si = singleton_index[o]
                    if si == -1:
                        # Should not happen if have_all_singletons is True; safeguard
                        # In the extreme case, fall back to picking any set covering o
                        # but per problem statement, singletons exist.
                        for j, s in enumerate(sets_list):
                            if o in s:
                                result.append(j)
                                covered.update(s)
                                break
                    else:
                        result.append(si)
            result.sort()
            return result

        # Try to use python-sat (Weighted MaxSAT). Fallback to OR-Tools if unavailable.
        try:
            from pysat.formula import WCNF
            from pysat.card import CardEnc, EncType
            from pysat.examples.rc2 import RC2

            if have_all_singletons:
                # Build reduced MaxSAT model over non-singleton sets only.
                # Vars: x_i for each non-singleton set i
                # Soft:
                #   - For each non-singleton x_i: (¬x_i) with weight 1  -> cost per chosen set
                #   - For each object o: (OR_{i covers o} x_i) with weight 1
                #     -> pay 1 if o not covered by any chosen non-singleton -> use its singleton
                # Hard:
                #   - For each conflict group: at most one among its non-singleton members
                k = len(non_singleton_indices)
                if k == 0:
                    # No non-singleton sets; choose all singletons
                    return finalize_solution([])

                # Map 1-based var ids to original set indices
                varid_by_orig = {}
                orig_by_varid = {}
                for vid, orig in enumerate(non_singleton_indices, start=1):
                    varid_by_orig[orig] = vid
                    orig_by_varid[vid] = orig

                wcnf = WCNF()

                # Soft: penalize selecting non-singletons
                for vid in range(1, k + 1):
                    wcnf.append([-vid], weight=1)

                # Soft: reward covering objects with non-singletons (penalize if not covered)
                # Build coverage lists using var ids
                cover_lits = [[] for _ in range(n)]
                for orig in non_singleton_indices:
                    vid = varid_by_orig[orig]
                    for o in sets_list[orig]:
                        if isinstance(o, int) and 0 <= o < n:
                            cover_lits[o].append(vid)
                for o in range(n):
                    lits = cover_lits[o]
                    # If lits is empty, append empty soft clause -> constant cost 1
                    wcnf.append(lits if lits else [], weight=1)

                # Hard: conflicts among non-singletons only
                top_id = k
                for grp in conflicts:
                    # Filter to non-singleton var ids
                    lits = [varid_by_orig[i] for i in grp if i in varid_by_orig]
                    if len(lits) <= 1:
                        continue
                    # Choose encoding by size: pairwise for very small, ladder otherwise
                    if len(lits) <= 5:
                        enc = CardEnc.atmost(lits=lits, bound=1, encoding=EncType.pairwise, top_id=top_id)
                    else:
                        enc = CardEnc.atmost(lits=lits, bound=1, encoding=EncType.ladder, top_id=top_id)
                    top_id = enc.nv
                    for cl in enc.clauses:
                        wcnf.append(cl)

                rc2 = RC2(wcnf)
                model = rc2.compute()
                if model is None:
                    raise ValueError("No feasible solution found.")

                chosen_non_singleton = [orig_by_varid[vid] for vid in range(1, k + 1) if vid in model]
                return finalize_solution(chosen_non_singleton)

            else:
                # Fall back to full MaxSAT model including all sets, as before
                wcnf = WCNF()
                # Soft: minimize number of selected sets
                for i in range(m):
                    wcnf.append([-(i + 1)], weight=1)

                # Hard: coverage constraints
                cover_lists = [[] for _ in range(n)]
                for si, s in enumerate(sets_list):
                    for o in s:
                        if 0 <= o < n:
                            cover_lists[o].append(si + 1)
                for o in range(n):
                    lits = cover_lists[o]
                    if lits:
                        wcnf.append(lits)

                # Hard: conflict constraints
                top_id = m
                for grp in conflicts:
                    if not grp or len(grp) <= 1:
                        continue
                    lits = [i + 1 for i in grp]
                    enc = CardEnc.atmost(lits=lits, bound=1, encoding=EncType.ladder, top_id=top_id)
                    top_id = enc.nv
                    for cl in enc.clauses:
                        wcnf.append(cl)

                rc2 = RC2(wcnf)
                model = rc2.compute()
                if model is None:
                    raise ValueError("No feasible solution found.")
                chosen = [i for i in range(m) if (i + 1) in model]
                chosen.sort()
                return chosen

        except Exception:
            # Fallback: OR-Tools CP-SAT
            from ortools.sat.python import cp_model  # type: ignore

            if have_all_singletons:
                # Variables: x_i for non-singletons, y_o for using singleton of object o
                model = cp_model.CpModel()
                k = len(non_singleton_indices)
                x = [model.NewBoolVar(f"x_{i}") for i in range(k)]
                y = [model.NewBoolVar(f"y_{o}") for o in range(n)]

                # Map from original non-singleton index to x position
                pos_by_orig = {orig: pos for pos, orig in enumerate(non_singleton_indices)}

                # Coverage: for each object, sum(x over covering sets) + y_o >= 1
                cover_pos = [[] for _ in range(n)]
                for orig, pos in pos_by_orig.items():
                    for o in sets_list[orig]:
                        if 0 <= o < n:
                            cover_pos[o].append(pos)
                for o in range(n):
                    # If cover_pos[o] is empty, then y_o must be 1
                    if cover_pos[o]:
                        model.Add(sum(x[p] for p in cover_pos[o]) + y[o] >= 1)
                    else:
                        model.Add(y[o] == 1)

                # Conflicts: at most one among non-singleton members
                for grp in conflicts:
                    xs = [x[pos_by_orig[i]] for i in grp if i in pos_by_orig]
                    if len(xs) > 1:
                        model.AddAtMostOne(xs)

                # Objective: minimize #non-singletons selected + #singletons used
                model.Minimize(sum(x) + sum(y))

                solver = cp_model.CpSolver()
                status = solver.Solve(model)
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    raise ValueError("No feasible solution found.")
                chosen_non_singleton = [
                    non_singleton_indices[p] for p in range(k) if solver.Value(x[p]) == 1
                ]
                # Add required singletons
                result = list(chosen_non_singleton)
                for o in range(n):
                    if solver.Value(y[o]) == 1:
                        result.append(singleton_index[o])
                result.sort()
                return result
            else:
                # Generic fallback (previous CP-SAT formulation)
                model = cp_model.CpModel()
                x = [model.NewBoolVar(f"s_{i}") for i in range(m)]

                # Coverage constraints
                cover_lists = [[] for _ in range(n)]
                for si, s in enumerate(sets_list):
                    for o in s:
                        if 0 <= o < n:
                            cover_lists[o].append(si)
                for o in range(n):
                    if cover_lists[o]:
                        model.Add(sum(x[i] for i in cover_lists[o]) >= 1)

                # Conflict constraints
                for grp in conflicts:
                    if len(grp) > 1:
                        model.AddAtMostOne(x[i] for i in grp)

                # Objective
                model.Minimize(sum(x))

                solver = cp_model.CpSolver()
                status = solver.Solve(model)
                if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    raise ValueError("No feasible solution found.")
                return [i for i in range(m) if solver.Value(x[i]) == 1]