from typing import Any, List
from ortools.sat.python import cp_model

def _lsb_index(x: int) -> int:
    return (x & -x).bit_length() - 1

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        if problem is None:
            return []
        n = len(problem)
        if n <= 1:
            return [0] if n == 1 else []

        # Simple fast path (works best for small/medium graphs)
        SIMPLE_THRESHOLD = 60
        if n <= SIMPLE_THRESHOLD:
            return self._solve_simple(problem)

        # Large graphs: bitset-based light preprocessing + reduced CP-SAT
        return self._solve_reduced(problem)

    def _solve_simple(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Domination constraints
        for i in range(n):
            row = problem[i]
            terms = [x[i]]
            for j, v in enumerate(row):
                if v == 1:
                    terms.append(x[j])
            model.Add(sum(terms) >= 1)

        # Greedy upper bound (fast)
        greedy = self._greedy_adj(problem)
        model.Minimize(sum(x))
        # Add UB
        if greedy and len(greedy) < n:
            model.Add(sum(x) <= len(greedy))
            # Add hints one by one (compatible across OR-Tools versions)
            for j in greedy:
                try:
                    model.AddHint(x[j], 1)
                except Exception:
                    break

        solver = cp_model.CpSolver()
        try:
            solver.parameters.num_search_workers = 8
        except Exception:
            pass
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        return greedy if greedy else []

    def _solve_reduced(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)
        # Build closed neighborhoods as bitsets: N[i] = {i} U neighbors(i)
        N = [0] * n
        for i, row in enumerate(problem):
            mask = 1 << i
            for j, v in enumerate(row):
                if v:
                    mask |= 1 << j
            N[i] = mask

        all_mask = (1 << n) - 1
        uncovered = all_mask
        avail_vars = all_mask
        forced_mask = 0
        # Leaf forcing: for leaves u (deg=1) with neighbor v where deg(v) > 1, force v
        deg = [N[i].bit_count() - 1 for i in range(n)]
        to_force_leaf = 0
        for u in range(n):
            if deg[u] == 1:
                nb = N[u] & ~(1 << u)
                v = _lsb_index(nb)
                if deg[v] > 1:
                    to_force_leaf |= 1 << v
        if to_force_leaf:
            forced_mask |= to_force_leaf
            covered = 0
            mm = to_force_leaf
            while mm:
                lb = mm & -mm
                v = _lsb_index(lb)
                covered |= N[v]
                mm ^= lb
            uncovered &= ~covered
            avail_vars &= ~to_force_leaf
            # Remove variables that don't cover any uncovered vertex
            new_avail = 0
            mm = avail_vars
            while mm:
                lb = mm & -mm
                v = _lsb_index(lb)
                if N[v] & uncovered:
                    new_avail |= 1 << v
                mm ^= lb
            avail_vars = new_avail
        # Phase 1: Unique dominator forcing on original domain
        while True:
            if uncovered == 0:
                break
            to_force = 0
            m = uncovered
            while m:
                lb = m & -m
                i = _lsb_index(lb)
                allowed = N[i] & avail_vars
                if allowed and (allowed & (allowed - 1)) == 0:
                    to_force |= allowed
                m ^= lb
            if to_force == 0:
                break
            # Apply forces
            forced_mask |= to_force
            covered = 0
            mm = to_force
            while mm:
                lb = mm & -mm
                v = _lsb_index(lb)
                covered |= N[v]
                mm ^= lb
            uncovered &= ~covered
            avail_vars &= ~to_force
            # Remove variables that don't cover any uncovered vertex
            new_avail = 0
            mm = avail_vars
            while mm:
                lb = mm & -mm
                v = _lsb_index(lb)
                if N[v] & uncovered:
                    new_avail |= 1 << v
                mm ^= lb
            avail_vars = new_avail

        if uncovered == 0:
            # All dominated by forced picks
            res = []
            mm = forced_mask
            while mm:
                lb = mm & -mm
                res.append(_lsb_index(lb))
                mm ^= lb
            return sorted(res)

        # Build reduced candidate list C and residual universe U
        C_list: List[int] = []
        mm = avail_vars
        while mm:
            lb = mm & -mm
            v = _lsb_index(lb)
            if N[v] & uncovered:
                C_list.append(v)
            mm ^= lb

        U_list: List[int] = []
        mm = uncovered
        while mm:
            lb = mm & -mm
            U_list.append(_lsb_index(lb))
            mm ^= lb

        if not C_list:
            # Fallback: greedy on residual
            res = self._greedy_from_bitsets(N, uncovered)
            selected = []
            mm = forced_mask
            while mm:
                lb = mm & -mm
                selected.append(_lsb_index(lb))
                mm ^= lb
            selected.extend(res)
            return sorted(set(selected))

        # Compress U indices to 0..mU-1
        u_pos = {u: i for i, u in enumerate(U_list)}
        mU = len(U_list)

        # Candidate coverage over compressed U
        cover_bits = []
        for v in C_list:
            cov = N[v] & uncovered
            cb = 0
            mm2 = cov
            while mm2:
                lb2 = mm2 & -mm2
                u = _lsb_index(lb2)
                cb |= 1 << u_pos[u]
                mm2 ^= lb2
            cover_bits.append(cb)

        # Phase 2: Eliminate duplicate and dominated candidates over U
        # 2a. Remove duplicates in coverage
        seen = {}
        uniq_candidates: List[int] = []
        uniq_covers: List[int] = []
        for j, cb in enumerate(cover_bits):
            if cb == 0:
                continue
            if cb in seen:
                continue
            seen[cb] = j
            uniq_candidates.append(C_list[j])
            uniq_covers.append(cb)

        # 2b. Remove candidates whose coverage is subset of another's coverage (keep maximal)
        order = sorted(range(len(uniq_covers)), key=lambda i: uniq_covers[i].bit_count(), reverse=True)
        kept_idx: List[int] = []
        kept_covers: List[int] = []
        for idx in order:
            cb = uniq_covers[idx]
            # If cb is subset of any kept cover, skip
            skip = False
            for kc in kept_covers:
                if cb & ~kc == 0:
                    skip = True
                    break
            if not skip:
                kept_idx.append(idx)
                kept_covers.append(cb)

        active_candidates = [uniq_candidates[i] for i in kept_idx]
        cover_active = [uniq_covers[i] for i in kept_idx]

        # Phase 3: Unique dominator forcing on reduced instance
        U_mask = (1 << mU) - 1
        forced_reduced: List[int] = []  # store original vertex ids

        def drop_empty_and_reindex(cands: List[int], covers: List[int], u_mask: int):
            new_cands: List[int] = []
            new_covers: List[int] = []
            for v, cb in zip(cands, covers):
                cbr = cb & u_mask
                if cbr:
                    new_cands.append(v)
                    new_covers.append(cbr)
            return new_cands, new_covers

        active_candidates, cover_active = drop_empty_and_reindex(active_candidates, cover_active, U_mask)

        while True:
            to_force_idx = set()
            if U_mask == 0 or not active_candidates:
                break
            # For each u, find if exactly one candidate covers it
            mmU = U_mask
            while mmU:
                lbu = mmU & -mmU
                u = _lsb_index(lbu)
                last = -1
                count = 0
                for j, cb in enumerate(cover_active):
                    if (cb >> u) & 1:
                        last = j
                        count += 1
                        if count >= 2:
                            break
                if count == 1:
                    to_force_idx.add(last)
                mmU ^= lbu
            if not to_force_idx:
                break
            # Apply forces
            covered = 0
            new_active_candidates: List[int] = []
            new_cover_active: List[int] = []
            for j, (v, cb) in enumerate(zip(active_candidates, cover_active)):
                if j in to_force_idx:
                    forced_reduced.append(v)
                    covered |= cb
                else:
                    new_active_candidates.append(v)
                    new_cover_active.append(cb)
            U_mask &= ~covered
            active_candidates, cover_active = drop_empty_and_reindex(new_active_candidates, new_cover_active, U_mask)

        if U_mask == 0:
            # Everything covered by forced picks
            selected = []
            # forced from original
            mm = forced_mask
            while mm:
                lb = mm & -mm
                selected.append(_lsb_index(lb))
                mm ^= lb
            # forced from reduced
            selected.extend(forced_reduced)
            return sorted(set(selected))

        # Phase 4: Constraint reduction (duplicate and subsumed constraints)
        # Build allowed masks for remaining U positions
        # Candidate indices are 0..k-1 where k = len(active_candidates)
        k = len(active_candidates)
        allowed_by = [0] * mU
        for j, cb in enumerate(cover_active):
            mmc = cb & U_mask
            while mmc:
                lbc = mmc & -mmc
                u = _lsb_index(lbc)
                allowed_by[u] |= 1 << j
                mmc ^= lbc

        # Only consider U positions still active (in U_mask) and with nonzero allowed
        final_us = []
        mmU = U_mask
        while mmU:
            lbu = mmU & -mmU
            u = _lsb_index(lbu)
            if allowed_by[u] != 0:
                final_us.append(u)
            mmU ^= lbu

        # Remove duplicate allowed sets and subsumed ones (keep minimal sets)
        # Map mask -> representative u
        seen_allowed = {}
        unique_us = []
        for u in final_us:
            mask = allowed_by[u]
            if mask not in seen_allowed:
                seen_allowed[mask] = u
                unique_us.append(u)

        # Subsumption: if allowed(u1) subseteq allowed(u2), drop u2
        unique_us_sorted = sorted(unique_us, key=lambda u: allowed_by[u].bit_count())
        keep_u = [True] * len(unique_us_sorted)
        for i in range(len(unique_us_sorted)):
            if not keep_u[i]:
                continue
            ui = unique_us_sorted[i]
            mi = allowed_by[ui]
            for j in range(i + 1, len(unique_us_sorted)):
                if not keep_u[j]:
                    continue
                uj = unique_us_sorted[j]
                mj = allowed_by[uj]
                if mi & ~mj == 0:
                    keep_u[j] = False
        final_us2 = [unique_us_sorted[i] for i in range(len(unique_us_sorted)) if keep_u[i]]

        # Prepare greedy UB on reduced instance
        U_need_mask = 0
        for u in final_us2:
            U_need_mask |= 1 << u
        greedy_indices = self._greedy_cover_indices(cover_active, U_need_mask)

        # Build CP-SAT on reduced instance
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{active_candidates[j]}") for j in range(len(active_candidates))]

        # Coverage constraints for final_us2 only
        for u in final_us2:
            terms = []
            for j, cb in enumerate(cover_active):
                if (cb >> u) & 1:
                    terms.append(x[j])
            if terms:
                model.Add(sum(terms) >= 1)

        # Objective
        model.Minimize(sum(x))

        # Upper bound using greedy
        ub_total = forced_mask.bit_count() + len(forced_reduced) + len(greedy_indices)
        if x and ub_total < n:
            model.Add(sum(x) <= len(greedy_indices))
            # Hints
            for j in greedy_indices:
                try:
                    model.AddHint(x[j], 1)
                except Exception:
                    break

        solver = cp_model.CpSolver()
        try:
            solver.parameters.num_search_workers = 8
        except Exception:
            pass
        status = solver.Solve(model)

        # Collect solution
        selected = []
        # forced from original
        mm = forced_mask
        while mm:
            lb = mm & -mm
            selected.append(_lsb_index(lb))
            mm ^= lb
        # forced from reduced
        selected.extend(forced_reduced)
        # solver picks
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for j, v in enumerate(active_candidates):
                if solver.Value(x[j]) == 1:
                    selected.append(v)
        else:
            # fallback to greedy residual
            for j in greedy_indices:
                selected.append(active_candidates[j])

        return sorted(set(selected))

    @staticmethod
    def _greedy_adj(problem: List[List[int]]) -> List[int]:
        # Fast greedy dominating set using bitsets
        n = len(problem)
        if n == 0:
            return []
        # Build closed neighborhoods as bitsets
        N = [0] * n
        for i, row in enumerate(problem):
            mask = 1 << i
            for j, v in enumerate(row):
                if v:
                    mask |= 1 << j
            N[i] = mask

        remaining = (1 << n) - 1
        selected: List[int] = []
        while remaining:
            best = -1
            best_gain = -1
            for v in range(n):
                gain = (N[v] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best = v
            selected.append(best)
            remaining &= ~N[best]
        return selected

    @staticmethod
    def _greedy_from_bitsets(N: List[int], U_mask: int) -> List[int]:
        # Greedy dominating set using bitsets, variables are all vertices
        n = len(N)
        remaining = U_mask
        selected: List[int] = []
        while remaining:
            best = -1
            best_gain = -1
            for v in range(n):
                gain = (N[v] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best = v
            selected.append(best)
            remaining &= ~N[best]
        return selected

    @staticmethod
    def _greedy_cover_indices(cover_bits: List[int], U_mask: int) -> List[int]:
        # Greedy set cover over given cover_bits (each covers subset of U_mask)
        remaining = U_mask
        selected_idx: List[int] = []
        while remaining:
            best = -1
            best_gain = -1
            for j, cb in enumerate(cover_bits):
                gain = (cb & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best = j
            if best == -1 or best_gain <= 0:
                # Pick any variable covering the first remaining u
                u = _lsb_index(remaining)
                for j, cb in enumerate(cover_bits):
                    if (cb >> u) & 1:
                        best = j
                        break
                if best == -1:
                    break
            selected_idx.append(best)
            remaining &= ~cover_bits[best]
        return selected_idx