from typing import Any, List, Dict, Set, Tuple

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Optimized exact Set Cover solver.

        Improvements over previous version:
        - Preprocessing: remove empty sets, collapse duplicates, select forced sets.
        - Group elements with identical covering-set signatures.
        - Precompute masks and weights.
        - Branch-and-bound with memoization and a stronger, cheap lower bound.
        - Avoid expensive per-node scans of all sets; branch on groups with fewest coverers.
        - Optional OR-Tools CP-SAT attempt for large instances to get good upper bounds quickly.
        """
        if not problem:
            return []

        # Convert to sets and filter empties, keep original indices
        orig_sets = [set(s) for s in problem]
        nonempty_orig_idx: List[int] = []
        nonempty_sets: List[Set[int]] = []
        for i, s in enumerate(orig_sets):
            if s:
                nonempty_orig_idx.append(i)
                nonempty_sets.append(s)
        if not nonempty_sets:
            return []

        # Collapse duplicate sets
        seen: Dict[frozenset, int] = {}
        comp_sets: List[Set[int]] = []
        comp_rep_orig_idx: List[int] = []
        for orig_idx, s in zip(nonempty_orig_idx, nonempty_sets):
            key = frozenset(s)
            if key not in seen:
                seen[key] = len(comp_sets)
                comp_sets.append(set(s))
                comp_rep_orig_idx.append(orig_idx)

        # Element -> list of comp set indices covering it
        elem_to_sets: Dict[int, List[int]] = {}
        for j, s in enumerate(comp_sets):
            for e in s:
                elem_to_sets.setdefault(e, []).append(j)

        # Sanity: every element must be covered
        for e, covers in elem_to_sets.items():
            if not covers:
                return []

        # Preprocessing forced sets (elements covered by only one active set)
        covered_elems: Set[int] = set()
        active = [True] * len(comp_sets)
        forced_selected_comp: Set[int] = set()

        while True:
            newly_forced: Set[int] = set()
            for e, covers in elem_to_sets.items():
                if e in covered_elems:
                    continue
                alive = -1
                cnt = 0
                for j in covers:
                    if active[j]:
                        cnt += 1
                        alive = j
                        if cnt > 1:
                            break
                if cnt == 1 and alive >= 0:
                    newly_forced.add(alive)
            newly_forced -= forced_selected_comp
            if not newly_forced:
                break
            for j in newly_forced:
                if not active[j]:
                    continue
                forced_selected_comp.add(j)
                active[j] = False
                for e in comp_sets[j]:
                    covered_elems.add(e)

        # Build reduced problem (only uncovered elements and active sets)
        new_sets: List[Set[int]] = []
        new_comp_to_orig: List[int] = []
        for j, s in enumerate(comp_sets):
            if not active[j]:
                continue
            s_rem = {e for e in s if e not in covered_elems}
            if s_rem:
                new_sets.append(s_rem)
                new_comp_to_orig.append(comp_rep_orig_idx[j])

        # If nothing left, return forced selections mapped to original indices (1-based)
        if not new_sets:
            return sorted({comp_rep_orig_idx[j] + 1 for j in forced_selected_comp})

        # Deduplicate reduced sets
        dedup: Dict[frozenset, int] = {}
        final_sets: List[Set[int]] = []
        final_comp_to_orig: List[int] = []
        for s, orig in zip(new_sets, new_comp_to_orig):
            key = frozenset(s)
            if key not in dedup:
                dedup[key] = len(final_sets)
                final_sets.append(s)
                final_comp_to_orig.append(orig)
        new_sets = final_sets
        new_comp_to_orig = final_comp_to_orig

        # Rebuild element -> sets map for reduced instance
        elem_to_sets2: Dict[int, List[int]] = {}
        for i, s in enumerate(new_sets):
            for e in s:
                elem_to_sets2.setdefault(e, []).append(i)
        for e, covers in elem_to_sets2.items():
            if not covers:
                return []

        # Group elements by identical coverer signatures (reduces universe size)
        sig_to_gid: Dict[Tuple[int, ...], int] = {}
        element_gid: Dict[int, int] = {}
        gid = 0
        for e, covers in elem_to_sets2.items():
            key = tuple(sorted(covers))
            if key not in sig_to_gid:
                sig_to_gid[key] = gid
                gid += 1
            element_gid[e] = sig_to_gid[key]
        n_groups = gid

        group_size = [0] * n_groups
        for e, g in element_gid.items():
            group_size[g] += 1

        # Build masks: for each set, which groups it covers
        masks_local: List[int] = [0] * len(new_sets)
        groups_per_set: List[List[int]] = [[] for _ in range(len(new_sets))]
        for e, g in element_gid.items():
            for j in elem_to_sets2[e]:
                masks_local[j] |= 1 << g
                groups_per_set[j].append(g)

        # Filter out any sets that cover no groups (shouldn't happen)
        filt_masks: List[int] = []
        filt_comp_to_orig: List[int] = []
        filt_groups_per_set: List[List[int]] = []
        for mask, orig, gps in zip(masks_local, new_comp_to_orig, groups_per_set):
            if mask:
                filt_masks.append(mask)
                filt_comp_to_orig.append(orig)
                # deduplicate groups_per_set entries
                filt_groups_per_set.append(sorted(set(gps)))
        masks_local = filt_masks
        new_comp_to_orig = filt_comp_to_orig
        groups_per_set = filt_groups_per_set
        m = len(masks_local)
        if m == 0:
            return sorted({comp_rep_orig_idx[j] + 1 for j in forced_selected_comp})

        # coverers_for_group
        coverers_for_group: List[List[int]] = [[] for _ in range(n_groups)]
        for si, mask in enumerate(masks_local):
            mm = mask
            while mm:
                lsb = mm & -mm
                b = lsb.bit_length() - 1
                coverers_for_group[b].append(si)
                mm &= mm - 1

        for g in range(n_groups):
            if not coverers_for_group[g]:
                return []

        # Precompute set full weights and per-group max set weight
        group_size_local = tuple(group_size)
        set_weight_full: List[int] = [0] * m
        for j in range(m):
            s = 0
            for g in groups_per_set[j]:
                s += group_size_local[g]
            set_weight_full[j] = s

        group_max_set_weight: List[int] = [0] * n_groups
        for g in range(n_groups):
            best = 0
            for j in coverers_for_group[g]:
                w = set_weight_full[j]
                if w > best:
                    best = w
            group_max_set_weight[g] = best

        # mask weight cache
        mask_weight_cache: Dict[int, int] = {}

        def weight_of_mask(mask: int) -> int:
            if mask == 0:
                return 0
            v = mask_weight_cache.get(mask)
            if v is not None:
                return v
            w = 0
            mm = mask
            while mm:
                lsb = mm & -mm
                b = lsb.bit_length() - 1
                w += group_size_local[b]
                mm &= mm - 1
            mask_weight_cache[mask] = w
            return w

        # Greedy initial solution (gives upper bound)
        universe_mask = (1 << n_groups) - 1
        def greedy_cover(full_mask: int) -> List[int]:
            uncovered = full_mask
            picked: List[int] = []
            # Precompute per-set current coverage weights via cache
            while uncovered:
                best_j = -1
                best_cov = 0
                for j, mask in enumerate(masks_local):
                    inter = mask & uncovered
                    if not inter:
                        continue
                    cov = weight_of_mask(inter)
                    if cov > best_cov:
                        best_cov = cov
                        best_j = j
                if best_j == -1:
                    return []
                picked.append(best_j)
                uncovered &= ~masks_local[best_j]
            return picked

        greedy_sel = greedy_cover(universe_mask)
        if not greedy_sel and universe_mask != 0:
            return []

        best_new_count = len(greedy_sel)
        best_new_sel = list(greedy_sel)

        # Optional CP-SAT attempt for large instances to get good upper bound / optimal
        try_cpsat = kwargs.get("try_cpsat", True)
        large_instance = m >= 80 or n_groups >= 80
        if try_cpsat and large_instance:
            try:
                from ortools.sat.python import cp_model
                model = cp_model.CpModel()
                vars_x = [model.NewBoolVar(f"x{i}") for i in range(m)]
                for g in range(n_groups):
                    covers = coverers_for_group[g]
                    model.Add(sum(vars_x[j] for j in covers) >= 1)
                model.Minimize(sum(vars_x))
                solver = cp_model.CpSolver()
                workers = kwargs.get("workers", None)
                if workers is None:
                    import os
                    cpu_count = os.cpu_count() or 1
                    workers = min(8, max(1, cpu_count))
                solver.parameters.num_search_workers = int(workers)
                solver.parameters.random_seed = 1
                status = solver.Solve(model)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    sel = [j for j in range(m) if solver.Value(vars_x[j]) == 1]
                    if sel:
                        if len(sel) < best_new_count:
                            best_new_count = len(sel)
                            best_new_sel = sel
                        if status == cp_model.OPTIMAL:
                            chosen = [new_comp_to_orig[j] + 1 for j in best_new_sel]
                            forced_result = [comp_rep_orig_idx[j] + 1 for j in forced_selected_comp]
                            final = sorted(set(chosen + forced_result))
                            return final
            except Exception:
                pass

        # Branch-and-bound DFS with memoization
        memo: Dict[int, int] = {}
        # order groups by fewest coverers
        group_order = sorted(range(n_groups), key=lambda g: len(coverers_for_group[g]))

        # Local references for speed
        masks = masks_local
        coverers = coverers_for_group
        set_w_full = set_weight_full
        g_max_w = group_max_set_weight
        gw = group_size_local

        def dfs(uncovered_mask: int, depth: int, stack: List[int]):
            nonlocal best_new_count, best_new_sel, memo

            if uncovered_mask == 0:
                if depth < best_new_count:
                    best_new_count = depth
                    best_new_sel = list(stack)
                return

            prev = memo.get(uncovered_mask)
            if prev is not None and prev <= depth:
                return
            memo[uncovered_mask] = depth

            # remaining weight
            rem_weight = weight_of_mask(uncovered_mask)

            # cheap max coverage estimate: max group_max_set_weight over uncovered groups
            mm = uncovered_mask
            max_cov = 0
            while mm:
                lsb = mm & -mm
                g = lsb.bit_length() - 1
                mm &= mm - 1
                w = g_max_w[g]
                if w > max_cov:
                    max_cov = w
            if max_cov == 0:
                return
            lb = (rem_weight + max_cov - 1) // max_cov
            if depth + lb >= best_new_count:
                return

            # choose branching group: first uncovered in group_order
            chosen_g = -1
            for g in group_order:
                if (uncovered_mask >> g) & 1:
                    chosen_g = g
                    break
            if chosen_g == -1:
                return

            # candidate sets are coverers of chosen_g (they all intersect uncovered)
            cand = coverers[chosen_g]
            # sort by actual intersection weight (descending)
            cand_sorted = sorted(cand, key=lambda j: weight_of_mask(masks[j] & uncovered_mask), reverse=True)

            for j in cand_sorted:
                if depth + 1 >= best_new_count:
                    break
                stack.append(j)
                new_uncovered = uncovered_mask & ~masks[j]
                dfs(new_uncovered, depth + 1, stack)
                stack.pop()

        dfs(universe_mask, 0, [])

        # Map results back to original 1-based indices and include forced selections
        chosen = [new_comp_to_orig[j] + 1 for j in best_new_sel]
        forced_result = [comp_rep_orig_idx[j] + 1 for j in forced_selected_comp]
        final = sorted(set(chosen + forced_result))
        return final