from typing import Any, List, Optional, Tuple, Dict
import time

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[int]:
        """
        Solve Set Cover with Conflicts using an optimized branch-and-bound on bitmasks.

        problem: tuple (n, sets, conflicts)
            - n: number of elements (0..n-1)
            - sets: list of lists of element indices
            - conflicts: list of lists of set indices that cannot all be chosen together

        kwargs:
            - time_limit: optional float, seconds to limit search (best-effort)

        Returns:
            list of selected set indices forming an optimal cover (or best found within time_limit)
        """
        # Unpack problem (support tuple/list and object with attributes)
        if isinstance(problem, (tuple, list)):
            if len(problem) != 3:
                raise ValueError("Problem must be (n, sets, conflicts)")
            n, sets_list, conflicts = problem
        else:
            n = getattr(problem, "n", None)
            sets_list = getattr(problem, "sets", None)
            conflicts = getattr(problem, "conflicts", None)
            if n is None or sets_list is None or conflicts is None:
                raise ValueError("Problem object missing attributes (n, sets, conflicts)")

        # Quick exits
        if n is None or n <= 0:
            return []
        m = len(sets_list)
        if m == 0:
            return []

        # Time limit handling
        time_limit = kwargs.get("time_limit", None)
        try:
            if time_limit is not None:
                time_limit = float(time_limit)
        except Exception:
            time_limit = None
        t0 = time.perf_counter()

        def time_exceeded() -> bool:
            return time_limit is not None and (time.perf_counter() - t0) > time_limit

        # Build bitmask for each set and element->sets mapping
        masks: List[int] = [0] * m
        elem_to_sets: List[List[int]] = [[] for _ in range(n)]
        for i, s in enumerate(sets_list):
            mask = 0
            # use set(s) to ignore duplicate elements within a set
            for e in set(s):
                if 0 <= e < n:
                    bit = 1 << e
                    mask |= bit
                    elem_to_sets[e].append(i)
            masks[i] = mask

        full_mask = (1 << n) - 1

        # Ensure every element is coverable
        for e in range(n):
            if not elem_to_sets[e]:
                raise ValueError(f"No set covers element {e}")

        # Precompute set-index bits, conflict masks
        idx_bits: List[int] = [(1 << i) for i in range(m)]
        conflict_mask: List[int] = [0] * m
        for conf in conflicts:
            # sanitize indices
            valid = [int(x) for x in conf if 0 <= int(x) < m]
            if len(valid) <= 1:
                continue
            bits = 0
            for idx in valid:
                bits |= idx_bits[idx]
            for idx in valid:
                # mark other sets in this conflict as forbidden when idx is chosen
                conflict_mask[idx] |= bits & ~idx_bits[idx]

        # fast popcount
        popcount = int.bit_count

        # Precompute sizes and ordering (prefer larger sets then by index)
        sizes = [popcount(mask) for mask in masks]
        set_order = list(range(m))
        set_order.sort(key=lambda i: (-sizes[i], i))
        pos_map = {i: pos for pos, i in enumerate(set_order)}

        # Sort element's candidate sets by descending set size and global order
        for e in range(n):
            lst = elem_to_sets[e]
            lst.sort(key=lambda i: (-sizes[i], pos_map.get(i, 0)))
            elem_to_sets[e] = lst

        # Greedy initial solution (respect conflicts), optimized with local bindings
        def greedy_solution() -> List[int]:
            uncovered = full_mask
            banned = 0
            sol: List[int] = []
            masks_local = masks
            set_order_local = set_order
            conflict_mask_local = conflict_mask
            idx_bits_local = idx_bits
            popcount_local = popcount

            while uncovered:
                if time_exceeded():
                    break
                best_i = -1
                best_cov = 0
                # choose set covering most uncovered elements
                for i in set_order_local:
                    if banned & idx_bits_local[i]:
                        continue
                    cov = masks_local[i] & uncovered
                    if cov == 0:
                        continue
                    c = popcount_local(cov)
                    if c > best_cov:
                        best_cov = c
                        best_i = i
                        if best_cov == popcount_local(uncovered):
                            break
                if best_i == -1:
                    # fallback: pick a set that covers the first uncovered element
                    lsb = uncovered & -uncovered
                    e = lsb.bit_length() - 1
                    found = False
                    for i in elem_to_sets[e]:
                        if (banned & idx_bits_local[i]) == 0 and masks_local[i] == (1 << e):
                            sol.append(i)
                            uncovered &= ~masks_local[i]
                            banned |= idx_bits_local[i] | conflict_mask_local[i]
                            found = True
                            break
                    if found:
                        continue
                    for i in elem_to_sets[e]:
                        if (banned & idx_bits_local[i]) == 0:
                            sol.append(i)
                            uncovered &= ~masks_local[i]
                            banned |= idx_bits_local[i] | conflict_mask_local[i]
                            found = True
                            break
                    if not found:
                        break
                else:
                    sol.append(best_i)
                    banned |= idx_bits_local[best_i] | conflict_mask_local[best_i]
                    uncovered &= ~masks_local[best_i]
            return sol

        best_sol: List[int] = []
        greedy = greedy_solution()
        if greedy:
            best_sol = greedy.copy()
            best_size = len(greedy)
        else:
            best_size = n

        # Trivial solution using known singleton sets (guaranteed present)
        singletons: List[Optional[int]] = [None] * n
        for i, mask in enumerate(masks):
            if mask != 0 and (mask & (mask - 1)) == 0:
                e = mask.bit_length() - 1
                if 0 <= e < n and singletons[e] is None:
                    singletons[e] = i
        trivial: List[int] = []
        for e in range(n):
            if singletons[e] is not None:
                trivial.append(singletons[e])
            else:
                # fallback to the first set that contains e
                trivial.append(elem_to_sets[e][0])
        # Deduplicate trivial while preserving order
        seen = set()
        trivial_unique: List[int] = []
        for i in trivial:
            if i not in seen:
                seen.add(i)
                trivial_unique.append(i)
        trivial = trivial_unique

        if len(trivial) < best_size:
            best_sol = trivial.copy()
            best_size = len(trivial)

        # Branch-and-bound search with memoization (pack key as single int)
        visited: Dict[int, int] = {}
        m_shift = m  # number of bits to shift uncovered when packing key

        masks_local = masks
        conflict_mask_local = conflict_mask
        elem_to_sets_local = elem_to_sets
        set_order_local = set_order
        pos_map_local = pos_map
        idx_bits_local = idx_bits
        popcount_local = popcount

        # DFS with pruning
        def dfs(uncovered: int, banned: int, selected: List[int]):
            nonlocal best_size, best_sol
            if time_exceeded():
                return
            if uncovered == 0:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = selected.copy()
                return

            key = (uncovered << m_shift) | banned
            prev = visited.get(key)
            if prev is not None and prev <= len(selected):
                return
            visited[key] = len(selected)

            remaining = popcount_local(uncovered)
            # bound: maximum cover any available set can provide
            max_cover = 0
            for i in set_order_local:
                if banned & idx_bits_local[i]:
                    continue
                cov = masks_local[i] & uncovered
                if cov == 0:
                    continue
                c = popcount_local(cov)
                if c > max_cover:
                    max_cover = c
                    if max_cover == remaining:
                        break
            if max_cover == 0:
                return
            # lower bound on number of additional sets needed
            lb = (remaining + max_cover - 1) // max_cover
            if len(selected) + lb >= best_size:
                return

            # MRV: pick uncovered element with fewest available covering sets
            best_e = -1
            best_opts: List[int] = []
            best_count = 10 ** 9
            u = uncovered
            while u:
                lsb = u & -u
                e = lsb.bit_length() - 1
                u -= lsb
                opts: List[int] = []
                for i in elem_to_sets_local[e]:
                    if (banned & idx_bits_local[i]) != 0:
                        continue
                    if (masks_local[i] & uncovered) == 0:
                        continue
                    opts.append(i)
                c = len(opts)
                if c == 0:
                    return
                if c < best_count:
                    best_count = c
                    best_e = e
                    best_opts = opts
                    if c == 1:
                        break

            # order options by descending newly-covered count and by global order
            # prepare a small list of tuples to avoid repeated attribute lookups in key
            tmp = []
            for i in best_opts:
                cov = masks_local[i] & uncovered
                tmp.append((-(popcount_local(cov)), pos_map_local.get(i, 0), i))
            tmp.sort()
            ordered_opts = [t[2] for t in tmp]

            for i in ordered_opts:
                if time_exceeded():
                    return
                # pruning: if even choosing this will not improve best_size
                if len(selected) + 1 >= best_size:
                    break
                new_banned = banned | idx_bits_local[i] | conflict_mask_local[i]
                new_uncovered = uncovered & ~masks_local[i]
                selected.append(i)
                dfs(new_uncovered, new_banned, selected)
                selected.pop()
                if time_exceeded():
                    return

        try:
            dfs(full_mask, 0, [])
        except RecursionError:
            # fallback: return the best found so far
            pass

        # If nothing found, return trivial
        if not best_sol:
            return trivial

        # Validate best_sol coverage and conflicts conservatively
        covered_mask = 0
        for i in best_sol:
            if 0 <= i < m:
                covered_mask |= masks[i]
        if (covered_mask & full_mask) != full_mask:
            return trivial

        sel_set = set(best_sol)
        for conf in conflicts:
            valid = [int(x) for x in conf if 0 <= int(x) < m]
            if set(valid).issubset(sel_set):
                return trivial

        return best_sol