from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

class _SetData:
    __slots__ = ("mask", "orig_index_1", "elems")

    def __init__(self, mask: int, orig_index_1: int, elems: Tuple[int, ...]) -> None:
        self.mask = mask
        self.orig_index_1 = orig_index_1  # 1-indexed for output
        self.elems = elems  # mapped element positions (0..n-1)

class Solver:
    def __init__(self) -> None:
        # No heavy init; compilation time doesn't count anyway.
        pass

    @staticmethod
    def _popcount(x: int) -> int:
        return x.bit_count()
    def solve(self, problem: List[List[int]], **kwargs: Any) -> Any:
        # ---- Build universe and remap elements to 0..n-1 for dense bitmasks ----
        if not problem:
            return []

        universe_set = set()
        for s in problem:
            universe_set.update(s)
        if not universe_set:
            return []

        universe = sorted(universe_set)
        n = len(universe)
        pos: Dict[int, int] = {e: i for i, e in enumerate(universe)}

        # ---- Build masks; de-duplicate identical masks (keep smallest orig index) ----
        dedup: Dict[int, _SetData] = {}
        for i0, subset in enumerate(problem):
            if not subset:
                continue
            # unique mapped elements
            mapped = {pos[e] for e in subset}
            if not mapped:
                continue
            mask = 0
            for p in mapped:
                mask |= 1 << p
            orig1 = i0 + 1
            prev = dedup.get(mask)
            if prev is None or orig1 < prev.orig_index_1:
                dedup[mask] = _SetData(mask=mask, orig_index_1=orig1, elems=tuple(mapped))

        sets: List[_SetData] = list(dedup.values())
        if not sets:
            return []

        # Universe mask
        universe_mask = 0
        for sd in sets:
            universe_mask |= sd.mask

        # ---- Remove dominated sets (subset of another set) when not too large ----
        # O(m^2), so guard it.
        if len(sets) <= 800:
            sets.sort(key=lambda sd: sd.mask.bit_count(), reverse=True)
            kept: List[_SetData] = []
            kept_masks: List[int] = []
            for sd in sets:
                m = sd.mask
                dominated = False
                for km in kept_masks:
                    if m & ~km == 0:
                        dominated = True
                        break
                if dominated:
                    continue
                # remove any kept that is subset of m
                j = 0
                while j < len(kept_masks):
                    if kept_masks[j] & ~m == 0:
                        kept_masks.pop(j)
                        kept.pop(j)
                    else:
                        j += 1
                kept.append(sd)
                kept_masks.append(m)
            sets = kept

        m = len(sets)
        if m == 0:
            return []

        # ---- Build element -> covering sets lists ----
        elem_to_sets: List[List[int]] = [[] for _ in range(n)]
        masks: List[int] = [0] * m
        orig1: List[int] = [0] * m
        elems_list: List[Tuple[int, ...]] = [tuple()] * m
        sizes: List[int] = [0] * m

        for si, sd in enumerate(sets):
            masks[si] = sd.mask
            orig1[si] = sd.orig_index_1
            elems_list[si] = sd.elems
            sz = sd.mask.bit_count()
            sizes[si] = sz
            for epos in sd.elems:
                elem_to_sets[epos].append(si)

        # Sort per-element options by static set size descending (good branching order)
        key_sz = sizes.__getitem__
        for e in range(n):
            lst = elem_to_sets[e]
            if len(lst) > 1:
                lst.sort(key=key_sz, reverse=True)
            elem_to_sets[e] = tuple(lst)  # type: ignore[assignment]
        # ---- Forced sets: any element covered by exactly one set => that set is mandatory ----
        forced_set_ids = set()
        for e in range(n):
            lst = elem_to_sets[e]
            if len(lst) == 1:
                forced_set_ids.add(lst[0])

        forced_mask = 0
        forced_out: List[int] = []
        if forced_set_ids:
            for si in forced_set_ids:
                forced_mask |= masks[si]
            forced_out = [orig1[si] for si in forced_set_ids]

        uncovered0 = universe_mask & ~forced_mask
        if uncovered0 == 0:
            forced_out.sort()
            return forced_out

        # ---- Remove sets that don't intersect the remaining uncovered elements ----
        keep_map: Dict[int, int] = {}
        new_masks: List[int] = []
        new_orig1: List[int] = []
        new_elems: List[Tuple[int, ...]] = []
        new_sizes: List[int] = []

        for si in range(m):
            if masks[si] & uncovered0:
                keep_map[si] = len(new_masks)
                new_masks.append(masks[si])
                new_orig1.append(orig1[si])
                new_elems.append(elems_list[si])
                new_sizes.append(sizes[si])

        masks = new_masks
        orig1 = new_orig1
        elems_list = new_elems
        sizes = new_sizes
        m = len(masks)

        # Rebuild elem_to_sets for remaining sets
        elem_to_sets = [[] for _ in range(n)]
        for si in range(m):
            for epos in elems_list[si]:
                # Only matters if element may still be uncovered; still fine to keep all.
                elem_to_sets[epos].append(si)
        for e in range(n):
            lst = elem_to_sets[e]
            if len(lst) > 1:
                lst.sort(key=lambda si: sizes[si], reverse=True)

        # Element order for pivot selection: least frequent first
        elem_order = list(range(n))
        elem_order.sort(key=lambda e: len(elem_to_sets[e]))

        max_cover = max((sz for sz in sizes if sz > 0), default=1)

        # ---- Greedy upper bound (fast initial cap for iterative deepening) ----
        def greedy_cover(uncovered: int) -> List[int]:
            chosen: List[int] = []
            u = uncovered
            # Simple O(m*k) greedy; usually small k.
            while u:
                best_si = -1
                best_gain = 0
                for si in range(m):
                    gain = (masks[si] & u).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_si = si
                        if best_gain == u.bit_count():
                            break
                if best_si < 0 or best_gain == 0:
                    # Should not happen in well-formed instances.
                    break
                chosen.append(best_si)
                u &= ~masks[best_si]
            return chosen

        greedy_extra = greedy_cover(uncovered0)

        # Validate greedy actually covers (use bitwise OR, not integer sum)
        greedy_cov = 0
        for si in greedy_extra:
            greedy_cov |= masks[si]

        # If greedy somehow failed (shouldn't in well-formed instances), fall back to greedy anyway.
        if uncovered0 & ~greedy_cov:
            out = forced_out + [orig1[si] for si in greedy_extra]
            out.sort()
            return out

        ub_extra = len(greedy_extra)

        # ---- Depth-limited DFS (exact) ----
        fail: Dict[Tuple[int, int], bool] = {}

        def dfs(uncovered: int, depth: int) -> Optional[Tuple[int, ...]]:
            if uncovered == 0:
                return ()
            if depth <= 0:
                return None

            # Simple lower-bound prune
            need_lb = (uncovered.bit_count() + max_cover - 1) // max_cover
            if need_lb > depth:
                return None

            key = (uncovered, depth)
            if key in fail:
                return None

            # pick pivot element (uncovered) with smallest static frequency
            pivot = -1
            for e in elem_order:
                if (uncovered >> e) & 1:
                    pivot = e
                    break
            if pivot < 0:
                return ()

            options = elem_to_sets[pivot]
            # options must exist for well-formed instances
            for si in options:
                new_uncovered = uncovered & ~masks[si]
                res = dfs(new_uncovered, depth - 1)
                if res is not None:
                    return (si,) + res

            fail[key] = True
            return None

        # Iterative deepening to guarantee optimality
        lb = (uncovered0.bit_count() + max_cover - 1) // max_cover
        best_extra: Optional[Tuple[int, ...]] = None
        for depth in range(lb, ub_extra + 1):
            fail.clear()
            r = dfs(uncovered0, depth)
            if r is not None:
                best_extra = r
                break

        if best_extra is None:
            # Fallback to greedy (shouldn't happen if union covers universe)
            out = forced_out + [orig1[si] for si in greedy_extra]
            out.sort()
            return out

        out = forced_out + [orig1[si] for si in best_extra]
        out.sort()
        return out