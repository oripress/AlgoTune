from __future__ import annotations

import os
from typing import Any

from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        if not problem:
            return []

        masks: list[int] = []
        orig_indices: list[int] = []
        universe = 0

        for idx, subset in enumerate(problem, 1):
            mask = 0
            for element in subset:
                mask |= 1 << (element - 1)
            if mask:
                masks.append(mask)
                orig_indices.append(idx)
                universe |= mask

        if universe == 0:
            return []

        masks, orig_indices, uncovered, forced = self._preprocess(
            masks, orig_indices, universe
        )

        if uncovered == 0:
            forced.sort()
            return forced

        for mask, orig in zip(masks, orig_indices):
            if mask == uncovered:
                result = forced + [orig]
                result.sort()
                return result

        pair = self._find_pair_cover(masks, orig_indices, uncovered)
        if pair is not None:
            result = forced + pair
            result.sort()
            return result

        masks, uncovered = self._compress_masks(masks, uncovered)

        if uncovered == 0:
            forced.sort()
            return forced

        extra = self._solve_exact(masks, orig_indices, uncovered)
        result = forced + extra
        result.sort()
        return result

    def _compress_masks(
        self, masks: list[int], uncovered: int
    ) -> tuple[list[int], int]:
        if uncovered & (uncovered + 1) == 0:
            return masks, uncovered

        bit_to_pos: dict[int, int] = {}
        bits = uncovered
        pos = 0
        while bits:
            bit = bits & -bits
            bits ^= bit
            bit_to_pos[bit] = pos
            pos += 1

        new_masks: list[int] = []
        for mask in masks:
            new_mask = 0
            bits = mask
            while bits:
                bit = bits & -bits
                bits ^= bit
                new_mask |= 1 << bit_to_pos[bit]
            if new_mask:
                new_masks.append(new_mask)

        return new_masks, (1 << pos) - 1

    def _find_pair_cover(
        self, masks: list[int], orig_indices: list[int], uncovered: int
    ) -> list[int] | None:
        m = len(masks)
        if m > 512:
            return None
        for i in range(m):
            mi = masks[i]
            for j in range(i + 1, m):
                if (mi | masks[j]) == uncovered:
                    return [orig_indices[i], orig_indices[j]]
        return None

    def _preprocess(
        self, masks: list[int], orig_indices: list[int], uncovered: int
    ) -> tuple[list[int], list[int], int, list[int]]:
        forced: list[int] = []

        while True:
            reduced: dict[int, int] = {}
            for mask, orig in zip(masks, orig_indices):
                mask &= uncovered
                if not mask:
                    continue
                prev = reduced.get(mask)
                if prev is None or orig < prev:
                    reduced[mask] = orig

            if not reduced:
                return [], [], uncovered, forced

            items = sorted(reduced.items(), key=lambda x: (-x[0].bit_count(), x[1]))

            if len(items) <= 256:
                kept_masks: list[int] = []
                kept_orig: list[int] = []
                for mask, orig in items:
                    dominated = False
                    for kept in kept_masks:
                        if mask & ~kept == 0:
                            dominated = True
                            break
                    if not dominated:
                        kept_masks.append(mask)
                        kept_orig.append(orig)
                masks = kept_masks
                orig_indices = kept_orig
            else:
                masks = [mask for mask, _ in items]
                orig_indices = [orig for _, orig in items]

            if not masks:
                return [], [], uncovered, forced

            if len(masks) <= 256 and uncovered.bit_count() <= 160:
                support: dict[int, int] = {}
                for pos, mask in enumerate(masks):
                    mark = 1 << pos
                    bits = mask
                    while bits:
                        bit = bits & -bits
                        bits ^= bit
                        support[bit] = support.get(bit, 0) | mark

                items_support = sorted(
                    support.items(), key=lambda x: (x[1].bit_count(), x[0])
                )
                removable = 0
                for i, (_, sup_i) in enumerate(items_support):
                    for bit_j, sup_j in items_support[i + 1 :]:
                        if sup_i & ~sup_j == 0:
                            removable |= bit_j

                if removable:
                    uncovered &= ~removable
                    if uncovered == 0:
                        return [], [], 0, forced
                    continue

            owner: dict[int, int] = {}
            forced_pos: set[int] = set()
            for pos, mask in enumerate(masks):
                bits = mask
                while bits:
                    bit = bits & -bits
                    bits ^= bit
                    prev = owner.get(bit)
                    if prev is None:
                        owner[bit] = pos
                    elif prev >= 0 and prev != pos:
                        owner[bit] = -1

            for pos in owner.values():
                if pos >= 0:
                    forced_pos.add(pos)

            if not forced_pos:
                return masks, orig_indices, uncovered, forced

            cover = 0
            for pos in forced_pos:
                forced.append(orig_indices[pos])
                cover |= masks[pos]

            uncovered &= ~cover
            if uncovered == 0:
                return [], [], 0, forced

    def _solve_exact(
        self, masks: list[int], orig_indices: list[int], uncovered: int
    ) -> list[int]:
        m = len(masks)
        n = uncovered.bit_count()

        set_sizes = [mask.bit_count() for mask in masks]

        greedy_choice = self._greedy_cover(masks, uncovered, set_sizes)
        cover = 0
        for idx in greedy_choice:
            cover |= masks[idx]
        if cover != uncovered:
            return self._solve_cpsat(masks, orig_indices, uncovered)

        greedy_choice = self._trim_cover(greedy_choice, masks, uncovered)
        upper = len(greedy_choice)

        if m > 140 or (m > 90 and n > 90 and upper > 8):
            return self._solve_cpsat(masks, orig_indices, uncovered)

        coverers: list[list[int]] = [[] for _ in range(n)]
        ordered_coverers: list[tuple[int, ...]] = [tuple() for _ in range(n)]
        cover_count = [0] * n
        conflict = [0] * n

        for i, mask in enumerate(masks):
            bits = mask
            while bits:
                bit = bits & -bits
                bits ^= bit
                coverers[bit.bit_length() - 1].append(i)

        for e in range(n):
            cs = coverers[e]
            ordered_coverers[e] = tuple(
                sorted(cs, key=lambda i: (set_sizes[i], -i), reverse=True)
            )
            cover_count[e] = len(cs)
            conf = 0
            for i in cs:
                conf |= masks[i]
            conflict[e] = conf

        elem_order = sorted(
            range(n), key=lambda e: (cover_count[e], conflict[e].bit_count(), e)
        )
        max_set_size = max(set_sizes)
        lb_cache: dict[int, int] = {}

        def lower_bound(state: int) -> int:
            cached = lb_cache.get(state)
            if cached is not None:
                return cached

            remaining = state.bit_count()
            if remaining <= max_set_size:
                lb_cache[state] = 1
                return 1

            lb1 = (remaining + max_set_size - 1) // max_set_size

            rem = state
            lb2 = 0
            for e in elem_order:
                bit = 1 << e
                if rem & bit:
                    lb2 += 1
                    rem &= ~conflict[e]
                    if not rem:
                        break

            bound = lb1 if lb1 >= lb2 else lb2
            lb_cache[state] = bound
            return bound

        lower = lower_bound(uncovered)
        if lower >= upper:
            return [orig_indices[i] for i in greedy_choice]

        node_limit = 250000 + 150000 * upper
        need: dict[int, int] = {}
        total_nodes = 0

        class SearchFallback(Exception):
            pass

        def choose_element(state: int) -> int:
            best_e = -1
            best_c = m + 1
            bits = state
            while bits:
                bit = bits & -bits
                bits ^= bit
                e = bit.bit_length() - 1
                c = cover_count[e]
                if c < best_c:
                    best_c = c
                    best_e = e
                    if c <= 2:
                        break
            return best_e

        def search(state: int, budget: int, depth: int, path: list[int]) -> bool:
            nonlocal total_nodes
            total_nodes += 1
            if total_nodes > node_limit:
                raise SearchFallback

            if state == 0:
                return True
            if budget == 0:
                return False

            if lower_bound(state) > budget:
                return False

            prev = need.get(state)
            if prev is not None and prev > budget:
                return False

            e = choose_element(state)
            options = ordered_coverers[e]

            for i in options:
                child = state & ~masks[i]
                if child == state:
                    continue
                if budget > 1 and lower_bound(child) > budget - 1:
                    continue
                path[depth] = i
                if search(child, budget - 1, depth + 1, path):
                    return True

            required = budget + 1
            old = need.get(state, 0)
            if required > old:
                need[state] = required
            return False

        for target in range(lower, upper):
            path = [0] * target
            try:
                if search(uncovered, target, 0, path):
                    return [orig_indices[i] for i in path]
            except SearchFallback:
                return self._solve_cpsat(masks, orig_indices, uncovered)

        return [orig_indices[i] for i in greedy_choice]

    def _greedy_cover(
        self, masks: list[int], uncovered: int, set_sizes: list[int] | None = None
    ) -> list[int]:
        if set_sizes is None:
            set_sizes = [mask.bit_count() for mask in masks]

        state = uncovered
        chosen: list[int] = []

        while state:
            best_i = -1
            best_gain = 0
            best_size = -1
            for i, mask in enumerate(masks):
                gain = (mask & state).bit_count()
                size = set_sizes[i]
                if gain > best_gain or (gain == best_gain and size > best_size):
                    best_i = i
                    best_gain = gain
                    best_size = size
            if best_i < 0 or best_gain == 0:
                break
            chosen.append(best_i)
            state &= ~masks[best_i]

        return chosen

    def _trim_cover(
        self, chosen: list[int], masks: list[int], universe: int
    ) -> list[int]:
        chosen = list(chosen)
        changed = True
        while changed:
            changed = False
            for pos in range(len(chosen)):
                cover = 0
                for j, idx in enumerate(chosen):
                    if j != pos:
                        cover |= masks[idx]
                if cover == universe:
                    chosen.pop(pos)
                    changed = True
                    break
        return chosen

    def _solve_cpsat(
        self, masks: list[int], orig_indices: list[int], uncovered: int
    ) -> list[int]:
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(len(masks))]

        greedy_choice = self._greedy_cover(masks, uncovered)
        if greedy_choice:
            greedy_set = set(greedy_choice)
            model.Add(sum(x) <= len(greedy_choice))
            for i in greedy_choice:
                model.AddHint(x[i], 1)
            for i in range(len(masks)):
                if i not in greedy_set:
                    model.AddHint(x[i], 0)

        bits = uncovered
        while bits:
            bit = bits & -bits
            bits ^= bit
            model.Add(sum(x[i] for i, mask in enumerate(masks) if mask & bit) >= 1)

        model.Minimize(sum(x))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = min(8, os.cpu_count() or 1)
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 0

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        return [orig for i, orig in enumerate(orig_indices) if solver.BooleanValue(x[i])]