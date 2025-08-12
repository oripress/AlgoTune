import itertools
from typing import List
from collections import deque

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> List[int]:
        """
        Finds an optimal (minimum cardinality) set cover.

        The universe is the union of all elements appearing in `problem`.
        Subsets are given as lists of integers (1‑indexed elements).
        Returns a list of 1‑indexed subset indices that form a minimum cover.
        """
        if not problem:
            return []

        # Build the universe and a bit‑mask for each subset.
        universe = set()
        for subset in problem:
            universe.update(subset)
        elem_to_bit = {e: i for i, e in enumerate(sorted(universe))}
        full_mask = (1 << len(universe)) - 1

        # Original masks together with their 1‑indexed positions.
        orig_masks = []
        for idx, subset in enumerate(problem, start=1):
            mask = 0
            for e in subset:
                mask |= 1 << elem_to_bit[e]
            orig_masks.append((mask, idx))

        # Remove any subset that is a strict subset of another (they can never be optimal
        # because all subsets have equal cost).
        # Keep only masks that are not contained in any other mask.
        filtered = []
        for i, (mask_i, idx_i) in enumerate(orig_masks):
            is_subset = False
            for j, (mask_j, _) in enumerate(orig_masks):
                if i != j and (mask_i | mask_j) == mask_j:
                    # mask_i ⊆ mask_j
                    is_subset = True
                    break
            if not is_subset:
                filtered.append((mask_i, idx_i))

        # Separate masks and their original indices.
        masks = [m for m, _ in filtered]
        indices = [idx for _, idx in filtered]
        m = len(masks)

        # Try increasing cover sizes k = 1 .. m.
        for k in range(1, m + 1):
            for combo in itertools.combinations(range(m), k):
                combined = 0
                for i in combo:
                    combined |= masks[i]
                    if combined == full_mask:
                        # Early exit – full cover reached.
                        break
                if combined == full_mask:
                    # Return the original 1‑indexed subset numbers.
                    return [indices[i] for i in combo]

        # Fallback (should never happen for a well‑formed instance).
        return []