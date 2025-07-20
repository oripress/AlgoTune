class Solver:
    def solve(self, problem, **kwargs):
        m = len(problem)
        if m == 0:
            return []
        # Build universe and bit mapping
        u_set = set()
        for s in problem:
            u_set |= set(s)
        if not u_set:
            return []
        elems = list(u_set)
        pos = {e: i for i, e in enumerate(elems)}
        nbits = len(elems)
        full_mask = (1 << nbits) - 1
        # Build masks list, remove empty
        raw = []
        for idx, s in enumerate(problem):
            mask = 0
            for e in s:
                mask |= 1 << pos[e]
            if mask:
                raw.append((idx, mask))
        # Remove duplicate masks, keep smallest index
        mask2idx = {}
        for idx, mask in raw:
            if mask not in mask2idx or idx < mask2idx[mask]:
                mask2idx[mask] = idx
        masks_list = [(idx, mask) for mask, idx in mask2idx.items()]
        # Sort masks by descending coverage for heuristic
        masks_list.sort(key=lambda x: x[1].bit_count(), reverse=True)
        idx_arr = [idx for idx, mask in masks_list]
        mask_arr = [mask for idx, mask in masks_list]
        # Precompute static max coverage bound
        static_max_cov = max(mask.bit_count() for mask in mask_arr)
        # Build bit to sets mapping
        bit_to_sets = [[] for _ in range(nbits)]
        for j, mask in enumerate(mask_arr):
            rem = mask
            while rem:
                lsb = rem & -rem
                b = lsb.bit_length() - 1
                bit_to_sets[b].append(j)
                rem ^= lsb
        # Mandatory sets: cover unique elements
        mandatory = set()
        for b, js in enumerate(bit_to_sets):
            if len(js) == 1:
                mandatory.add(js[0])
        # Initial cover bits from mandatory
        init_cover = 0
        for j in mandatory:
            init_cover |= mask_arr[j]
        rem_bits0 = full_mask & ~init_cover
        # Greedy upper bound for remaining bits
        greedy_sel = []
        rem = rem_bits0
        while rem:
            best_j = -1
            best_cov = 0
            for j, mask in enumerate(mask_arr):
                cov = (mask & rem).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_j = j
            if best_j < 0:
                break
            greedy_sel.append(best_j)
            rem &= ~mask_arr[best_j]
        best_size = len(mandatory) + len(greedy_sel)
        best_sel_rem = greedy_sel.copy()
        # Depth-first search with branch and bound
        def dfs(rem_bits, sel_rem):
            nonlocal best_size, best_sel_rem
            if rem_bits == 0:
                size = len(sel_rem) + len(mandatory)
                if size < best_size:
                    best_size = size
                    best_sel_rem = sel_rem.copy()
                return
            # Lower bound prune
            bits_needed = rem_bits.bit_count()
            lb = (bits_needed + static_max_cov - 1) // static_max_cov
            if len(sel_rem) + len(mandatory) + lb >= best_size:
                return
            # Choose an uncovered bit
            lsb = rem_bits & -rem_bits
            b = lsb.bit_length() - 1
            # Branch on sets covering this bit
            for j in bit_to_sets[b]:
                new_bits = rem_bits & ~mask_arr[j]
                sel_rem.append(j)
                dfs(new_bits, sel_rem)
                sel_rem.pop()
        dfs(rem_bits0, [])
        # Combine mandatory and best remaining
        sol_set = set(mandatory) | set(best_sel_rem)
        # Map to original indices, convert to 1-based
        ans = [idx_arr[j] + 1 for j in sol_set]
        ans.sort()
        return ans