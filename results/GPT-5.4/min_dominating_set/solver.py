from __future__ import annotations

from typing import Any

class Solver:
    def __init__(self) -> None:
        pass

    def solve(self, problem, **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []

        adj = [0] * n
        closed = [0] * n
        for i, row in enumerate(problem):
            mask = 0
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            adj[i] = mask
            closed[i] = mask | (1 << i)
        def solve_tree_component(verts: list[int], comp_mask: int) -> list[int]:
            m = len(verts)
            pos = {v: i for i, v in enumerate(verts)}
            parent = [-1] * m
            children = [[] for _ in range(m)]
            order = [0]

            qi = 0
            while qi < len(order):
                u_idx = order[qi]
                qi += 1
                nb = adj[verts[u_idx]] & comp_mask
                while nb:
                    b = nb & -nb
                    nb ^= b
                    v = b.bit_length() - 1
                    v_idx = pos[v]
                    if v_idx == parent[u_idx] or v_idx == 0:
                        continue
                    if parent[v_idx] != -1:
                        continue
                    parent[v_idx] = u_idx
                    children[u_idx].append(v_idx)
                    order.append(v_idx)

            inf = m + 1
            a = [0] * m  # selected
            b = [0] * m  # not selected, dominated by a child
            c = [0] * m  # not selected, expects domination from parent

            for u in reversed(order):
                ch = children[u]
                if not ch:
                    a[u] = 1
                    b[u] = inf
                    c[u] = 0
                    continue

                sa = 1
                sb = 0
                sc = 0
                has_a = False
                extra = inf

                for v in ch:
                    av = a[v]
                    bv = b[v]
                    cv = c[v]

                    best_abc = av
                    if bv < best_abc:
                        best_abc = bv
                    if cv < best_abc:
                        best_abc = cv
                    sa += best_abc

                    best_ab = av if av <= bv else bv
                    sb += best_ab
                    sc += best_ab
                    if av <= bv:
                        has_a = True
                    else:
                        diff = av - bv
                        if diff < extra:
                            extra = diff

                a[u] = sa
                b[u] = sb if has_a else sb + extra
                c[u] = sc

            root_state = 0 if a[0] <= b[0] else 1
            chosen: list[int] = []

            def reconstruct(u: int, state: int) -> None:
                if state == 0:
                    chosen.append(verts[u])
                    for v in children[u]:
                        av = a[v]
                        bv = b[v]
                        cv = c[v]
                        best_state = 0
                        best_val = av
                        if bv < best_val:
                            best_val = bv
                            best_state = 1
                        if cv < best_val:
                            best_state = 2
                        reconstruct(v, best_state)
                    return

                pick_a = [False] * len(children[u])
                has_a_local = False
                best_switch = -1
                best_diff = inf

                for i, v in enumerate(children[u]):
                    av = a[v]
                    bv = b[v]
                    if av <= bv:
                        pick_a[i] = True
                        has_a_local = True
                    else:
                        diff = av - bv
                        if diff < best_diff:
                            best_diff = diff
                            best_switch = i

                if state == 1 and not has_a_local and best_switch >= 0:
                    pick_a[best_switch] = True

                for i, v in enumerate(children[u]):
                    reconstruct(v, 0 if pick_a[i] else 1)

            reconstruct(0, root_state)
            chosen.sort()
            return chosen

        result: list[int] = []
        unseen = (1 << n) - 1

        while unseen:
            seed_bit = unseen & -unseen
            seed = seed_bit.bit_length() - 1

            comp = 0
            stack = seed_bit
            unseen ^= seed_bit
            while stack:
                b = stack & -stack
                stack ^= b
                v = b.bit_length() - 1
                comp |= b
                nxt = adj[v] & unseen
                stack |= nxt
                unseen &= ~nxt

            if comp.bit_count() == 1:
                result.append(seed)
                continue

            verts: list[int] = []
            mm = comp
            while mm:
                b = mm & -mm
                verts.append(b.bit_length() - 1)
                mm ^= b

            edge_count = 0
            for v in verts:
                edge_count += (adj[v] & comp).bit_count()
            if edge_count == 2 * (len(verts) - 1):
                result.extend(solve_tree_component(verts, comp))
                continue

            idx = {v: i for i, v in enumerate(verts)}
            m = len(verts)
            full = (1 << m) - 1

            local_closed = [0] * m
            for i, v in enumerate(verts):
                cm = 0
                nb = closed[v] & comp
                while nb:
                    b = nb & -nb
                    cm |= 1 << idx[b.bit_length() - 1]
                    nb ^= b
                local_closed[i] = cm

            rep_of_mask: dict[int, int] = {}
            rep_to_orig: list[int] = []
            reduced_masks: list[int] = []
            local_to_rep = [0] * m

            for i, cm in enumerate(local_closed):
                rep = rep_of_mask.get(cm)
                if rep is None:
                    rep = len(reduced_masks)
                    rep_of_mask[cm] = rep
                    reduced_masks.append(cm)
                    rep_to_orig.append(verts[i])
                local_to_rep[i] = rep

            if len(reduced_masks) != m:
                remapped = [0] * len(reduced_masks)
                for i, cm in enumerate(reduced_masks):
                    nm = 0
                    mm2 = cm
                    while mm2:
                        b = mm2 & -mm2
                        nm |= 1 << local_to_rep[b.bit_length() - 1]
                        mm2 ^= b
                    remapped[i] = nm
                local_closed = remapped
                m = len(local_closed)
                full = (1 << m) - 1
            else:
                rep_to_orig = verts[:]

            done = False
            for i, cm in enumerate(local_closed):
                if cm == full:
                    result.append(rep_to_orig[i])
                    done = True
                    break
            if done:
                continue

            candidate_mask = full
            for v in range(m):
                Sv = local_closed[v]
                for w in range(m):
                    if v != w and (Sv & ~local_closed[w]) == 0:
                        candidate_mask &= ~(1 << v)
                        break
            if candidate_mask == 0:
                candidate_mask = full

            coverers = [0] * m
            for u in range(m):
                coverers[u] = local_closed[u] & candidate_mask

            if any(c == 0 for c in coverers):
                candidate_mask = full
                for u in range(m):
                    coverers[u] = local_closed[u]

            coverer_sizes = [c.bit_count() for c in coverers]
            singleton_coverer = [c.bit_length() - 1 if c and (c & (c - 1)) == 0 else -1 for c in coverers]
            coverer_order = list(range(m))
            coverer_order.sort(key=coverer_sizes.__getitem__)

            cand_list: list[int] = []
            cm = candidate_mask
            while cm:
                b = cm & -cm
                cand_list.append(b.bit_length() - 1)
                cm ^= b

            def greedy_solution() -> list[int]:
                uncovered = full
                sol: list[int] = []
                while uncovered:
                    best_v = cand_list[0]
                    best_gain = -1
                    for v in cand_list:
                        gain = (local_closed[v] & uncovered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    sol.append(best_v)
                    uncovered &= ~local_closed[best_v]
                return sol

            best = greedy_solution()
            best_len = len(best)

            if best_len > 2 and len(cand_list) <= 200:
                pair_found = False
                for i, v in enumerate(cand_list):
                    miss = full & ~local_closed[v]
                    for w in cand_list[i + 1 :]:
                        if (miss & ~local_closed[w]) == 0:
                            best = [v, w]
                            best_len = 2
                            pair_found = True
                            break
                    if pair_found:
                        break

            if best_len <= 2:
                for v in best:
                    result.append(rep_to_orig[v])
                continue

            seen: dict[int, int] = {}
            lb_cache: dict[int, int] = {}

            def lower_bound(uncovered: int) -> int:
                cached = lb_cache.get(uncovered)
                if cached is not None:
                    return cached

                rem = uncovered.bit_count()
                max_gain = 1
                for v in cand_list:
                    gain = (local_closed[v] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                lb1 = (rem + max_gain - 1) // max_gain

                packed = 0
                used = 0
                for u in coverer_order:
                    if not ((uncovered >> u) & 1):
                        continue
                    cs = coverers[u]
                    if cs & used:
                        continue
                    used |= cs
                    packed += 1

                val = lb1 if lb1 > packed else packed
                lb_cache[uncovered] = val
                return val

            def search(uncovered: int, chosen: list[int]) -> None:
                nonlocal best, best_len

                chosen_len = len(chosen)
                if chosen_len >= best_len:
                    return
                if uncovered == 0:
                    best = chosen.copy()
                    best_len = chosen_len
                    return

                prev = seen.get(uncovered)
                if prev is not None and prev <= chosen_len:
                    return
                seen[uncovered] = chosen_len

                if chosen_len + lower_bound(uncovered) >= best_len:
                    return

                base_len = chosen_len

                while True:
                    forced_v = -1
                    uu = uncovered
                    while uu:
                        b = uu & -uu
                        u = b.bit_length() - 1
                        fv = singleton_coverer[u]
                        if fv >= 0:
                            forced_v = fv
                            break
                        uu ^= b

                    if forced_v < 0:
                        break

                    chosen.append(forced_v)
                    uncovered &= ~local_closed[forced_v]

                    if len(chosen) >= best_len:
                        del chosen[base_len:]
                        return
                    if uncovered == 0:
                        best = chosen.copy()
                        best_len = len(chosen)
                        del chosen[base_len:]
                        return

                    prev = seen.get(uncovered)
                    if prev is not None and prev <= len(chosen):
                        del chosen[base_len:]
                        return
                    seen[uncovered] = len(chosen)

                    if len(chosen) + lower_bound(uncovered) >= best_len:
                        del chosen[base_len:]
                        return

                target_opts = 0
                target_key = None
                uu = uncovered
                while uu:
                    b = uu & -uu
                    u = b.bit_length() - 1
                    opts = coverers[u]
                    key = (coverer_sizes[u], -(local_closed[u] & uncovered).bit_count(), u)
                    if target_key is None or key < target_key:
                        target_key = key
                        target_opts = opts
                        if key[0] <= 2:
                            break
                    uu ^= b

                raw_options: list[tuple[int, int]] = []
                oo = target_opts
                while oo:
                    b = oo & -oo
                    v = b.bit_length() - 1
                    raw_options.append((local_closed[v] & uncovered, v))
                    oo ^= b

                options: list[tuple[int, int]] = []
                for cov, v in raw_options:
                    dominated = False
                    for cov2, v2 in raw_options:
                        if v == v2:
                            continue
                        if (cov & ~cov2) == 0 and (cov != cov2 or v > v2):
                            dominated = True
                            break
                    if not dominated:
                        options.append((cov, v))

                options.sort(key=lambda cv: (cv[0].bit_count(), cv[0]), reverse=True)

                for cov, v in options:
                    chosen.append(v)
                    search(uncovered & ~cov, chosen)
                    chosen.pop()

                del chosen[base_len:]

            search(full, [])
            for v in best:
                result.append(rep_to_orig[v])

        result.sort()
        return result