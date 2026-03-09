from __future__ import annotations

class Solver:
    def __init__(self) -> None:
        self._neighbors: list[int] = []
        self._n = 0
        self._memo: dict[int, int] = {}

    def solve(self, problem, **kwargs):
        try:
            n = len(problem)
            if n == 0:
                return []

            neighbors = [0] * n
            edge_count = 0
            for i in range(n):
                row = problem[i]
                for j in range(i + 1, n):
                    if row[j]:
                        bi = 1 << i
                        bj = 1 << j
                        neighbors[i] |= bj
                        neighbors[j] |= bi
                        edge_count += 1

            if edge_count == 0:
                return []

            self._neighbors = neighbors
            self._n = n
            self._memo = {}

            active = 0
            for i, nb in enumerate(neighbors):
                if nb:
                    active |= 1 << i

            cover = self._solve_mask(active)
            return [i for i in range(n) if (cover >> i) & 1]
        except Exception:
            return list(range(len(problem)))

    def _solve_mask(self, active: int) -> int:
        forced, active = self._reduce(active)
        if active == 0:
            return forced

        cached = self._memo.get(active)
        if cached is not None:
            return forced | cached

        first = self._first_component(active)
        if first != active:
            result = 0
            rest = active
            while rest:
                comp = self._first_component(rest)
                result |= self._solve_mask(comp)
                rest &= ~comp
            self._memo[active] = result
            return forced | result

        bip_cover = self._solve_bipartite(active)
        if bip_cover is not None:
            self._memo[active] = bip_cover
            return forced | bip_cover

        size = active.bit_count()
        if size >= 18:
            edges = self._count_edges(active)
            if 6 * edges <= size * (size - 1):
                result = active & ~self._max_independent_set(active)
                self._memo[active] = result
                return forced | result

        result = self._solve_connected_vc(active)
        self._memo[active] = result
        return forced | result

    def _reduce(self, active: int) -> tuple[int, int]:
        neighbors = self._neighbors
        forced = 0

        while True:
            changed = False

            isolates = 0
            m = active
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                m ^= v_bit
                if (neighbors[v] & active) == 0:
                    isolates |= v_bit

            if isolates:
                active &= ~isolates
                changed = True

            if changed:
                continue

            chosen = 0
            m = active
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                m ^= v_bit

                nb = neighbors[v] & active
                if nb and (nb & (nb - 1)) == 0:
                    u_bit = nb
                    u = u_bit.bit_length() - 1
                    if (neighbors[u] & active).bit_count() >= 2:
                        chosen |= u_bit

            if chosen:
                forced |= chosen
                active &= ~chosen
                changed = True

            if not changed:
                return forced, active

    def _first_component(self, active: int) -> int:
        neighbors = self._neighbors
        comp = 0
        frontier = active & -active

        while frontier:
            comp |= frontier
            nxt = 0
            f = frontier
            while f:
                v_bit = f & -f
                v = v_bit.bit_length() - 1
                f ^= v_bit
                nxt |= neighbors[v] & active
            frontier = nxt & ~comp

        return comp

    def _matching_lb(self, active: int) -> int:
        neighbors = self._neighbors
        used = 0
        count = 0
        m = active

        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            if used & v_bit:
                continue

            nb = neighbors[v] & active & ~used
            if nb:
                u_bit = nb & -nb
                used |= v_bit | u_bit
                count += 1

        return count

    def _choose_branch_vertex(self, active: int) -> int:
        neighbors = self._neighbors
        best_v = -1
        best_deg = -1
        best_score = -1

        m = active
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit

            nb = neighbors[v] & active
            deg = nb.bit_count()
            if deg < best_deg:
                continue

            score = 0
            t = nb
            while t:
                u_bit = t & -t
                u = u_bit.bit_length() - 1
                t ^= u_bit
                score += (neighbors[u] & active).bit_count()

            if deg > best_deg or score > best_score:
                best_v = v
                best_deg = deg
                best_score = score

        return best_v

    def _greedy_cover(self, active: int) -> int:
        neighbors = self._neighbors
        cover = 0
        rem = active

        while rem:
            forced, rem = self._reduce(rem)
            cover |= forced
            if rem == 0:
                break

            best_v = -1
            best_deg = 0
            m = rem
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                m ^= v_bit
                deg = (neighbors[v] & rem).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_deg == 0:
                break

            cover |= 1 << best_v
            rem &= ~(1 << best_v)

        return cover

    def _solve_connected_vc(self, active: int) -> int:
        cached = self._memo.get(active)
        if cached is not None:
            return cached

        best_mask = self._greedy_cover(active)
        best_size = best_mask.bit_count()

        if self._matching_lb(active) >= best_size:
            self._memo[active] = best_mask
            return best_mask

        neighbors = self._neighbors

        def dfs(rem: int, current_mask: int, current_size: int) -> None:
            nonlocal best_mask, best_size

            if current_size >= best_size:
                return

            forced, rem = self._reduce(rem)
            if forced:
                current_mask |= forced
                current_size += forced.bit_count()
                if current_size >= best_size:
                    return

            if rem == 0:
                best_mask = current_mask
                best_size = current_size
                return

            cached_mask = self._memo.get(rem)
            if cached_mask is not None:
                total_size = current_size + cached_mask.bit_count()
                if total_size < best_size:
                    best_mask = current_mask | cached_mask
                    best_size = total_size
                return

            first = self._first_component(rem)
            if first != rem:
                total_mask = current_mask
                total_size = current_size
                rest = rem
                while rest:
                    comp = self._first_component(rest)
                    piece = self._solve_mask(comp)
                    total_mask |= piece
                    total_size += piece.bit_count()
                    if total_size >= best_size:
                        return
                    rest &= ~comp
                best_mask = total_mask
                best_size = total_size
                return

            bip_cover = self._solve_bipartite(rem)
            if bip_cover is not None:
                total_size = current_size + bip_cover.bit_count()
                if total_size < best_size:
                    best_mask = current_mask | bip_cover
                    best_size = total_size
                self._memo[rem] = bip_cover
                return

            lb = self._matching_lb(rem)
            if current_size + lb >= best_size:
                return

            v = self._choose_branch_vertex(rem)
            v_bit = 1 << v
            nb = neighbors[v] & rem
            deg = nb.bit_count()

            if deg <= 2:
                if current_size + deg < best_size:
                    dfs(rem & ~(v_bit | nb), current_mask | nb, current_size + deg)
                if current_size + 1 < best_size:
                    dfs(rem & ~v_bit, current_mask | v_bit, current_size + 1)
            else:
                if current_size + 1 < best_size:
                    dfs(rem & ~v_bit, current_mask | v_bit, current_size + 1)
                if current_size + deg < best_size:
                    dfs(rem & ~(v_bit | nb), current_mask | nb, current_size + deg)

        dfs(active, 0, 0)
        self._memo[active] = best_mask
        return best_mask

    def _solve_bipartite(self, active: int) -> int | None:
        neighbors = self._neighbors
        n = self._n
        color = [-1] * n
        left_vertices = []

        rest = active
        while rest:
            start_bit = rest & -rest
            start = start_bit.bit_length() - 1
            queue = [start]
            qi = 0
            color[start] = 0
            left_vertices.append(start)

            while qi < len(queue):
                v = queue[qi]
                qi += 1
                t = neighbors[v] & active
                while t:
                    u_bit = t & -t
                    u = u_bit.bit_length() - 1
                    t ^= u_bit
                    cu = color[u]
                    if cu == -1:
                        color[u] = 1 - color[v]
                        if color[u] == 0:
                            left_vertices.append(u)
                        queue.append(u)
                    elif cu == color[v]:
                        return None

            for v in queue:
                rest &= ~(1 << v)

        left_list = [v for v in range(n) if ((active >> v) & 1) and color[v] == 0]
        adj_left: list[list[int]] = [[] for _ in range(n)]
        for u in left_list:
            t = neighbors[u] & active
            lst = adj_left[u]
            while t:
                v_bit = t & -t
                v = v_bit.bit_length() - 1
                t ^= v_bit
                if color[v] == 1:
                    lst.append(v)

        match_l = [-1] * n
        match_r = [-1] * n
        dist = [-1] * n

        def bfs() -> bool:
            queue = []
            qi = 0
            found = False

            for u in left_list:
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1

            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj_left[u]:
                    mu = match_r[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)

            return found

        def dfs(u: int) -> bool:
            for v in adj_left[u]:
                mu = match_r[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = -1
            return False

        while bfs():
            for u in left_list:
                if match_l[u] == -1:
                    dfs(u)

        visited_left = 0
        visited_right = 0
        queue = [u for u in left_list if match_l[u] == -1]
        qi = 0

        for u in queue:
            visited_left |= 1 << u

        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj_left[u]:
                v_bit = 1 << v
                if match_l[u] != v and not (visited_right & v_bit):
                    visited_right |= v_bit
                    mu = match_r[v]
                    if mu != -1:
                        mu_bit = 1 << mu
                        if not (visited_left & mu_bit):
                            visited_left |= mu_bit
                            queue.append(mu)

        cover = 0
        m = active
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            if color[v] == 0:
                if not (visited_left & v_bit):
                    cover |= v_bit
            else:
                if visited_right & v_bit:
                    cover |= v_bit

        return cover

    def _count_edges(self, active: int) -> int:
        neighbors = self._neighbors
        total = 0
        m = active
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            total += (neighbors[v] & active).bit_count()
        return total >> 1

    def _max_independent_set(self, active: int) -> int:
        neighbors = self._neighbors
        comp_adj = [0] * self._n
        m = active
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            comp_adj[v] = active & ~neighbors[v] & ~v_bit

        best_mask = 0
        best_size = 0

        def color_sort(cands: int):
            order = []
            bounds = []
            rest = cands
            color = 0

            while rest:
                color += 1
                q = rest
                while q:
                    v_bit = q & -q
                    v = v_bit.bit_length() - 1
                    q ^= v_bit
                    order.append(v)
                    bounds.append(color)
                    rest ^= v_bit
                    q &= ~comp_adj[v]

            return order, bounds

        def expand(cands: int, size: int, chosen: int) -> None:
            nonlocal best_mask, best_size

            if not cands:
                if size > best_size:
                    best_size = size
                    best_mask = chosen
                return

            order, bounds = color_sort(cands)
            for idx in range(len(order) - 1, -1, -1):
                if size + bounds[idx] <= best_size:
                    return

                v = order[idx]
                v_bit = 1 << v
                new_cands = cands & comp_adj[v]
                if new_cands:
                    expand(new_cands, size + 1, chosen | v_bit)
                elif size + 1 > best_size:
                    best_size = size + 1
                    best_mask = chosen | v_bit
                cands &= ~v_bit

        expand(active, 0, 0)
        return best_mask