from __future__ import annotations

from typing import Any
import sys

sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Build bitset adjacency for the original graph, using symmetry.
        orig_adj = [0] * n
        for i in range(n):
            row = problem[i]
            bi = 1 << i
            ai = orig_adj[i]
            for j in range(i + 1, n):
                if row[j]:
                    bj = 1 << j
                    ai |= bj
                    orig_adj[j] |= bi
            orig_adj[i] = ai
        local_index = [-1] * n

        def solve_component(component_mask: int) -> list[int]:
            size = component_mask.bit_count()
            if size <= 1:
                if size == 0:
                    return []
                return [component_mask.bit_length() - 1]

            # Extract component vertices and original degrees inside the component.
            vertices = []
            orig_degrees = []
            degree_sum = 0
            cm = component_mask
            while cm:
                vb = cm & -cm
                cm ^= vb
                v = vb.bit_length() - 1
                deg = (orig_adj[v] & component_mask).bit_count()
                vertices.append(v)
                orig_degrees.append(deg)
                degree_sum += deg

            if degree_sum == size * (size - 1):
                return [min(vertices)]

            # Order by descending degree in the complement graph.
            indexed = list(range(size))
            indexed.sort(key=lambda i: (size - 1 - orig_degrees[i], -vertices[i]), reverse=True)
            ordered_vertices = [vertices[i] for i in indexed]

            k = len(ordered_vertices)
            if k == 2:
                u, v = ordered_vertices
                if (orig_adj[u] >> v) & 1:
                    return [min(u, v)]
                return sorted([u, v])

            full_mask = (1 << k) - 1
            # Build complement adjacency on the induced subgraph from original bitset neighbors.
            li = local_index
            for i, v in enumerate(ordered_vertices):
                li[v] = i

            comp_adj = [0] * k
            has_edge = 0
            adj_orig = orig_adj
            for i, gi in enumerate(ordered_vertices):
                local_neighbors = 0
                nbrs = adj_orig[gi] & component_mask
                while nbrs:
                    vb = nbrs & -nbrs
                    nbrs ^= vb
                    local_neighbors |= 1 << li[vb.bit_length() - 1]
                mask = full_mask ^ local_neighbors ^ (1 << i)
                comp_adj[i] = mask
                has_edge |= mask

            if not has_edge:
                return [min(ordered_vertices)]

            adj = comp_adj

            def color_sort(candidates: int) -> tuple[list[int], list[int]]:
                order: list[int] = []
                colors: list[int] = []
                append_order = order.append
                append_color = colors.append
                rem = candidates
                color = 0
                adj_local = adj

                while rem:
                    color += 1
                    q = rem
                    while q:
                        vb = q & -q
                        v = vb.bit_length() - 1
                        append_order(v)
                        append_color(color)
                        rem ^= vb
                        q ^= vb
                        q &= ~adj_local[v]
                return order, colors

            # Cheap greedy clique for an initial lower bound.
            best_mask = 0
            best_size = 0
            greedy_candidates = full_mask
            greedy_mask = 0
            greedy_size = 0
            while greedy_candidates:
                vb = greedy_candidates & -greedy_candidates
                v = vb.bit_length() - 1
                greedy_mask |= vb
                greedy_size += 1
                greedy_candidates &= adj[v]
            best_mask = greedy_mask
            best_size = greedy_size

            def expand(candidates: int, cur_size: int, cur_mask: int) -> None:
                nonlocal best_mask, best_size

                if not candidates:
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_mask
                    return

                if cur_size + candidates.bit_count() <= best_size:
                    return

                order, colors = color_sort(candidates)
                i = len(order) - 1

                while i >= 0:
                    if cur_size + colors[i] <= best_size:
                        return

                    v = order[i]
                    vb = 1 << v
                    new_size = cur_size + 1
                    new_mask = cur_mask | vb
                    new_candidates = candidates & adj[v]

                    if not new_candidates:
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = new_mask
                    elif new_size + new_candidates.bit_count() > best_size:
                        expand(new_candidates, new_size, new_mask)

                    candidates ^= vb
                    i -= 1

            expand(full_mask, 0, 0)

            selected = []
            bm = best_mask
            while bm:
                vb = bm & -bm
                bm ^= vb
                selected.append(ordered_vertices[vb.bit_length() - 1])
            selected.sort()
            return selected

        # MIS is additive over connected components in the original graph.
        result: list[int] = []
        unseen = (1 << n) - 1

        while unseen:
            seed = unseen & -unseen
            unseen ^= seed
            component_mask = seed
            frontier = seed

            while frontier:
                vb = frontier & -frontier
                frontier ^= vb
                v = vb.bit_length() - 1
                nxt = orig_adj[v] & unseen
                if nxt:
                    unseen ^= nxt
                    frontier |= nxt
                    component_mask |= nxt

            result.extend(solve_component(component_mask))

        result.sort()
        return result