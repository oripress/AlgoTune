from typing import Any, List

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Maximum Independent Set solver.

        Approach:
        - Convert adjacency matrix to bitset neighbor masks.
        - Decompose graph into connected components and solve each component separately.
        - For each component, find a maximum clique in the complement graph using a
          bitset branch-and-bound (Tomita-style) with greedy coloring for pruning.
        - Map local results back to original vertex indices and combine.
        """
        if problem is None:
            return []
        n = len(problem)
        if n == 0:
            return []

        # Build neighbor bitmasks for the original graph
        neighbors: List[int] = [0] * n
        for i, row in enumerate(problem):
            mask = 0
            lim = min(len(row), n)
            for j in range(lim):
                if row[j]:
                    mask |= 1 << j
            neighbors[i] = mask

        # Find connected components (on the original graph)
        visited = 0
        components: List[List[int]] = []
        for i in range(n):
            if (visited >> i) & 1:
                continue
            stack = [i]
            comp_mask = 0
            while stack:
                v = stack.pop()
                if ((visited >> v) & 1):
                    continue
                visited |= 1 << v
                comp_mask |= 1 << v
                t = neighbors[v]
                while t:
                    lsb = t & -t
                    u = lsb.bit_length() - 1
                    if ((visited >> u) & 1) == 0:
                        stack.append(u)
                    t ^= lsb
            # collect node indices from comp_mask
            comp_nodes: List[int] = []
            t = comp_mask
            while t:
                lsb = t & -t
                v = lsb.bit_length() - 1
                comp_nodes.append(v)
                t ^= lsb
            components.append(comp_nodes)

        result_nodes: List[int] = []

        # fast bit count
        bit_count = int.bit_count if hasattr(int, "bit_count") else lambda x: bin(x).count("1")

        # ensure recursion limit is adequate
        import sys
        sys.setrecursionlimit(max(10000, 1000 + n * 10))

        # Solve MIS per component
        for comp in components:
            k = len(comp)
            if k == 0:
                continue
            if k == 1:
                result_nodes.append(comp[0])
                continue

            # map original index -> local index
            orig_to_local = [-1] * n
            for idx, node in enumerate(comp):
                orig_to_local[node] = idx

            # mask of nodes in component (original indexing)
            comp_orig_mask = 0
            for node in comp:
                comp_orig_mask |= 1 << node

            # build local neighbor masks (original graph, restricted to component)
            local_neighbors: List[int] = [0] * k
            for i, u in enumerate(comp):
                mask = neighbors[u] & comp_orig_mask
                lm = 0
                t = mask
                while t:
                    lsb = t & -t
                    v = lsb.bit_length() - 1
                    lm |= 1 << orig_to_local[v]
                    t ^= lsb
                local_neighbors[i] = lm

            local_all = (1 << k) - 1
            # complement neighbors in local indexing (no self-loops)
            comp_nbrs = [((~local_neighbors[i]) & local_all) & ~(1 << i) for i in range(k)]

            # Quick greedy independent set (on original) to seed best solution
            rem = local_all
            greedy_mask = 0
            while rem:
                t = rem
                pick = None
                min_deg = None
                while t:
                    lsb = t & -t
                    v = lsb.bit_length() - 1
                    d = bit_count(local_neighbors[v] & rem)
                    if min_deg is None or d < min_deg:
                        min_deg = d
                        pick = v
                        if d == 0:
                            break
                    t ^= lsb
                if pick is None:
                    break
                greedy_mask |= 1 << pick
                rem &= ~((local_neighbors[pick] | (1 << pick)))

            best_mask = greedy_mask
            best_size = bit_count(best_mask)
            nbrs = comp_nbrs  # alias for complement adjacency

            # Greedy coloring on the current candidate set to provide bounds
            def greedy_color(P_mask: int):
                nodes: List[int] = []
                t = P_mask
                while t:
                    lsb = t & -t
                    v = lsb.bit_length() - 1
                    nodes.append(v)
                    t ^= lsb
                # order by degree (within P) descending to improve coloring quality
                nodes.sort(key=lambda x: bit_count(nbrs[x] & P_mask), reverse=True)

                order: List[int] = []
                colors: List[int] = []
                node_list = nodes
                color = 0
                while node_list:
                    color += 1
                    cur_mask = 0
                    next_list: List[int] = []
                    for v in node_list:
                        if (nbrs[v] & cur_mask) == 0:
                            order.append(v)
                            colors.append(color)
                            cur_mask |= 1 << v
                        else:
                            next_list.append(v)
                    node_list = next_list
                return order, colors

            # Recursive branch-and-bound (search for maximum clique in complement)
            def expand(R_mask: int, P_mask: int, rsize: int):
                nonlocal best_mask, best_size
                if P_mask == 0:
                    if rsize > best_size:
                        best_size = rsize
                        best_mask = R_mask
                    return
                order, colors = greedy_color(P_mask)
                # process vertices in reverse order with color bounds
                for i in range(len(order) - 1, -1, -1):
                    v = order[i]
                    c = colors[i]
                    if rsize + c <= best_size:
                        return
                    newR = R_mask | (1 << v)
                    newP = P_mask & nbrs[v]
                    expand(newR, newP, rsize + 1)
                    P_mask &= ~(1 << v)

            if best_size < k:
                expand(0, local_all, 0)

            # map best_mask (local indices) back to original indices
            bm = best_mask
            while bm:
                lsb = bm & -bm
                v = lsb.bit_length() - 1
                result_nodes.append(comp[v])
                bm ^= lsb

        result_nodes.sort()
        return result_nodes