class Solver:
    def solve(self, problem, **kwargs):
        # Build bitsets for original and complement graphs
        n = len(problem)
        mask_all = (1 << n) - 1
        orig_adj = [0] * n
        for i in range(n):
            bits = 0
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    bits |= 1 << j
            orig_adj[i] = bits
        comp_adj = [((~orig_adj[i]) & mask_all) & ~(1 << i) for i in range(n)]
        # Initial greedy independent set for a good lower bound
        deg = [orig_adj[i].bit_count() for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda x: deg[x])
        best_sol = []
        P_g = mask_all
        for v in order:
            if (P_g >> v) & 1:
                best_sol.append(v)
                P_g &= ~((1 << v) | orig_adj[v])
        # best_ref[0] = best size, best_ref[1] = best solution list
        best_ref = [len(best_sol), best_sol.copy()]
        # Aliases for speed
        cd = comp_adj
        bl = int.bit_length
        # Branch-and-bound with greedy coloring bound
        def expand(P, size, R):
            # Greedy coloring on P for upper bounds
            c = [0] * n
            order2 = []
            color = 0
            Q = P
            while Q:
                color += 1
                avail = Q
                while avail:
                    b = avail & -avail
                    v = bl(b) - 1
                    avail ^= b
                    Q ^= b
                    c[v] = color
                    order2.append(v)
                    avail &= ~cd[v]
            # Branch vertices in reverse color order
            for v in reversed(order2):
                if size + c[v] <= best_ref[0]:
                    return
                R.append(v)
                new_size = size + 1
                if new_size > best_ref[0]:
                    best_ref[0] = new_size
                    best_ref[1] = R.copy()
                newP = P & cd[v]
                if newP:
                    expand(newP, new_size, R)
                R.pop()
                P &= ~(1 << v)
        expand(mask_all, 0, [])
        # Return sorted solution indices
        return sorted(best_ref[1])