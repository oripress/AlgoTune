try:
    from solver_c import solve_c as _solve_c
    has_c = True
except ImportError:
    _solve_c = None
    has_c = False
class Solver:
    def solve(self, problem, **kwargs):
        if has_c:
            return _solve_c(problem)
        # Maximum Independent Set via maximum clique in complement graph (bitset Branch & Bound)
        n = len(problem)
        # build complement neighbor bitsets
        neighbors = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            mask = full_mask & ~(1 << i)
            row = problem[i]
            for j in range(n):
                if row[j]:
                    mask &= ~(1 << j)
            neighbors[i] = mask
        # initial greedy clique in complement graph for lower bound
        order = sorted(range(n), key=lambda x: neighbors[x].bit_count())
        greedyR = 0
        for v in order:
            if (greedyR & ~neighbors[v]) == 0:
                greedyR |= (1 << v)
        bestR = greedyR
        best_size = greedyR.bit_count()
        # increase recursion limit
        import sys
        sys.setrecursionlimit(10000)
        # Branch & Bound recursion (Bron–Kerbosch with pivot)
        def bk(R, P, X):
            nonlocal bestR, best_size
            if P == 0 and X == 0:
                sz = R.bit_count()
                if sz > best_size:
                    best_size = sz
                    bestR = R
                return
            # bound: no possible improvement
            if R.bit_count() + P.bit_count() <= best_size:
                return
            # choose pivot u from P|X to minimize branches
            PU = P | X
            maxd = -1
            u = 0
            tmp = PU
            while tmp:
                v = tmp & -tmp
                tmp ^= v
                idx = v.bit_length() - 1
                d = (P & neighbors[idx]).bit_count()
                if d > maxd:
                    maxd = d
                    u = idx
            # branch on vertices not adjacent to pivot
            ext = P & ~neighbors[u]
            tmp = ext
            while tmp:
                v = tmp & -tmp
                tmp ^= v
                idx = v.bit_length() - 1
                vk = 1 << idx
                bk(R | vk, P & neighbors[idx], X & neighbors[idx])
                P &= ~vk
                X |= vk
        # initial call
        bk(0, full_mask, 0)
        # extract result vertices
        return [i for i in range(n) if (bestR >> i) & 1]