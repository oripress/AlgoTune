import sys
sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        # build all possible mapping pairs (i in G, p in H)
        Pairs = [(i, p) for i in range(n) for p in range(m)]
        N = len(Pairs)
        # adjacency bitmasks for the association graph
        neighbors = [0] * N
        # connect u--v iff i!=j, p!=q and A[i][j]==B[p][q]
        for u in range(N):
            i, p = Pairs[u]
            for v in range(u + 1, N):
                j, q = Pairs[v]
                if i != j and p != q and A[i][j] == B[p][q]:
                    neighbors[u] |= 1 << v
                    neighbors[v] |= 1 << u
        # reorder vertices by degree descending (heuristic)
        degs = [nb.bit_count() for nb in neighbors]
        order = sorted(range(N), key=lambda u: degs[u], reverse=True)
        old_to_new = {old: new for new, old in enumerate(order)}
        new_neighbors = [0] * N
        for old in range(N):
            new_u = old_to_new[old]
            mask_old = neighbors[old]
            mask_new = 0
            # remap neighbor bits
            while mask_old:
                v_old = (mask_old & -mask_old).bit_length() - 1
                mask_old &= mask_old - 1
                mask_new |= 1 << old_to_new[v_old]
            new_neighbors[new_u] = mask_new
        neighbors = new_neighbors
        # prepare for Bron–Kerbosch pivot without exclusion set
        max_mask = 0
        max_size = 0
        neighbor_local = neighbors
        popcount = int.bit_count
        bitlen = int.bit_length
        def expand(P, r_count, R_mask):
            nonlocal max_mask, max_size
            if P == 0:
                if r_count > max_size:
                    max_size = r_count
                    max_mask = R_mask
                return
            # bound: prune if maximal possible clique <= current best
            if r_count + popcount(P) <= max_size:
                return
            # pivot selection: u in P maximizing |P & N(u)|
            tmp = P
            max_u = -1
            max_cnt = -1
            while tmp:
                lsb = tmp & -tmp
                u = bitlen(lsb) - 1
                tmp &= tmp - 1
                cnt = popcount(P & neighbor_local[u])
                if cnt > max_cnt:
                    max_cnt = cnt
                    max_u = u
            # candidates: P without neighbors of pivot
            cand = P & ~neighbor_local[max_u]
            # explore each candidate
            while cand:
                lsb = cand & -cand
                cand &= cand - 1
                v = bitlen(lsb) - 1
                expand(P & neighbor_local[v], r_count + 1, R_mask | lsb)
                P &= ~lsb
        # start search
        expand((1 << N) - 1, 0, 0)
        # reconstruct result
        result = []
        mask = max_mask
        while mask:
            lsb = mask & -mask
            v = bitlen(lsb) - 1
            result.append(Pairs[order[v]])
            mask &= mask - 1
        result.sort()
        return result