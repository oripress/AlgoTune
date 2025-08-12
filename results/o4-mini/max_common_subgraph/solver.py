from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: dict, **kwargs) -> List[Tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        # build list of all possible node‐pairs (i,p)
        V = [(i, p) for i in range(n) for p in range(m)]
        Vcount = len(V)
        # build association adjacencies as bitsets via optimized precompute
        M_edge = [[0] * m for _ in range(n)]
        M_non = [[0] * m for _ in range(n)]
        for j0 in range(n):
            for p0 in range(m):
                edge_mask = 0
                non_mask = 0
                for q0 in range(m):
                    if q0 == p0:
                        continue
                    bit = 1 << (j0 * m + q0)
                    if B[p0][q0]:
                        edge_mask |= bit
                    else:
                        non_mask |= bit
                M_edge[j0][p0] = edge_mask
                M_non[j0][p0] = non_mask
        N = [0] * Vcount
        for i0 in range(n):
            for p0 in range(m):
                v = i0 * m + p0
                mask = 0
                Ai = A[i0]
                for j0 in range(n):
                    if j0 == i0:
                        continue
                    if Ai[j0]:
                        mask |= M_edge[j0][p0]
                    else:
                        mask |= M_non[j0][p0]
                N[v] = mask
        # degrees
        deg = [nb.bit_count() for nb in N]
        # initial greedy clique for lower bound
        best_clique = 0
        # pick highest‐degree start
        if Vcount > 0:
            v0 = max(range(Vcount), key=lambda x: deg[x])
            clique = 1 << v0
            P = N[v0]
            while P:
                # pick next by global degree
                lsb = P & -P
                # find all candidates
                cand = P
                best_v = None
                best_d = -1
                while cand:
                    lb = cand & -cand
                    vv = lb.bit_length() - 1
                    if deg[vv] > best_d:
                        best_d = deg[vv]
                        best_v = vv
                    cand &= ~lb
                clique |= (1 << best_v)
                P &= N[best_v]
            best_clique = clique
        best_size = best_clique.bit_count()
        # Tomita MCR maximum clique with coloring bound
        full = (1 << Vcount) - 1
        def mcr(R: int, P: int, r_size: int):
            nonlocal best_size, best_clique
            # if no candidates, update best clique
            if P == 0:
                if r_size > best_size:
                    best_size = r_size
                    best_clique = R
                return
            # build list of vertices in P
            vs = []
            tmp = P
            while tmp:
                lb = tmp & -tmp
                v = lb.bit_length() - 1
                vs.append(v)
                tmp &= tmp - 1
            # greedy coloring bound
            vs.sort(key=lambda v: deg[v], reverse=True)
            color_masks: List[int] = []
            color: dict[int, int] = {}
            for v in vs:
                for ci, mask in enumerate(color_masks):
                    if (mask & N[v]) == 0:
                        color_masks[ci] |= (1 << v)
                        color[v] = ci + 1
                        break
                else:
                    color_masks.append(1 << v)
                    color[v] = len(color_masks)
            # branch on candidates in order of decreasing color label
            for v in sorted(vs, key=lambda v: color[v], reverse=True):
                # pruning: if even best-color bound can't beat current best
                if r_size + color[v] <= best_size:
                    return
                lbv = 1 << v
                mcr(R | lbv, P & N[v], r_size + 1)
                P &= ~lbv
        # launch MCR search
        mcr(0, full, 0)
        # extract solution mapping from bitset
        solution: List[Tuple[int, int]] = []
        mask = best_clique
        while mask:
            lb = mask & -mask
            idx = lb.bit_length() - 1
            solution.append(V[idx])
            mask &= mask - 1
        return solution