import sys
from typing import Any, List, Tuple

import numpy as np

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Any:
        """
        Solve the Queens with Obstacles problem.

        Approach:
        - Build the conflict graph: vertices are free squares; an edge exists
          if two squares see each other along queen moves without an obstacle
          in between.
        - The task is a maximum independent set (MIS) on this graph.
        - Compute MIS by finding a maximum clique on the complement graph
          using a bitset-based Tomita-style branch-and-bound with greedy
          coloring (fast in practice for moderate sizes).
        """
        board = np.asarray(problem, dtype=bool)
        if board.size == 0:
            return []

        n_rows, n_cols = board.shape

        # Collect free cells
        cells: List[Tuple[int, int]] = []
        pos2idx = {}
        for r in range(n_rows):
            for c in range(n_cols):
                if not board[r, c]:
                    idx = len(cells)
                    cells.append((r, c))
                    pos2idx[(r, c)] = idx

        N = len(cells)
        if N == 0:
            return []

        # Build conflict adjacency as bitsets: conf[i] has bits set for vertices that attack i
        conf = [0] * N
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i, (r, c) in enumerate(cells):
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                while 0 <= rr < n_rows and 0 <= cc < n_cols:
                    if board[rr, cc]:
                        break
                    j = pos2idx.get((rr, cc))
                    if j is not None:
                        conf[i] |= (1 << j)
                    rr += dr
                    cc += dc

        # degrees for heuristics
        deg = [x.bit_count() for x in conf]

        ALL_MASK = (1 << N) - 1

        # Quick greedy independent set to get a lower bound
        available = ALL_MASK
        sol_mask = 0
        while available:
            tmp = available
            best_v = None
            best_deg = 10 ** 9
            # choose vertex with minimal degree among available
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                tmp ^= lsb
                dv = deg[v]
                if dv < best_deg:
                    best_deg = dv
                    best_v = v
            if best_v is None:
                break
            sol_mask |= (1 << best_v)
            # remove chosen and its conflicting neighbors
            available &= ~((1 << best_v) | conf[best_v])

        best_mask = sol_mask
        best_size = sol_mask.bit_count()

        # Complement adjacency for maximum clique search
        comp_adj = [0] * N
        for i in range(N):
            comp_adj[i] = ALL_MASK & ~(conf[i] | (1 << i))

        sys.setrecursionlimit(10000)

        # Tomita-style expand with greedy coloring bound
        def expand(R_mask: int, P_mask: int, sizeR: int):
            nonlocal best_mask, best_size, comp_adj

            if P_mask == 0:
                if sizeR > best_size:
                    best_size = sizeR
                    best_mask = R_mask
                return

            # Greedy coloring of P to produce order and color bounds
            order: List[int] = []
            color: List[int] = []
            P_copy = P_mask
            color_num = 0
            while P_copy:
                available_local = P_copy
                while available_local:
                    lsb = available_local & -available_local
                    v = lsb.bit_length() - 1
                    order.append(v)
                    color.append(color_num + 1)
                    P_copy ^= (1 << v)
                    available_local ^= (1 << v)
                    # remove neighbors (in complement graph) from this color class
                    available_local &= ~comp_adj[v]
                color_num += 1

            # Process vertices in reverse order (largest color first)
            for i in range(len(order) - 1, -1, -1):
                v = order[i]
                c = color[i]
                # Bound: if even with color-based bound we cannot beat best, prune
                if sizeR + c <= best_size:
                    return
                # Include v
                newR = R_mask | (1 << v)
                newP = P_mask & comp_adj[v]
                expand(newR, newP, sizeR + 1)
                # Exclude v from P_mask for the remaining loop
                P_mask &= ~(1 << v)

        # Launch search (clique in complement == independent set in original)
        expand(0, ALL_MASK, 0)

        # Decode best_mask into positions
        result: List[Tuple[int, int]] = []
        bm = best_mask
        while bm:
            lsb = bm & -bm
            v = lsb.bit_length() - 1
            result.append(cells[v])
            bm ^= lsb

        return result