from __future__ import annotations

from typing import Any

import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def __init__(self) -> None:
        self._node_limit = 200000

    def solve(self, problem, **kwargs) -> Any:
        board = np.asarray(problem, dtype=bool)
        n, m = board.shape
        if n == 0 or m == 0:
            return []

        cells, family_cells, family_counts = self._build_cells_and_segments(board)
        if not cells:
            return []

        search_result = self._try_search(cells, family_cells, family_counts)
        if search_result is not None:
            return search_result

        return self._solve_cpsat(cells, family_cells)

    def _build_cells_and_segments(
        self, board: np.ndarray
    ) -> tuple[
        list[tuple[tuple[int, int], tuple[int, int, int, int]]],
        tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]],
        tuple[int, int, int, int],
    ]:
        n, m = board.shape
        row_ids = np.full((n, m), -1, dtype=np.int32)
        col_ids = np.full((n, m), -1, dtype=np.int32)
        d1_ids = np.full((n, m), -1, dtype=np.int32)
        d2_ids = np.full((n, m), -1, dtype=np.int32)

        row_count = 0
        for r in range(n):
            active = False
            current = -1
            for c in range(m):
                if board[r, c]:
                    active = False
                    continue
                if not active:
                    current = row_count
                    row_count += 1
                    active = True
                row_ids[r, c] = current

        col_count = 0
        for c in range(m):
            active = False
            current = -1
            for r in range(n):
                if board[r, c]:
                    active = False
                    continue
                if not active:
                    current = col_count
                    col_count += 1
                    active = True
                col_ids[r, c] = current

        d1_count = 0
        for start_r in range(n):
            r = start_r
            c = 0
            active = False
            current = -1
            while r < n and c < m:
                if board[r, c]:
                    active = False
                else:
                    if not active:
                        current = d1_count
                        d1_count += 1
                        active = True
                    d1_ids[r, c] = current
                r += 1
                c += 1
        for start_c in range(1, m):
            r = 0
            c = start_c
            active = False
            current = -1
            while r < n and c < m:
                if board[r, c]:
                    active = False
                else:
                    if not active:
                        current = d1_count
                        d1_count += 1
                        active = True
                    d1_ids[r, c] = current
                r += 1
                c += 1

        d2_count = 0
        for start_c in range(m):
            r = 0
            c = start_c
            active = False
            current = -1
            while r < n and c >= 0:
                if board[r, c]:
                    active = False
                else:
                    if not active:
                        current = d2_count
                        d2_count += 1
                        active = True
                    d2_ids[r, c] = current
                r += 1
                c -= 1
        for start_r in range(1, n):
            r = start_r
            c = m - 1
            active = False
            current = -1
            while r < n and c >= 0:
                if board[r, c]:
                    active = False
                else:
                    if not active:
                        current = d2_count
                        d2_count += 1
                        active = True
                    d2_ids[r, c] = current
                r += 1
                c -= 1

        cells: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
        row_cells = [[] for _ in range(row_count)]
        col_cells = [[] for _ in range(col_count)]
        d1_cells = [[] for _ in range(d1_count)]
        d2_cells = [[] for _ in range(d2_count)]

        open_positions = np.argwhere(~board)
        for idx, (r, c) in enumerate(open_positions):
            ids = (
                int(row_ids[r, c]),
                int(col_ids[r, c]),
                int(d1_ids[r, c]),
                int(d2_ids[r, c]),
            )
            cells.append(((int(r), int(c)), ids))
            row_cells[ids[0]].append(idx)
            col_cells[ids[1]].append(idx)
            d1_cells[ids[2]].append(idx)
            d2_cells[ids[3]].append(idx)

        return cells, (row_cells, col_cells, d1_cells, d2_cells), (
            row_count,
            col_count,
            d1_count,
            d2_count,
        )

    def _try_search(
        self,
        cells: list[tuple[tuple[int, int], tuple[int, int, int, int]]],
        family_cells: tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]],
        family_counts: tuple[int, int, int, int],
    ) -> list[tuple[int, int]] | None:
        primary = min(range(4), key=lambda i: family_counts[i])
        primary_count = family_counts[primary]
        if primary_count > 60 and len(cells) > 150:
            return None

        other_families = [i for i in range(4) if i != primary]
        segments = family_cells[primary]

        seg_data: list[list[tuple[int, int, int, int]]] = []
        for seg in segments:
            options: list[tuple[int, int, int, int]] = []
            for cell_idx in seg:
                ids = cells[cell_idx][1]
                options.append(
                    (
                        1 << ids[other_families[0]],
                        1 << ids[other_families[1]],
                        1 << ids[other_families[2]],
                        cell_idx,
                    )
                )
            seg_data.append(options)

        order = list(range(primary_count))
        order.sort(key=lambda s: len(seg_data[s]))
        ordered_segments = [seg_data[s] for s in order]

        suffix1 = [0] * (primary_count + 1)
        suffix2 = [0] * (primary_count + 1)
        suffix3 = [0] * (primary_count + 1)
        for i in range(primary_count - 1, -1, -1):
            u1 = suffix1[i + 1]
            u2 = suffix2[i + 1]
            u3 = suffix3[i + 1]
            for m1, m2, m3, _ in ordered_segments[i]:
                u1 |= m1
                u2 |= m2
                u3 |= m3
            suffix1[i] = u1
            suffix2[i] = u2
            suffix3[i] = u3

        best: list[int] = self._greedy_incumbent(ordered_segments)
        best_len = len(best)
        chosen = [0] * primary_count
        nodes = 0
        aborted = False

        def dfs(i: int, used1: int, used2: int, used3: int, depth: int) -> None:
            nonlocal best, best_len, nodes, aborted
            if aborted:
                return
            nodes += 1
            if nodes > self._node_limit:
                aborted = True
                return

            rem = primary_count - i
            if depth + rem <= best_len:
                return

            ub = depth + rem
            free1 = (suffix1[i] & ~used1).bit_count()
            if depth + free1 < ub:
                ub = depth + free1
            free2 = (suffix2[i] & ~used2).bit_count()
            if depth + free2 < ub:
                ub = depth + free2
            free3 = (suffix3[i] & ~used3).bit_count()
            if depth + free3 < ub:
                ub = depth + free3
            if ub <= best_len:
                return

            if i == primary_count:
                if depth > best_len:
                    best_len = depth
                    best = chosen[:depth]
                return

            seg = ordered_segments[i]
            placed_any = False
            for m1, m2, m3, cell_idx in seg:
                if (used1 & m1) or (used2 & m2) or (used3 & m3):
                    continue
                placed_any = True
                chosen[depth] = cell_idx
                dfs(i + 1, used1 | m1, used2 | m2, used3 | m3, depth + 1)
                if aborted:
                    return

            if depth + (primary_count - i - 1) > best_len or not placed_any:
                dfs(i + 1, used1, used2, used3, depth)

        dfs(0, 0, 0, 0, 0)
        if aborted:
            return None
        return [cells[idx][0] for idx in best]

    def _greedy_incumbent(
        self, ordered_segments: list[list[tuple[int, int, int, int]]]
    ) -> list[int]:
        used1 = 0
        used2 = 0
        used3 = 0
        sol: list[int] = []
        for seg in ordered_segments:
            best_item: tuple[int, int, int, int] | None = None
            for item in seg:
                m1, m2, m3, _ = item
                if (used1 & m1) or (used2 & m2) or (used3 & m3):
                    continue
                best_item = item
                break
            if best_item is not None:
                m1, m2, m3, idx = best_item
                used1 |= m1
                used2 |= m2
                used3 |= m3
                sol.append(idx)
        return sol

    def _solve_cpsat(
        self,
        cells: list[tuple[tuple[int, int], tuple[int, int, int, int]]],
        family_cells: tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]],
    ) -> list[tuple[int, int]]:
        model = cp_model.CpModel()
        vars_ = [model.NewBoolVar(f"x{i}") for i in range(len(cells))]

        for fam in family_cells:
            for seg in fam:
                if len(seg) > 1:
                    model.AddAtMostOne(vars_[i] for i in seg)

        model.Maximize(sum(vars_))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 0
        solver.parameters.symmetry_level = 2

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        return [cells[i][0] for i, _ in enumerate(vars_) if solver.BooleanValue(vars_[i])]