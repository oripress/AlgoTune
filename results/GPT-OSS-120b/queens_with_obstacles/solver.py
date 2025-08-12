import numpy as np
from ortools.sat.python import cp_model
from typing import List, Tuple, Set

def _build_segments(board: np.ndarray):
    """Assign segment IDs for rows, columns, and both diagonals.
    Cells with obstacles get -1. Returns four 2‑D arrays of segment IDs."""
    n, m = board.shape
    row_seg = -np.ones((n, m), dtype=int)
    col_seg = -np.ones((n, m), dtype=int)
    diag1_seg = -np.ones((n, m), dtype=int)   # top‑left to bottom‑right (r‑c)
    diag2_seg = -np.ones((n, m), dtype=int)   # top‑right to bottom‑left (r+ c)

    seg_counter = 0
    # Row segments
    for r in range(n):
        cur = -1
        for c in range(m):
            if board[r, c]:
                cur = -1
            else:
                if cur == -1:
                    cur = seg_counter
                    seg_counter += 1
                row_seg[r, c] = cur

    # Column segments
    for c in range(m):
        cur = -1
        for r in range(n):
            if board[r, c]:
                cur = -1
            else:
                if cur == -1:
                    cur = seg_counter
                    seg_counter += 1
                col_seg[r, c] = cur

    # Diagonal (r‑c) segments
    for start in range(-n + 1, m):
        cur = -1
        r_start = max(0, -start)
        c_start = max(0, start)
        while r_start < n and c_start < m:
            r, c = r_start, c_start
            if board[r, c]:
                cur = -1
            else:
                if cur == -1:
                    cur = seg_counter
                    seg_counter += 1
                diag1_seg[r, c] = cur
            r_start += 1
            c_start += 1

    # Diagonal (r+ c) segments
    for start in range(n + m - 1):
        cur = -1
        r_start = max(0, start - m + 1)
        c_start = min(m - 1, start)
        while r_start < n and c_start >= 0:
            r, c = r_start, c_start
            if board[r, c]:
                cur = -1
            else:
                if cur == -1:
                    cur = seg_counter
                    seg_counter += 1
                diag2_seg[r, c] = cur
            r_start += 1
            c_start -= 1

    return row_seg, col_seg, diag1_seg, diag2_seg
 
def _greedy_solution(board: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Greedy placement: scan cells row‑major and place a queen if it does not
    attack any already placed queen, respecting obstacles.
    Returns a set of positions (r, c).
    """
    n, m = board.shape
    placed: Set[Tuple[int, int]] = set()
    obstacles = {(r, c) for r in range(n) for c in range(m) if board[r, c]}

    def conflicts(r: int, c: int) -> bool:
        for pr, pc in placed:
            dr = pr - r
            dc = pc - c
            if dr == 0 or dc == 0 or dr == dc or dr == -dc:
                step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
                step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
                nr, nc = r + step_r, c + step_c
                blocked = False
                while (nr, nc) != (pr, pc):
                    if (nr, nc) in obstacles:
                        blocked = True
                        break
                    nr += step_r
                    nc += step_c
                if not blocked:
                    return True
        return False

    for r in range(n):
        for c in range(m):
            if board[r, c]:
                continue
            if not conflicts(r, c):
                placed.add((r, c))
    return placed
class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> List[Tuple[int, int]]:
        """
        Solve the Queens with Obstacles problem using CP‑SAT.
        """
        board = problem
        n, m = board.shape
        model = cp_model.CpModel()

        # Decision variables for each free cell
        queen_vars = {}
        for r in range(n):
            for c in range(m):
                if not board[r, c]:
                    queen_vars[(r, c)] = model.NewBoolVar(f"q_{r}_{c}")

        # Build segment IDs
        row_seg, col_seg, diag1_seg, diag2_seg = _build_segments(board)

        # Prepare per‑segment variable lists using a dict
        seg_vars: dict[int, List[cp_model.IntVar]] = {}
        for (r, c), var in queen_vars.items():
            for seg_id in (row_seg[r, c], col_seg[r, c],
                           diag1_seg[r, c], diag2_seg[r, c]):
                if seg_id != -1:
                    seg_vars.setdefault(seg_id, []).append(var)

        # Add at‑most‑one constraints per segment
        for vars_in_seg in seg_vars.values():
            if len(vars_in_seg) > 1:
                model.AddAtMostOne(vars_in_seg)

        # Objective: maximize number of queens
        model.Maximize(sum(queen_vars.values()))

        # Provide a greedy hint to speed up solving
        greedy_solution = _greedy_solution(board)
        for (r, c), var in queen_vars.items():
            hint = 1 if (r, c) in greedy_solution else 0
            model.AddHint(var, hint)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = kwargs.get("num_workers", 8)
        if "time_limit_seconds" in kwargs:
            solver.parameters.max_time_in_seconds = kwargs["time_limit_seconds"]

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for (r, c), var in queen_vars.items() if solver.Value(var)]
        else:
            return []

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for (r, c), var in queen_vars.items() if solver.Value(var)]
        else:
            return []
            solver.parameters.max_time_in_seconds = kwargs["time_limit_seconds"]

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [(r, c) for (r, c), var in queen_vars.items()
                    if solver.Value(var) == 1]
        return []