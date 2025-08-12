from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        A_list = problem["A"]
        b_list = problem["b"]
        n = len(b_list)
        # trivial 1×1
        if n == 1:
            return [b_list[0] / A_list[0][0]]
        # trivial 2×2
        if n == 2:
            a, b = A_list[0]
            c, d = A_list[1]
            e, f = b_list
            det = a * d - b * c
            return [(d * e - b * f) / det, (-c * e + a * f) / det]
        # pure Python Gaussian elimination for small systems
        if n <= 16:
            # make copies of rows
            A2 = [row.copy() for row in A_list]
            b2 = b_list.copy()
            # forward elimination
            for i in range(n):
                # partial pivoting
                max_row = i
                max_val = abs(A2[i][i])
                for r in range(i + 1, n):
                    v = abs(A2[r][i])
                    if v > max_val:
                        max_val = v
                        max_row = r
                if max_row != i:
                    A2[i], A2[max_row] = A2[max_row], A2[i]
                    b2[i], b2[max_row] = b2[max_row], b2[i]
                diag = A2[i][i]
                row_i = A2[i]
                # eliminate below
                for r in range(i + 1, n):
                    row_r = A2[r]
                    factor = row_r[i] / diag
                    for c in range(i + 1, n):
                        row_r[c] -= factor * row_i[c]
                    b2[r] -= factor * b2[i]
            # back substitution
            x = [0.0] * n
            for i in range(n - 1, -1, -1):
                s = b2[i]
                row_i = A2[i]
                for c in range(i + 1, n):
                    s -= row_i[c] * x[c]
                x[i] = s / row_i[i]
            return x
        # fallback to NumPy solver for larger systems
        A = np.array(A_list, dtype=float, order='F')
        b = np.array(b_list, dtype=float, order='F')
        x = np.linalg.solve(A, b)
        return x.tolist()