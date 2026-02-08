import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import csr_matrix


class Solver:
    def solve(self, problem, **kwargs):
        T = problem["T"]
        N = problem["N"]

        if T == 0 or N == 0:
            return [-1] * T

        prices = np.array(problem["prices"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.int64)
        probs = np.array(problem["probs"], dtype=np.float64)

        # Expected revenue matrix T x N
        rev = prices[np.newaxis, :] * probs

        # Quick check: if all capacities >= T, just pick best per period
        if np.all(capacities >= T):
            best_i = np.argmax(rev, axis=1)
            offer = []
            for t in range(T):
                if rev[t, best_i[t]] > 0:
                    offer.append(int(best_i[t]))
                else:
                    offer.append(-1)
            return offer

        # Try linear_sum_assignment for moderate sizes
        capped = np.minimum(capacities, T)
        total_product_cols = int(np.sum(capped))
        total_cols = total_product_cols + T  # + idle columns

        if total_cols <= 10000 and T <= 5000:
            # Build cost matrix
            cost = np.zeros((T, total_cols), dtype=np.float64)
            col_product = np.empty(total_cols, dtype=np.intp)

            col = 0
            for i in range(N):
                c = int(capped[i])
                if c > 0:
                    cost[:, col:col + c] = -rev[:, i:i + 1]
                    col_product[col:col + c] = i
                    col += c

            # Idle columns (cost = 0)
            col_product[total_product_cols:] = -1

            row_ind, col_ind = linear_sum_assignment(cost)

            offer = [-1] * T
            for r, c_idx in zip(row_ind, col_ind):
                offer[r] = int(col_product[c_idx])
            return offer

        # Fallback: solve LP (TU constraint matrix => integral solution)
        num_vars = T * N
        c = -rev.ravel()

        indices = np.arange(num_vars)
        row_period = indices // N
        row_product = (indices % N) + T

        rows = np.concatenate([row_period, row_product])
        cols = np.concatenate([indices, indices])
        data = np.ones(2 * num_vars)

        A_ub = csr_matrix((data, (rows, cols)), shape=(T + N, num_vars))
        b_ub = np.concatenate([np.ones(T), capacities.astype(float)])

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')

        if not result.success:
            return [-1] * T

        x = result.x.reshape(T, N)
        offer = []
        for t in range(T):
            idx = int(np.argmax(x[t]))
            if x[t, idx] > 0.5:
                offer.append(idx)
            else:
                offer.append(-1)

        return offer