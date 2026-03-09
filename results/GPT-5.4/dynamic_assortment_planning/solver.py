from __future__ import annotations

from typing import Any
import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[int]:
        T = int(problem["T"])
        N = int(problem["N"])
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        if T <= 0:
            return []

        active_products: list[int] = []
        active_caps: list[int] = []
        values: list[list[float]] = []

        for i in range(N):
            cap_i = capacities[i]
            if cap_i <= 0:
                continue
            price_i = prices[i]
            if price_i <= 0:
                continue
            vals_i = [price_i * probs[t][i] for t in range(T)]
            if max(vals_i, default=0.0) <= 0.0:
                continue
            active_products.append(i)
            active_caps.append(min(cap_i, T))
            values.append(vals_i)

        if not active_products:
            return [-1] * T

        m = len(active_products)
        val_mt = np.asarray(values, dtype=float)  # shape (m, T)
        val_tm = val_mt.T  # shape (T, m)

        if m == 1:
            vals0 = val_mt[0]
            ans = [-1] * T
            cap0 = active_caps[0]
            for t in heapq.nlargest(cap0, range(T), key=vals0.__getitem__):
                if vals0[t] > 0.0:
                    ans[t] = active_products[0]
            return ans

        best_idx = np.argmax(val_tm, axis=1)
        best_val = val_tm[np.arange(T), best_idx]
        pos_mask = best_val > 0.0
        best_counts = np.bincount(best_idx[pos_mask], minlength=m)
        if np.all(best_counts <= np.asarray(active_caps)):
            ans = [-1] * T
            active_arr = np.asarray(active_products)
            chosen_periods = np.nonzero(pos_mask)[0]
            chosen_products = active_arr[best_idx[pos_mask]]
            for t, p in zip(chosen_periods.tolist(), chosen_products.tolist()):
                ans[t] = p
            return ans

        total_slots = sum(active_caps)
        if total_slots > 0 and (T + total_slots) <= 700 and T * (T + total_slots) <= 250000:
            slot_prod = np.repeat(np.arange(m, dtype=np.int32), np.asarray(active_caps, dtype=np.int32))
            w = np.zeros((T, total_slots + T), dtype=float)
            w[:, :total_slots] = val_tm[:, slot_prod]
            row_ind, col_ind = linear_sum_assignment(-w)
            ans = [-1] * T
            active_arr = np.asarray(active_products)
            for t, c in zip(row_ind.tolist(), col_ind.tolist()):
                if c < total_slots and w[t, c] > 0.0:
                    ans[t] = int(active_arr[slot_prod[c]])
            return ans

        nz_t, nz_i = np.nonzero(val_tm > 0.0)
        E = len(nz_t)
        if E == 0:
            return [-1] * T

        edge_v = val_tm[nz_t, nz_i]
        edge_ids = np.arange(E, dtype=np.int32)
        rows = np.concatenate((nz_t.astype(np.int32), (T + nz_i).astype(np.int32)))
        cols = np.concatenate((edge_ids, edge_ids))
        data = np.ones(2 * E, dtype=float)

        A_ub = coo_matrix((data, (rows, cols)), shape=(T + m, E))
        b_ub = np.concatenate((np.ones(T, dtype=float), np.asarray(active_caps, dtype=float)))
        c = -edge_v.astype(float, copy=False)

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(0.0, 1.0),
            method="highs-ds",
        )

        if not res.success:
            ans = [-1] * T
            active_arr = np.asarray(active_products)
            chosen_periods = np.nonzero(pos_mask)[0]
            chosen_products = active_arr[best_idx[pos_mask]]
            counts = np.zeros(m, dtype=np.int32)
            for t in chosen_periods.tolist():
                i = int(best_idx[t])
                if counts[i] < active_caps[i]:
                    counts[i] += 1
                    ans[t] = int(active_arr[i])
            return ans

        x = res.x
        ans = [-1] * T
        best_x = np.zeros(T, dtype=float)
        active_arr = np.asarray(active_products)
        for e, val in enumerate(x):
            t = int(nz_t[e])
            if val > best_x[t] and val > 1e-7:
                best_x[t] = float(val)
                ans[t] = int(active_arr[int(nz_i[e])])

        return ans