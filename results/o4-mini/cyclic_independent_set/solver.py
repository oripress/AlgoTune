import numpy as np
from numba import njit
from itertools import product

@njit(cache=True, fastmath=True)
def c7_solve_njit(n, num_nodes, mod, powers, weights1, weights2,
                  freq, p_values, sum_w1, limit, shift_arr, c_mod_order):
    # compute total vertices
    total = 1
    for _ in range(n):
        total *= num_nodes
    # compute clipped-sum mod list
    c_mod_list = np.empty(total, dtype=np.int64)
    for code in range(total):
        rem = code
        acc = sum_w1
        for i in range(n):
            digit = rem // powers[i]
            rem -= digit * powers[i]
            if digit > limit:
                digit = limit
            acc += weights1[i] * digit
        c_mod_list[code] = acc % mod
    # bucket sort codes by priority buckets in descending order
    sorted_codes = np.empty(total, dtype=np.int64)
    pos = 0
    for t in range(c_mod_order.shape[0]):
        cm = c_mod_order[t]
        for code in range(total):
            if c_mod_list[code] == cm:
                sorted_codes[pos] = code
                pos += 1
    # greedy selection
    blocked = np.zeros(total, dtype=np.uint8)
    sel = np.empty(total, dtype=np.int64)
    sel_count = 0
    # temporary digits array
    digits = np.empty(n, dtype=np.int64)
    sh_count = shift_arr.shape[0]
    for idx in range(total):
        code = sorted_codes[idx]
        if blocked[code] != 0:
            continue
        # select
        sel[sel_count] = code
        sel_count += 1
        # decode digits
        rem = code
        for i in range(n):
            d = rem // powers[i]
            rem -= d * powers[i]
            digits[i] = d
        # block neighbors (shift_arr now in [0..num_nodes-1])
        for s_idx in range(sh_count):
            nei = 0
            for i in range(n):
                d2 = digits[i] + shift_arr[s_idx, i]
                if d2 >= num_nodes:
                    d2 -= num_nodes
                nei += d2 * powers[i]
            blocked[nei] = 1
    return sel, sel_count

class Solver:
    def __init__(self):
        # force-compile the Numba function (not counted in solve runtime)
        n0 = 1
        num_nodes0 = 7
        mod0 = num_nodes0 - 2
        powers0 = np.array([1], dtype=np.int64)
        w1_0 = np.array([1], dtype=np.int64)
        w2_0 = np.array([1], dtype=np.int64)
        freq0 = np.zeros(mod0, dtype=np.int64)
        pv0 = np.zeros(mod0, dtype=np.int64)
        sum_w1_0 = int(w1_0.sum() % mod0)
        limit0 = num_nodes0 - 3
        # dummy shift in [0..6]
        shift0 = np.array([[0]], dtype=np.int64)
        cmo0 = np.array(sorted(range(mod0), key=lambda i: -pv0[i]), dtype=np.int64)
        _ = c7_solve_njit(n0, num_nodes0, mod0, powers0, w1_0, w2_0,
                          freq0, pv0, sum_w1_0, limit0,
                          shift0, cmo0)

    def solve(self, problem, **kwargs):
        num_nodes, n = problem
        if num_nodes != 7:
            raise ValueError(f"Unsupported num_nodes: {num_nodes}")
        mod = num_nodes - 2
        # positional powers for code ↔ tuple
        powers = np.array([num_nodes ** (n - 1 - i) for i in range(n)],
                          dtype=np.int64)
        # weights
        weights2 = (2 * powers) % mod
        weights1 = powers % mod
        # build frequency of 2*v*weights2 mod
        freq = np.zeros(mod, dtype=np.int64)
        for vec in product(range(1, n), repeat=n):
            a = 0
            for i, v in enumerate(vec):
                a = (a + v * weights2[i]) % mod
            freq[a] += 1
        # compute priority contributions
        p_values = np.zeros(mod, dtype=np.int64)
        for c_mod in range(mod):
            s = 0
            for k in range(mod):
                s += freq[k] * ((k + c_mod) % mod)
            p_values[c_mod] = s
        sum_w1 = int(weights1.sum() % mod)
        limit = num_nodes - 3
        # compute shift patterns and convert to 0..num_nodes-1
        shift_arr = np.array(list(product((-1, 0, 1), repeat=n)), dtype=np.int64)
        shift_arr = (shift_arr + num_nodes) % num_nodes
        # order of c_mod buckets by descending priority
        c_mod_order = np.array(sorted(range(mod), key=lambda i: -p_values[i]),
                               dtype=np.int64)
        # run Numba accelerated greedy
        sel_arr, sel_count = c7_solve_njit(n, num_nodes, mod, powers,
                                           weights1, weights2, freq,
                                           p_values, sum_w1, limit,
                                           shift_arr, c_mod_order)
        # decode selected codes
        result = []
        for k in range(sel_count):
            code = int(sel_arr[k])
            rem = code
            tup = []
            for i in range(n):
                d = rem // powers[i]
                rem -= d * powers[i]
                tup.append(int(d))
            result.append(tuple(tup))
        return result
        # neighbor shift patterns
        shift_arr = np.array(list(product((-1, 0, 1), repeat=n)),
                             dtype=np.int64)
        # run Numba accelerated greedy
        sel_arr, sel_count = c7_solve_njit(n, num_nodes, mod, powers,
                                           weights1, weights2, freq,
                                           p_values, sum_w1, limit,
                                           shift_arr, c_mod_order)
        # decode selected codes to tuples
        codes = sel_arr[:sel_count]
        result = []
        for code in codes:
            rem = int(code)
            tup = []
            for i in range(n):
                d = rem // powers[i]
                rem -= d * powers[i]
                tup.append(int(d))
            result.append(tuple(tup))
        return result