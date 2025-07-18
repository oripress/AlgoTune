# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
cimport cython
cdef extern from *:
    int __builtin_ctzll(unsigned long long)
cpdef tuple mwis_nb(int n, unsigned long long[::1] neighbor_masks, long long[::1] weights):
    """
    Cython-optimized iterative branch-and-bound for Maximum Weighted Independent Set
    on up to 64 nodes.
    Returns (best_weight, best_mask).
    """
    cdef unsigned long long full
    if n == 64:
        full = <unsigned long long>-1
    else:
        full = ((<unsigned long long>1 << n) - 1)
    cdef unsigned long long cand, excl, bitv, cmask, rem, lsb, tmp, cand_bits, lsb2, gre_cand, gre_mask, greedy_bits
    cdef long long cw, rem_sum, best_w, w2, max_wt, gre_w, greedy_max_wt
    cdef int idx, idx2, max_v, greedy_max_v
    cdef Py_ssize_t pos = 0
    cdef unsigned long long stack_cand[65]
    cdef unsigned long long stack_mask[65]
    cdef long long stack_weight[65]

    # initial state
    stack_cand[0] = full
    stack_mask[0] = 0
    stack_weight[0] = 0
    pos = 1
    best_w = 0
    best_mask = 0
    # Greedy initial solution for bound
    gre_cand = full
    gre_mask = 0
    gre_w = 0
    while gre_cand:
        greedy_bits = gre_cand
        greedy_max_wt = <long long>-9223372036854775808
        greedy_max_v = 0
        # find highest-weight vertex
        while greedy_bits:
            idx = __builtin_ctzll(greedy_bits)
            w2 = weights[idx]
            if w2 > greedy_max_wt:
                greedy_max_wt = w2
                greedy_max_v = idx
            greedy_bits &= greedy_bits - 1
        bitv = (<unsigned long long>1 << greedy_max_v)
        gre_mask |= bitv
        gre_w += weights[greedy_max_v]
        gre_cand &= ~ (neighbor_masks[greedy_max_v] | bitv)
    best_w = gre_w
    best_mask = gre_mask
    # iterative B&B
    while pos:
        pos -= 1
        cand = stack_cand[pos]
        cw = stack_weight[pos]
        cmask = stack_mask[pos]

        # bound: sum of remaining weights
        rem = cand
        rem_sum = 0
        while rem:
            lsb = rem & -rem
            tmp = lsb
            idx = 0
            while (tmp & 1) == 0:
                tmp >>= 1
                idx += 1
            rem_sum += weights[idx]
            rem &= rem - 1

        if cw + rem_sum <= best_w:
            continue

        # no candidates -> update best
        if cand == 0:
            if cw > best_w:
                best_w = cw
                best_mask = cmask
            continue

        # pick vertex of max weight in cand
        cand_bits = cand
        max_wt = -9223372036854775808  # min int64
        max_v = 0
        while cand_bits:
            lsb2 = cand_bits & -cand_bits
            tmp = lsb2
            idx2 = 0
            while (tmp & 1) == 0:
                tmp >>= 1
                idx2 += 1
            w2 = weights[idx2]
            if w2 > max_wt:
                max_wt = w2
                max_v = idx2
            cand_bits &= cand_bits - 1

        # branch: exclude vs include
        bitv = (<unsigned long long>1 << max_v)
        excl = neighbor_masks[max_v] | bitv

        # exclude branch
        stack_cand[pos] = cand & (~bitv)
        stack_weight[pos] = cw
        stack_mask[pos] = cmask
        pos += 1

        # include branch
        stack_cand[pos] = cand & (~excl)
        stack_weight[pos] = cw + weights[max_v]
        stack_mask[pos] = cmask | bitv
        pos += 1

    return best_w, best_mask