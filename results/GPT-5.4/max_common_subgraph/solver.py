class Solver:
    def solve(self, problem, **kwargs):
        A_in = problem["A"]
        B_in = problem["B"]

        n0 = len(A_in)
        m0 = len(B_in)
        if n0 == 0 or m0 == 0:
            return []

        swapped = False
        if n0 > m0:
            A_in, B_in = B_in, A_in
            n0, m0 = m0, n0
            swapped = True

        n = n0
        m = m0

        deg_a0 = [sum(row) for row in A_in]
        deg_b0 = [sum(row) for row in B_in]

        row_perm = sorted(range(n), key=lambda i: deg_a0[i], reverse=True)
        col_perm = sorted(range(m), key=lambda p: deg_b0[p], reverse=True)

        row_deg = [deg_a0[i] for i in row_perm]
        col_deg = [deg_b0[p] for p in col_perm]

        a_adj = [0] * n
        for i_new, i_old in enumerate(row_perm):
            row = A_in[i_old]
            bits = 0
            for j_new, j_old in enumerate(row_perm):
                if j_new != i_new and row[j_old]:
                    bits |= 1 << j_new
            a_adj[i_new] = bits

        b_adj = [0] * m
        for p_new, p_old in enumerate(col_perm):
            row = B_in[p_old]
            bits = 0
            for q_new, q_old in enumerate(col_perm):
                if q_new != p_new and row[q_old]:
                    bits |= 1 << q_new
            b_adj[p_new] = bits

        all_g = (1 << n) - 1
        all_h = (1 << m) - 1

        best_pairs = []
        best_size = 0

        def choose_class(classes):
            best_idx = -1
            best_h = m + 1
            best_g = -1
            for idx, (_, gcnt, _, hcnt) in enumerate(classes):
                if hcnt < best_h or (hcnt == best_h and gcnt > best_g):
                    best_idx = idx
                    best_h = hcnt
                    best_g = gcnt
            return best_idx

        def refine(classes, idx, v_bit, w_bit, v, w):
            av = a_adj[v]
            bw = b_adj[w]
            new_classes = []
            remain_bound = 0

            for j, (gmask, gcnt, hmask, hcnt) in enumerate(classes):
                if j == idx:
                    g2 = gmask ^ v_bit
                    h2 = hmask ^ w_bit
                    g2c = gcnt - 1
                    h2c = hcnt - 1
                    if g2c == 0 or h2c == 0:
                        continue
                else:
                    g2 = gmask
                    h2 = hmask
                    g2c = gcnt
                    h2c = hcnt

                ga = g2 & av
                ha = h2 & bw
                gac = ga.bit_count()
                hac = ha.bit_count()

                if gac and hac:
                    new_classes.append((ga, gac, ha, hac))
                    remain_bound += gac if gac < hac else hac

                gnc = g2c - gac
                if gnc:
                    hnc = h2c - hac
                    if hnc:
                        gn = g2 ^ ga
                        hn = h2 ^ ha
                        new_classes.append((gn, gnc, hn, hnc))
                        remain_bound += gnc if gnc < hnc else hnc

            return new_classes, remain_bound

        def greedy_seed(classes):
            cur = []
            while classes:
                idx = choose_class(classes)
                if idx < 0:
                    break

                gmask, _, hmask, hcnt = classes[idx]
                if hcnt == 0:
                    break

                v_bit = gmask & -gmask
                v = v_bit.bit_length() - 1

                if hcnt > 1:
                    rv = row_deg[v]
                    best_w = -1
                    best_score = -10**9
                    bits = hmask
                    while bits:
                        w_bit = bits & -bits
                        w = w_bit.bit_length() - 1
                        diff = rv - col_deg[w]
                        score = -diff if diff >= 0 else diff
                        if score > best_score:
                            best_score = score
                            best_w = w
                        bits ^= w_bit
                    w = best_w
                    w_bit = 1 << w
                else:
                    w_bit = hmask & -hmask
                    w = w_bit.bit_length() - 1

                cur.append((v, w))
                classes, _ = refine(classes, idx, v_bit, w_bit, v, w)

            return cur

        def search(classes, remain_bound, cur_pairs):
            nonlocal best_pairs, best_size

            cur_size = len(cur_pairs)
            if cur_size + remain_bound <= best_size:
                return

            if not classes:
                if cur_size > best_size:
                    best_pairs = cur_pairs.copy()
                    best_size = cur_size
                return

            idx = choose_class(classes)
            gmask, gcnt, hmask, hcnt = classes[idx]

            v_bit = gmask & -gmask
            v = v_bit.bit_length() - 1

            bits = hmask
            while bits:
                w_bit = bits & -bits
                w = w_bit.bit_length() - 1
                new_classes, new_bound = refine(classes, idx, v_bit, w_bit, v, w)
                if cur_size + 1 + new_bound > best_size:
                    cur_pairs.append((v, w))
                    search(new_classes, new_bound, cur_pairs)
                    cur_pairs.pop()

                    if best_size == n:
                        return
                bits ^= w_bit

            new_gcnt = gcnt - 1
            new_bound = remain_bound - (gcnt if gcnt < hcnt else hcnt)
            if new_gcnt:
                new_bound += new_gcnt if new_gcnt < hcnt else hcnt

            if cur_size + new_bound <= best_size:
                return

            if new_gcnt:
                new_classes = classes.copy()
                new_classes[idx] = (gmask ^ v_bit, new_gcnt, hmask, hcnt)
            else:
                new_classes = classes[:idx] + classes[idx + 1 :]

            search(new_classes, new_bound, cur_pairs)

        root_classes = [(all_g, n, all_h, m)]

        seed = greedy_seed(root_classes)
        best_pairs = seed
        best_size = len(seed)

        search(root_classes, n, [])

        result = [(row_perm[i], col_perm[p]) for i, p in best_pairs]
        if swapped:
            result = [(p, i) for i, p in result]
        result.sort()
        return result