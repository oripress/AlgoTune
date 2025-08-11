import itertools
import numpy as np
from numba import njit

@njit
def select_sorted(sorted_indices, to_block, powers, num_nodes):
    N = sorted_indices.shape[0]
    blocked = np.zeros(N, dtype=np.bool_)
    selected = np.empty(N, dtype=np.int64)
    sel_count = 0
    n = powers.shape[0]
    coords = np.empty(n, dtype=np.int64)
    P = to_block.shape[0]
    for k in range(N):
        idx = sorted_indices[k]
        if not blocked[idx]:
            selected[sel_count] = idx
            sel_count += 1
            # compute coordinates of this index
            for j in range(n):
                coords[j] = (idx // powers[j]) % num_nodes
            # block neighbors in strong product
            for p in range(P):
                nn_idx = 0
                for j in range(n):
                    val = coords[j] + to_block[p, j]
                    if val < 0:
                        val += num_nodes
                    elif val >= num_nodes:
                        val -= num_nodes
                    nn_idx += val * powers[j]
                blocked[nn_idx] = True
    return selected[:sel_count]

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes, n = problem
        M = num_nodes - 2

        # Total number of vertices in the product graph
        total = num_nodes ** n

        # Enumerate all vertices as n-tuples
        children = np.array(
            list(itertools.product(range(num_nodes), repeat=n)),
            dtype=np.int32
        )

        # Precompute priority components
        if n > 0:
            # All m-vectors in product(range(1,n), repeat=n)
            values = np.array(
                list(itertools.product(range(1, n), repeat=n)),
                dtype=np.int32
            ) * 2  # multiply by 2 as per reference priority
            # Multipliers for converting tuples to integer weights
            multipliers = np.array(
                [num_nodes ** i for i in range(n - 1, -1, -1)],
                dtype=np.int32
            )

            # Compute priority scores for each vertex
            scores = np.empty(total, dtype=np.int64)
            for i in range(total):
                el = children[i]
                # clip values above num_nodes - 3
                el_clipped = np.minimum(el, num_nodes - 3)
                # weighted sum and modular reduction
                x = np.sum((1 + values + el_clipped) * multipliers, axis=1)
                scores[i] = np.sum(x % M)
        else:
            # trivial case n=0
            scores = np.zeros(total, dtype=np.int64)

        # Sort vertices by descending priority
        sorted_indices = np.argsort(-scores)

        # Precompute powers for index<->coordinate conversion
        powers = np.array(
            [num_nodes ** (n - 1 - j) for j in range(n)],
            dtype=np.int64
        )

        # All neighbor shifts in each coordinate for strong product adjacency
        to_block = np.array(
            list(itertools.product([-1, 0, 1], repeat=n)),
            dtype=np.int8
        )

        # Greedy selection via compiled routine
        selected_idxs = select_sorted(
            sorted_indices, to_block, powers, num_nodes
        )

        # Convert selected linear indices back to n-tuples
        solution = [tuple(children[int(idx)]) for idx in selected_idxs]
        return solution