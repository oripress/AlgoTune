from typing import Any, List, Tuple
import itertools

class Solver:
    def solve(self, problem: Tuple[int, int], **kwargs) -> List[Tuple[int, ...]]:
        """
        Construct an optimal independent set for the strong product of a cycle.

        For a cycle with num_nodes nodes, a maximum independent set can be taken as
        the vertices {0,2,4,...} of size floor(num_nodes/2). For the n-th strong
        product, the Cartesian product of that base independent set across n
        coordinates is a valid independent set of size (floor(num_nodes/2))**n.

        This function returns the list of tuples representing that product.

        Args:
            problem: Tuple (num_nodes, n)

        Returns:
            List of n-tuples (as tuples of ints)
        """
        num_nodes, n = problem

        # Handle degenerate cases
        if n <= 0:
            # The 0-th power yields a single empty-tuple vertex
            return [tuple()]

        if num_nodes <= 0:
            return []

        # Build a maximum independent set for the base cycle:
        # take vertices 0,2,4,... up to floor(num_nodes/2) elements.
        k = num_nodes // 2
        base = [(2 * i) % num_nodes for i in range(k)]

        # If base is empty (e.g., num_nodes == 1), fall back to [0]
        if not base:
            base = [0]

        # Return the Cartesian product (as list of tuples)
        return [tuple(t) for t in itertools.product(base, repeat=n)]