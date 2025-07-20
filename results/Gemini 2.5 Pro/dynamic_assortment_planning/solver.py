from typing import Any
from ortools.graph.python import min_cost_flow

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solves the Dynamic Assortment Planning problem using a min-cost max-flow
        formulation.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        # Scale factor to convert float costs to integers for the ortools solver.
        # A large factor is used to maintain precision.
        SCALE_FACTOR = 10**7

        mcf = min_cost_flow.SimpleMinCostFlow()

        # --- Node Definition ---
        # Node indices are defined as follows:
        # source: 0
        # time_nodes: 1 to T
        # product_nodes: T+1 to T+N
        # dummy_node (for 'offer nothing'): T+N+1
        # sink: T+N+2
        source = 0
        time_nodes_start = 1
        product_nodes_start = T + 1
        dummy_node = T + N + 1
        sink = T + N + 2

        # --- Arc Definition ---

        # 1. Arcs from source to time nodes
        # Each time period can be assigned at most one offer.
        for t in range(T):
            mcf.add_arc_with_capacity_and_unit_cost(
                source, time_nodes_start + t, capacity=1, unit_cost=0
            )

        # 2. Arcs from product nodes to sink
        # Each product has a limited capacity.
        for i in range(N):
            if capacities[i] > 0:
                mcf.add_arc_with_capacity_and_unit_cost(
                    product_nodes_start + i, sink, capacity=int(capacities[i]), unit_cost=0
                )
        
        # 3. Arc from dummy node to sink
        # This represents the "offer nothing" option, which has no capacity limit
        # (up to T times) and zero cost.
        mcf.add_arc_with_capacity_and_unit_cost(
            dummy_node, sink, capacity=T, unit_cost=0
        )

        # 4. Arcs from time nodes to product nodes
        # The cost is the negative expected revenue.
        for t in range(T):
            for i in range(N):
                if prices[i] > 0 and probs[t][i] > 0 and capacities[i] > 0:
                    cost = -int(prices[i] * probs[t][i] * SCALE_FACTOR)
                    mcf.add_arc_with_capacity_and_unit_cost(
                        time_nodes_start + t, product_nodes_start + i, capacity=1, unit_cost=cost
                    )
        
        # 5. Arcs from time nodes to dummy node
        # Represents choosing to offer nothing in a period. Cost is 0.
        for t in range(T):
            mcf.add_arc_with_capacity_and_unit_cost(
                time_nodes_start + t, dummy_node, capacity=1, unit_cost=0
            )

        # --- Supply and Demand ---
        # We need to satisfy the demand for each time period, which is 1.
        # Total flow is T.
        mcf.set_node_supply(source, T)
        mcf.set_node_supply(sink, -T)

        # --- Solve ---
        status = mcf.solve()

        if status != mcf.OPTIMAL:
            # This should not happen in a feasible problem.
            # Fallback to offering nothing.
            return [-1] * T

        # --- Reconstruct Solution ---
        offer = [-1] * T
        for arc in range(mcf.num_arcs()):
            # Check for flow on arcs from time nodes to product nodes.
            if mcf.flow(arc) > 0:
                tail = mcf.tail(arc)
                head = mcf.head(arc)
                
                # Is it an arc from a time_node?
                if time_nodes_start <= tail < product_nodes_start:
                    # Is it to a product_node?
                    if product_nodes_start <= head < dummy_node:
                        t = tail - time_nodes_start
                        i = head - product_nodes_start
                        offer[t] = i
        
        return offer