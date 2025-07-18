from typing import Any, List
from ortools.constraint_solver import pywrapcp

class Solver:
    """
    A solver for the Capacitated Vehicle Routing Problem (CVRP) using
    Google's OR-Tools, specifically its specialized routing library.
    """
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solves the VRP/CVRP using the ortools routing library. This version
        uses a fast, deterministic first solution strategy combined with a
        time-limited deterministic local search to find a valid solution
        quickly and consistently.
        """
        # Extract problem data
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(n, K, depot)
        routing = pywrapcp.RoutingModel(manager)

        # --- Callbacks ---
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return D[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # --- Constraints ---
        # Force all vehicles to be used by ensuring they visit at least one node.
        for vehicle_id in range(K):
            routing.solver().Add(
                routing.NextVar(routing.Start(vehicle_id)) != routing.End(vehicle_id)
            )

        # Optional Capacity Constraints
        if 'demands' in problem and 'vehicle_capacities' in problem:
            demands = problem["demands"]
            vehicle_capacities_data = problem["vehicle_capacities"]

            vehicle_capacities: List[int]
            if isinstance(vehicle_capacities_data, list):
                vehicle_capacities = vehicle_capacities_data
            else:
                vehicle_capacities = [int(vehicle_capacities_data)] * K

            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return demands[from_node]

            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index, 0, vehicle_capacities, True, 'Capacity'
            )

        # --- Search Parameters ---
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # Use a fast, deterministic first solution strategy.
        search_parameters.first_solution_strategy = 3 # PATH_CHEAPEST_ARC
        
        # Use a deterministic local search to fix/improve the initial solution.
        search_parameters.local_search_metaheuristic = 2 # GREEDY_DESCENT

        # A small time limit allows local search to find a feasible solution
        # without causing timeouts. The search remains deterministic.
        search_parameters.time_limit.FromMilliseconds(20)

        # --- Solve ---
        solution = routing.SolveWithParameters(search_parameters)

        # --- Reconstruct Solution ---
        if not solution:
            # This can happen if the problem is infeasible with the given constraints
            # and short time limit. Return an empty list as a failure signal.
            return []

        final_routes = []
        for vehicle_id in range(K):
            index = routing.Start(vehicle_id)
            route = []
            while True:
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                if routing.IsEnd(index):
                    break
                index = solution.Value(routing.NextVar(index))
            final_routes.append(route)

        return final_routes