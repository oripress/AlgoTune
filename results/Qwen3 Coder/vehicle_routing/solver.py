import random
from typing import Any
import heapq

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the VRP problem using a simple greedy approach.
        
        :param problem: Dict with "D", "K", and "depot".
        :return: A list of K routes, each a list of nodes starting and ending at the depot.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        customers = [i for i in range(n) if i != depot]
        
        # Handle trivial cases
        if n == 1:
            return [[depot, depot]] * K
            
        if K >= len(customers):  # More vehicles than customers
            routes = []
            # Assign each customer to a separate vehicle
            for i, customer in enumerate(customers):
                routes.append([depot, customer, depot])
            # Fill remaining vehicles with empty routes
            while len(routes) < K:
                routes.append([depot, depot])
            return routes
            
        # Simple round-robin assignment
        routes = [[] for _ in range(K)]
        for i, customer in enumerate(customers):
            routes[i % K].append(customer)
        
        # Add depot at start and end of each route
        for route in routes:
            route.insert(0, depot)
            route.append(depot)
            
        return routes