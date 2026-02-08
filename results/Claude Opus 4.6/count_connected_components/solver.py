from uf_cython import count_components_cy

class Solver:
    def solve(self, problem, **kwargs):
        n = problem.get("num_nodes", 0)
        if n == 0:
            return {"number_connected_components": 0}
        
        edges = problem["edges"]
        
        if not edges:
            return {"number_connected_components": n}
        
        num_components = count_components_cy(n, edges)
        
        return {"number_connected_components": num_components}