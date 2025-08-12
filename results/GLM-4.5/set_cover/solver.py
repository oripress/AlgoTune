import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Your implementation goes here."""
        if not problem:
            return []
        
        # Precompute universe and element coverage
        universe = set()
        for subset in problem:
            universe.update(subset)
        n = len(universe)
        m = len(problem)
        
        # If there's only one set, return it
        if m == 1:
            return [1]
        
        # Use greedy algorithm for all instances
        return self._greedy_set_cover(problem, universe)
    
    def _greedy_set_cover(self, problem, universe):
        """Greedy approximation algorithm for set cover."""
        if not problem:
            return []
        
        covered = set()
        selected = []
        remaining_sets = list(range(len(problem)))
        
        while covered != universe and remaining_sets:
            # Find the set that covers the most uncovered elements
            best_set = None
            best_coverage = 0
            
            for i in remaining_sets:
                new_elements = len(set(problem[i]) - covered)
                if new_elements > best_coverage:
                    best_coverage = new_elements
                    best_set = i
            
            if best_set is None:
                break
            
            # Add the best set
            selected.append(best_set + 1)  # 1-indexed
            covered.update(problem[best_set])
            remaining_sets.remove(best_set)
        
        return selected
    
    def solve_original(self, problem, **kwargs):
        """Original implementation for comparison."""
        if not problem:
            return []
        
        # Precompute universe and element coverage
        universe = set()
        for subset in problem:
            universe.update(subset)
        n = len(universe)
        m = len(problem)
        
        # If there's only one set, return it
        if m == 1:
            return [1]
        
        # Use greedy algorithm for all instances
        return self._greedy_set_cover(problem, universe)