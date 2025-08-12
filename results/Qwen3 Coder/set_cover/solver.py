class Solver:
    def solve(self, problem):
        """
        Solves the set cover problem using branch and bound.

        :param problem: A list of subsets (each subset is a list of integers).
        :return: A list of indices (1-indexed) of the selected subsets.
        """
        if not problem:
            return []

        # Calculate the universe (all elements that need to be covered)
        universe = set()
        for subset in problem:
            universe.update(subset)

        # If universe is empty, nothing to cover
        if not universe:
            return []

        # Convert problem to sets for faster operations
        sets = [set(subset) for subset in problem]
        
        # Use branch and bound for small problems, greedy for large ones
        if len(problem) <= 12:
            return self._branch_and_bound_solve(sets, universe)
        else:
            return self._greedy_solve(sets, universe)
    
    def _branch_and_bound_solve(self, sets, universe):
        """Branch and bound approach for finding optimal solution"""
        best_solution = None
        best_size = len(sets) + 1  # Upper bound
        
        def backtrack(selected, covered, index):
            nonlocal best_solution, best_size
            
            # Pruning: if current solution is already worse than best, stop
            if len(selected) >= best_size:
                return
            
            # If we've covered everything, we have a solution
            if covered >= universe:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = selected[:]
                return
            
            # If we've considered all sets, stop
            if index >= len(sets):
                return
                
            # Pruning: if even selecting all remaining sets won't improve best solution, stop
            remaining_indices = len(sets) - index
            if len(selected) + remaining_indices >= best_size:
                return
            
            # Option 1: Include current set
            new_covered = covered | sets[index]
            selected.append(index)
            backtrack(selected, new_covered, index + 1)
            selected.pop()
            
            # Option 2: Don't include current set (only if we still have hope of a better solution)
            backtrack(selected, covered, index + 1)
        
        backtrack([], set(), 0)
        
        if best_solution is not None:
            # Convert to 1-indexed
            return [i + 1 for i in best_solution]
        else:
            # Fallback to greedy
            return self._greedy_solve(sets, universe)
    
    def _greedy_solve(self, sets, universe):
        """Greedy approximation for larger problems"""
        # Convert sets back to list format if needed
        if isinstance(sets[0], set):
            problem = [list(s) for s in sets]
        else:
            problem = sets
            
        # Greedy algorithm for set cover
        uncovered = set(universe)
        solution = []  # 0-indexed indices

        while uncovered:
            best_count = 0
            best_index = -1

            # Try each available set
            for i in range(len(problem)):
                # Count how many uncovered elements this set covers
                count = len(set(problem[i]) & uncovered)  # Intersection
                if count > best_count:
                    best_count = count
                    best_index = i

            # If we can't cover more elements, we're done
            if best_index == -1 or best_count == 0:
                break

            # Add the best set to our solution
            solution.append(best_index)
            
            # Update uncovered elements
            newly_covered = set(problem[best_index]) & uncovered
            uncovered -= newly_covered
            
        # Convert to 1-indexed
        return [i + 1 for i in solution]