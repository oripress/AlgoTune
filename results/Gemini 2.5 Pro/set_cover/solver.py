from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> list[int]:
        """
        Solves the set cover problem using a multi-stage approach:
        1. A fast, iterative presolver to handle essential subsets.
        2. A greedy heuristic on the smaller, residual problem for warm-starting.
        3. The OR-Tools CP-SAT solver to find an optimal solution for the
           residual problem within a time limit.
        """
        from collections import defaultdict

        num_subsets = len(problem)
        if num_subsets == 0:
            return []

        # Use sets for performance
        subsets = [set(s) for s in problem]
        
        universe = set()
        for s in subsets:
            universe.update(s)
        if not universe:
            return []

        final_solution_indices = set()  # Stores 0-indexed original indices
        uncovered_elements = universe.copy()
        available_subset_indices = set(range(num_subsets))

        # --- 1. Presolve Loop for Essential Subsets ---
        while True:
            if not uncovered_elements:
                break

            # Find elements covered by only one available subset
            element_to_available_subsets = defaultdict(list)
            for i in available_subset_indices:
                for element in subsets[i]:
                    if element in uncovered_elements:
                        element_to_available_subsets[element].append(i)

            essential_subset_indices = set()
            for containing_subsets in element_to_available_subsets.values():
                if len(containing_subsets) == 1:
                    essential_subset_indices.add(containing_subsets[0])
            
            if not essential_subset_indices:
                break  # No more essential subsets, move to main solver

            # Process essential subsets
            for i in essential_subset_indices:
                if i in available_subset_indices:
                    final_solution_indices.add(i)
                    uncovered_elements.difference_update(subsets[i])
                    available_subset_indices.remove(i)

        # --- 2. Solve Residual Problem ---
        if not uncovered_elements:
            return sorted([i + 1 for i in final_solution_indices])

        # Create the subproblem from remaining sets and elements
        subproblem_orig_indices = sorted(list(available_subset_indices))
        map_sub_to_orig = {sub_idx: orig_idx for sub_idx, orig_idx in enumerate(subproblem_orig_indices)}
        
        subproblem_sets = [subsets[orig_idx].intersection(uncovered_elements) for orig_idx in subproblem_orig_indices]

        # --- 2a. Greedy on Subproblem for Warm Start ---
        sub_uncovered_greedy = uncovered_elements.copy()
        sub_greedy_solution_indices = []
        sub_available_greedy = set(range(len(subproblem_sets)))

        while sub_uncovered_greedy and sub_available_greedy:
            best_sub_index = max(
                sub_available_greedy,
                key=lambda i: len(subproblem_sets[i].intersection(sub_uncovered_greedy))
            )
            newly_covered = subproblem_sets[best_sub_index].intersection(sub_uncovered_greedy)
            if not newly_covered: break
            
            sub_greedy_solution_indices.append(best_sub_index)
            sub_uncovered_greedy.difference_update(newly_covered)
            sub_available_greedy.remove(best_sub_index)

        # --- 2b. CP-SAT on Subproblem ---
        model = cp_model.CpModel()
        num_subproblem_subsets = len(subproblem_sets)
        x = [model.NewBoolVar(f'x_{i}') for i in range(num_subproblem_subsets)]

        sub_element_to_subsets = defaultdict(list)
        for i, s in enumerate(subproblem_sets):
            for element in s:
                sub_element_to_subsets[element].append(i)
        for subset_indices in sub_element_to_subsets.values():
            model.AddBoolOr([x[i] for i in subset_indices])

        model.Minimize(sum(x))

        greedy_set = set(sub_greedy_solution_indices)
        for i in range(num_subproblem_subsets):
            model.AddHint(x[i], 1 if i in greedy_set else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 4.5
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        sub_solution_indices = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i in range(num_subproblem_subsets):
                if solver.Value(x[i]) == 1:
                    sub_solution_indices.append(i)
        else: # Fallback to greedy if solver fails
            if not sub_uncovered_greedy: # Check if greedy found a full cover
                sub_solution_indices = sub_greedy_solution_indices

        # --- 3. Combine Solutions ---
        for sub_idx in sub_solution_indices:
            final_solution_indices.add(map_sub_to_orig[sub_idx])

        return sorted([i + 1 for i in final_solution_indices])