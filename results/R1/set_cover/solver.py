from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from collections import defaultdict, deque

class Solver:
    def solve(self, problem, **kwargs):
        if not problem:
            return []
        
        # Preprocessing: handle sets that must be included
        active_sets = [set(subset) for subset in problem]
        active_indices = list(range(len(problem)))
        selected_pre = []  # Stores indices of sets we must include
        
        # Build element frequency and element->set mapping
        freq = defaultdict(int)
        element_sets = defaultdict(list)
        for i, s in enumerate(active_sets):
            for e in s:
                freq[e] += 1
                element_sets[e].append(i)
        
        # Queue of elements with frequency 1
        queue = deque(e for e, count in freq.items() if count == 1)
        
        # Process unique elements
        while queue:
            e = queue.popleft()
            if freq[e] != 1:  # Skip if frequency changed
                continue
                
            # Find the set containing this element
            set_idx = element_sets[e][0]
            s = active_sets[set_idx]
            selected_pre.append(active_indices[set_idx])
            
            # Remove set and update frequency
            covered = s
            for x in covered:
                freq[x] -= 1
                # Remove this set from element_sets[x]
                element_sets[x] = [i for i in element_sets[x] if i != set_idx]
                if freq[x] == 1:
                    queue.append(x)
            
            # Remove set from active structures
            del active_sets[set_idx]
            del active_indices[set_idx]
            # Remove element from freq and element_sets
            del freq[e]
            del element_sets[e]
        
        # Compute universe covered by forced sets
        covered_by_forced = set()
        for idx in selected_pre:
            covered_by_forced.update(problem[idx])
        
        # Remove covered elements from active sets
        active_sets = [s - covered_by_forced for s in active_sets]
        
        # Remove empty sets
        non_empty_indices = [i for i, s in enumerate(active_sets) if s]
        active_sets = [active_sets[i] for i in non_empty_indices]
        active_indices = [active_indices[i] for i in non_empty_indices]
        
        # Remove dominated sets: if set A is a subset of set B (and A != B), then set A is dominated by set B and can be removed.
        # Sort sets by size (ascending) for more efficient subset checks
        active_sets, active_indices = zip(*sorted(zip(active_sets, active_indices), 
                                                key=lambda x: len(x[0])))
        active_sets = list(active_sets)
        active_indices = list(active_indices)
        
        non_dominated_sets = []
        non_dominated_indices = []
        dominated = [False] * len(active_sets)
        
        for i in range(len(active_sets)):
            if dominated[i]:
                continue
            for j in range(i+1, len(active_sets)):
                if dominated[j]:
                    continue
                # Only check larger sets (smaller sets can't contain larger ones)
                if len(active_sets[j]) > len(active_sets[i]) and active_sets[i].issubset(active_sets[j]):
                    dominated[i] = True
                    break
                # Also check if j is subset of i (could happen with same size)
                elif active_sets[j].issubset(active_sets[i]):
                    dominated[j] = True
            if not dominated[i]:
                non_dominated_sets.append(active_sets[i])
                non_dominated_indices.append(active_indices[i])
        
        active_sets = non_dominated_sets
        active_indices = non_dominated_indices

        # If no active sets left, just return forced selections
        if not active_sets:
            return [idx + 1 for idx in selected_pre]
        # If no active sets left, just return forced selections
        if not active_sets:
            return [idx + 1 for idx in selected_pre]
        
        # If no active sets left, just return forced selections
        if not active_sets:
            return [idx + 1 for idx in selected_pre]
        
        # Get remaining universe to cover
        full_universe = set()
        for s in problem:
            full_universe.update(s)
        remaining_universe = full_universe - covered_by_forced
        
        # Use optimized MaxSAT solver with preprocessing
        m = len(active_sets)
        wcnf = WCNF()
        
        # Hard constraints: coverage for remaining universe
        for e in remaining_universe:
            covering_sets = []
            for i, subset in enumerate(active_sets):
                if e in subset:
                    covering_sets.append(i + 1)
            wcnf.append(covering_sets)
        
        # Soft constraints: minimize set count
        for i in range(1, m + 1):
            wcnf.append([-i], weight=1)
        
        # Solve with RC2
        solver = RC2(wcnf)
        model = solver.compute()
        
        if model is None:
            return [idx + 1 for idx in selected_pre]
        
        # Extract solution for reduced problem
        reduced_solution = [x for x in model if 1 <= x <= m]
        
        # Map back to original indices
        solution = selected_pre + [active_indices[x - 1] for x in reduced_solution]
        return [idx + 1 for idx in solution]