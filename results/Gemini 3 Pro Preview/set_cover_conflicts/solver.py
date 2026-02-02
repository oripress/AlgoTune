import highspy
import numpy as np
from numba import njit
import itertools

@njit
def build_csr_transpose(sets_flat, sets_indptr, n, num_sets):
    # Count objects
    obj_counts = np.zeros(n, dtype=np.int32)
    for obj in sets_flat:
        obj_counts[obj] += 1
        
    # Build row pointers
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    current = 0
    for i in range(n):
        row_ptr[i] = current
        current += obj_counts[i]
    row_ptr[n] = current
    
    # Fill indices
    indices = np.empty(len(sets_flat), dtype=np.int32)
    current_pos = np.copy(row_ptr)
    
    for s_idx in range(num_sets):
        start = sets_indptr[s_idx]
        end = sets_indptr[s_idx+1]
        for k in range(start, end):
            obj = sets_flat[k]
            pos = current_pos[obj]
            indices[pos] = s_idx
            current_pos[obj] += 1
            
    return row_ptr, indices

class Solver:
    def __init__(self):
        # Pre-compile numba function
        dummy_flat = np.array([0], dtype=np.int32)
        dummy_indptr = np.array([0, 1], dtype=np.int32)
        build_csr_transpose(dummy_flat, dummy_indptr, 1, 1)
        
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("presolve", "on")
        self.h.setOptionValue("threads", 1)

    def solve(self, problem, **kwargs):
        self.h.clearModel()
        n, sets, conflicts = problem
        num_sets = len(sets)
        
        # Flatten sets
        sets_lengths = [len(s) for s in sets]
        total_len = sum(sets_lengths)
        
        sets_indptr = np.zeros(num_sets + 1, dtype=np.int32)
        np.cumsum(sets_lengths, out=sets_indptr[1:])
        
        sets_flat = np.fromiter(itertools.chain.from_iterable(sets), dtype=np.int32, count=total_len)
        
        # Build transpose CSR
        row_ptr, indices = build_csr_transpose(sets_flat, sets_indptr, n, num_sets)
        data = np.ones(len(indices), dtype=np.float64)
        
        # Add Variables
        costs = np.ones(num_sets, dtype=np.float64)
        col_lower = np.zeros(num_sets, dtype=np.float64)
        col_upper = np.ones(num_sets, dtype=np.float64)
        
        self.h.addCols(num_sets, costs, col_lower, col_upper, 0, np.zeros(num_sets + 1, dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
        
        for i in range(num_sets):
            self.h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
            
        self.h.setOptionValue("mip_abs_gap", 0.99)
        self.h.setOptionValue("presolve", "on")
        
        # Add Cover Constraints
        row_lower = np.ones(n, dtype=np.float64)
        row_upper = np.full(n, np.inf, dtype=np.float64)
        self.h.addRows(n, row_lower, row_upper, len(data), row_ptr, indices, data)
        
        # Add Conflict Constraints
        if conflicts:
            num_conflicts = len(conflicts)
            c_lengths = [len(c) for c in conflicts]
            total_c_len = sum(c_lengths)
            
            c_indptr = np.zeros(num_conflicts + 1, dtype=np.int32)
            np.cumsum(c_lengths, out=c_indptr[1:])
            
            c_flat = np.fromiter(itertools.chain.from_iterable(conflicts), dtype=np.int32, count=total_c_len)
            
            c_data = np.ones(len(c_flat), dtype=np.float64)
            c_lower = np.full(num_conflicts, -np.inf, dtype=np.float64)
            c_upper = np.ones(num_conflicts, dtype=np.float64)
            
            self.h.addRows(num_conflicts, c_lower, c_upper, len(c_data), c_indptr, c_flat, c_data)
            
        self.h.run()
        
        sol = self.h.getSolution()
        col_vals = np.array(sol.col_value)
        return np.where(col_vals > 0.5)[0].tolist()