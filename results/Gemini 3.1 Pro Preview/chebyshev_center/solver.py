import numpy as np
import highspy

class Solver:
    def __init__(self):
        self.h = highspy.Highs()
        self.h.setOptionValue('output_flag', False)
        self.h.setOptionValue('presolve', 'off')
        self.h.setOptionValue('solver', 'simplex')
        self.h.setOptionValue('simplex_strategy', 1)
        self.cache = {}
        
    def solve(self, problem: dict, **kwargs) -> dict:
        a = np.array(problem["a"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        m, n = a.shape
        
        if (m, n) not in self.cache:
            col_cost = np.zeros(n + 1, dtype=np.float64)
            col_cost[-1] = -1.0
            
            col_lower = np.full(n + 1, -np.inf, dtype=np.float64)
            col_lower[-1] = 0.0
            col_upper = np.full(n + 1, np.inf, dtype=np.float64)
            
            row_lower = np.full(m, -np.inf, dtype=np.float64)
            
            start = np.arange(0, (n + 2) * m, m, dtype=np.int32)
            index = np.tile(np.arange(m, dtype=np.int32), n + 1)
            
            lp = highspy.HighsLp()
            lp.num_col_ = n + 1
            lp.num_row_ = m
            lp.col_cost_ = col_cost
            lp.col_lower_ = col_lower
            lp.col_upper_ = col_upper
            lp.row_lower_ = row_lower
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_ = start
            lp.a_matrix_.index_ = index
            
            self.cache[(m, n)] = lp
            
        lp = self.cache[(m, n)]
        
        norms = np.sqrt(np.einsum('ij,ij->i', a, a))
        
        lp.row_upper_ = b
        lp.a_matrix_.value_ = np.concatenate((a.ravel(order='F'), norms))
        
        self.h.passModel(lp)
        self.h.run()
        
        return {"solution": self.h.getSolution().col_value[:n]}