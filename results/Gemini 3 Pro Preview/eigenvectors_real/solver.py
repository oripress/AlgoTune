import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Use numpy.linalg.eigh as it is robust and used by reference.
        # We optimize the post-processing.
        vals, vecs = np.linalg.eigh(problem)
        
        # vals are ascending. vecs columns are eigenvectors.
        # We want descending.
        
        # Reverse vals
        vals = vals[::-1]
        
        # vecs.T has eigenvectors as rows, corresponding to ascending vals.
        # We want rows corresponding to descending vals.
        # So we reverse the rows of vecs.T.
        # vecs.T is a view, slicing it is fast.
        vecs_t = vecs.T[::-1]
        
        return vals.tolist(), vecs_t.tolist()