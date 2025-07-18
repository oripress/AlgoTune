import numpy as np
from scipy import sparse
from solver import Solver

# Create a small test sparse matrix
n = 10
data = [1.0, 2.0, 3.0, 4.0]
row = [0, 1, 2, 3]
col = [1, 2, 3, 4]
A = sparse.csc_matrix((data, (row, col)), shape=(n, n))

# Create problem dict
problem = {"matrix": A}

# Test solver
solver = Solver()
result = solver.solve(problem)
print("Result type:", type(result))
print("Result shape:", result.shape)
print("Result format:", result.format)