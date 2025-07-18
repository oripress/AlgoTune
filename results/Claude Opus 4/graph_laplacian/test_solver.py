import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

# Test case from the problem description
problem = {
    "data": [1.0, 1.0, 2.0, 2.0],
    "indices": [1, 0, 2, 1],
    "indptr": [0, 1, 3, 4],
    "shape": [3, 3],
    "normed": False
}

# Create the graph
graph_csr = scipy.sparse.csr_matrix(
    (problem["data"], problem["indices"], problem["indptr"]), 
    shape=problem["shape"]
)

print("Original graph:")
print(graph_csr.toarray())

# Compute Laplacian using scipy
L = scipy.sparse.csgraph.laplacian(graph_csr, normed=problem["normed"])
L_csr = L.tocsr()
L_csr.eliminate_zeros()

print("\nLaplacian (scipy):")
print(L_csr.toarray())
print(f"data: {L_csr.data}")
print(f"indices: {L_csr.indices}")
print(f"indptr: {L_csr.indptr}")
print(f"data type: {type(L_csr.data)}")