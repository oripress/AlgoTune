import numpy as np
import scipy.sparse
import multiprocessing
from utils import parallel_dijkstra

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        try:
            # Construct CSR matrix
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"]
            )
            
            # Transpose graph for undirected traversal (incoming edges become outgoing)
            # Since the problem is undirected, we need to traverse both G and G.T
            graph_csc = graph_csr.tocsc()
            
            # Extract CSR components
            data = graph_csr.data.astype(np.float64)
            indices = graph_csr.indices.astype(np.int32)
            indptr = graph_csr.indptr.astype(np.int32)
            
            # Extract CSC components (which are CSR of the transpose)
            data_T = graph_csc.data.astype(np.float64)
            indices_T = graph_csc.indices.astype(np.int32)
            indptr_T = graph_csc.indptr.astype(np.int32)
            
            n_nodes = problem["shape"][0]
            n_threads = multiprocessing.cpu_count()
            
            # Run parallel Dijkstra
            dist_matrix_list = parallel_dijkstra(
                n_nodes,
                data, indices, indptr,
                data_T, indices_T, indptr_T,
                n_threads
            )
            
            return {"distance_matrix": dist_matrix_list}
            
        except Exception:
            return {"distance_matrix": []}