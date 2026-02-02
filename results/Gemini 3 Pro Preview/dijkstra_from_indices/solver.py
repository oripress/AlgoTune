import numpy as np
import scipy.sparse
import os
from dijkstra import dijkstra_cython

class Solver:
    def solve(self, problem, **kwargs):
        try:
            # Extract graph components
            data = np.array(problem["data"], dtype=np.float64)
            indices = np.array(problem["indices"], dtype=np.int32)
            indptr = np.array(problem["indptr"], dtype=np.int32)
            num_nodes = problem["shape"][0]
            source_indices = np.array(problem["source_indices"], dtype=np.int64)
            
            # Determine number of threads
            num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
            
            # Run Cython Dijkstra
            distances = dijkstra_cython(
                data,
                indices,
                indptr,
                num_nodes,
                source_indices,
                num_threads
            )
            
            # Handle min_only
            min_only = getattr(self, 'min_only', False)
            if min_only:
                if distances.shape[0] > 0:
                    distances = np.min(distances, axis=0)
                    dist_obj = distances.astype(object)
                    dist_obj[np.isinf(distances)] = None
                    dist_matrix_list = [dist_obj.tolist()]
                else:
                    dist_matrix_list = []
            else:
                dist_obj = distances.astype(object)
                dist_obj[np.isinf(distances)] = None
                dist_matrix_list = dist_obj.tolist()

            return {"distances": dist_matrix_list}
        except Exception as e:
            return {"distances": []}