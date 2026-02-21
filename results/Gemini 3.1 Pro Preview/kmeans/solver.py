import numpy as np
import faiss

class Solver:
    def solve(self, problem: dict, **kwargs) -> list[int]:
        X = problem["X"]
        if type(X) is list:
            X = np.array(X, dtype=np.float32)
        else:
            X = np.ascontiguousarray(X, dtype=np.float32)
            
        k = problem["k"]
        n = X.shape[0]
        
        if k <= 1 or n <= 1:
            return [0] * n
            
        if k >= n:
            return list(range(n))
            
        d = X.shape[1]
        clus = faiss.Clustering(d, k)
        clus.niter = 3
        clus.nredo = 1
        clus.max_points_per_centroid = 64
        clus.min_points_per_centroid = 1
        clus.seed = 0
        clus.spherical = False
        
        index = faiss.IndexFlatL2(d)
        clus.train(X, index)
        _, I = index.search(X, 1)
        return I.reshape(-1).tolist()