import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=np.float32)
        k = int(problem["k"])
        n, d = X.shape
        
        if k >= n:
            return list(range(n))
        
        if k == 1:
            return [0] * n

        # Subsample for training if dataset is large
        max_train_pts = min(n, max(k * 20, 2000))
        
        if n > max_train_pts:
            step = n // max_train_pts
            X_train = np.ascontiguousarray(X[::step][:max_train_pts])
        else:
            X_train = X
        
        # 4 iterations - good balance
        niter = 4
        
        kmeans = faiss.Kmeans(d, k, niter=niter, nredo=1, verbose=False, seed=42)
        kmeans.train(X_train)
        
        _, labels = kmeans.index.search(X, 1)
        return labels.ravel().tolist()