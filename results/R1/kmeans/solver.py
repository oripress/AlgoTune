import numpy as np
import os
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class Solver:
    def solve(self, problem, **kwargs) -> list[int]:
        try:
            # Convert to float32 for efficiency
            X = np.array(problem["X"], dtype=np.float32)
            n = X.shape[0]
            k = problem["k"]
            
            # Handle trivial cases
            if n == 0:
                return []
            if k == 1:
                return [0] * n
            if k >= n:
                return list(range(n))
            
            # Use FAISS for all datasets if available
            if FAISS_AVAILABLE:
                d = X.shape[1]
                X = np.ascontiguousarray(X)
                
                # Set number of threads for parallel processing
                threads = min(8, os.cpu_count() or 1)
                if n < 1000:
                    threads = 1  # Single thread for small datasets
                faiss.omp_set_num_threads(threads)
                
                # Configure FAISS with optimized parameters
                kmeans = faiss.Kmeans(
                    d, 
                    k, 
                    niter=5,  # Very few iterations
                    verbose=False,
                    gpu=False,
                    min_points_per_centroid=1,
                    max_points_per_centroid=10000000,
                    seed=42,
                    nredo=1,
                    update_index=True
                )
                kmeans.train(X)
                
                # Get cluster assignments
                _, labels = kmeans.index.search(X, 1)
                return labels.flatten().astype(int).tolist()
            
            # Fallback for environments without FAISS
            else:
                # For very small datasets, use optimized k-means
                if n <= 100:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(
                        n_clusters=k,
                        init="random",
                        n_init=1,
                        max_iter=10,
                        algorithm="lloyd",
                        random_state=42
                    )
                else:
                    from sklearn.cluster import MiniBatchKMeans
                    batch_size = min(1024, max(256, n // 10))
                    kmeans = MiniBatchKMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init=1,
                        max_iter=5,
                        batch_size=batch_size,
                        compute_labels=True,
                        tol=1e-3,
                        max_no_improvement=2,
                        n_jobs=-1
                    )
                kmeans.fit(X)
                return kmeans.labels_.tolist()
            
        except Exception as e:
            # Fallback to reference implementation
            from sklearn.cluster import KMeans as RefKMeans
            kmeans = RefKMeans(n_clusters=problem["k"]).fit(problem["X"])
            return kmeans.labels_.tolist()