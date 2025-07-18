from typing import Any
import numpy as np

try:
    import faiss
    HAS_FAISS = True
    # set to use all available threads
    try:
        faiss.omp_set_num_threads(0)
    except:
        pass
except ImportError:
    HAS_FAISS = False

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[int]:
        # parse inputs
        X = problem.get("X", [])
        try:
            k = int(problem.get("k", 0))
        except Exception:
            return []
        if k <= 0:
            return []
        # build data array
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            return []
        n, d = X_arr.shape
        if n == 0:
            return []
        k_eff = min(k, n)
        # prepare sample for training
        sample_size = min(n, max(10000, k_eff * 20))
        if sample_size < n:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=sample_size, replace=False)
            X_train = X_arr[idx]
        else:
            X_train = X_arr
        # FAISS-based clustering on sample and assign
        if HAS_FAISS:
            # train on sample
            kmeans = faiss.Kmeans(d, k_eff, niter=10, verbose=False, seed=0)
            kmeans.train(np.ascontiguousarray(X_train))
            centroids = kmeans.centroids.reshape(k_eff, d)
            # assign all points
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            _, labels = index.search(np.ascontiguousarray(X_arr), 1)
            return labels.ravel().tolist()
        # fallback: pure NumPy k-means++ init on sample + one Lloyd update
        rng = np.random.default_rng(0)
        m = X_train.shape[0]
        centers = np.empty((k_eff, d), dtype=np.float32)
        # init first center
        j0 = rng.integers(m)
        centers[0] = X_train[j0]
        # init distances
        D2 = np.sum((X_train - centers[0]) ** 2, axis=1)
        for j in range(1, k_eff):
            total = D2.sum()
            if total > 0.0:
                probs = D2 / total
                j_idx = rng.choice(m, p=probs)
            else:
                j_idx = rng.integers(m)
            centers[j] = X_train[j_idx]
            dist = np.sum((X_train - centers[j]) ** 2, axis=1)
            D2 = np.minimum(D2, dist)
        # one Lloyd iteration on sample
        d_train = (np.einsum('ij,ij->i', X_train, X_train)[:, None]
                   + np.einsum('ij,ij->i', centers, centers)[None, :]
                   - 2 * X_train.dot(centers.T))
        labels_tr = np.argmin(d_train, axis=1)
        new_centers = np.zeros_like(centers)
        np.add.at(new_centers, labels_tr, X_train)
        counts = np.bincount(labels_tr, minlength=k_eff)
        nonzero = counts > 0
        new_centers[nonzero] /= counts[nonzero, None]
        # reinit empty clusters
        for j in range(k_eff):
            if counts[j] == 0:
                new_centers[j] = X_train[rng.integers(m)]
        centers = new_centers
        # assign all data
        X_norm = np.einsum('ij,ij->i', X_arr, X_arr)
        C_norm = np.einsum('ij,ij->i', centers, centers)
        D = X_norm[:, None] + C_norm[None, :] - 2 * X_arr.dot(centers.T)
        labels = np.argmin(D, axis=1)
        return labels.tolist()