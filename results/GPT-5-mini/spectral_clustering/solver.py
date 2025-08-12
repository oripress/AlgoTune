import numpy as np
from typing import Any, Dict

# Optional fast eigen solvers
try:
    from scipy.sparse.linalg import eigsh, LinearOperator  # type: ignore
    _HAS_SPARSE_EIG = True
except Exception:
    _HAS_SPARSE_EIG = False

try:
    from scipy.linalg import eigh as scipy_eigh  # type: ignore
    _HAS_SCIPY_EIGH = True
except Exception:
    _HAS_SCIPY_EIGH = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Spectral clustering solver.

        Steps:
         - Parse input
         - Build normalized affinity A = D^{-1/2} W D^{-1/2}
         - Compute top-k eigenvectors of A (largest eigenvalues)
         - Row-normalize embedding and run k-means (k-means++)
         - If k-means collapses to 1 cluster, iteratively split largest clusters
        """
        # Accept dict or JSON-like string
        if not isinstance(problem, dict):
            import json
            import ast

            if isinstance(problem, (bytes, bytearray)):
                try:
                    problem = problem.decode()
                except Exception:
                    raise ValueError("Problem must be a dict or JSON-serializable string")

            if isinstance(problem, str):
                try:
                    problem = json.loads(problem)
                except Exception:
                    try:
                        problem = ast.literal_eval(problem)
                    except Exception:
                        raise ValueError("Problem string could not be parsed into a dict")
            else:
                try:
                    problem = dict(problem)
                except Exception:
                    raise ValueError("Problem must be a dict or JSON-serializable string")

        sim = problem.get("similarity_matrix")
        requested_k = problem.get("n_clusters", 1)
        try:
            requested_k = int(requested_k)
        except Exception:
            requested_k = 1
        if requested_k < 1:
            requested_k = 1

        W = np.asarray(sim, dtype=float)
        if W.size == 0:
            return {"labels": [], "n_clusters": requested_k}
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("similarity_matrix must be a square 2D array")

        n = W.shape[0]
        # trivial/small cases
        if n == 0:
            return {"labels": [], "n_clusters": requested_k}
        if requested_k == 1:
            return {"labels": [0] * n, "n_clusters": 1}
        if requested_k >= n:
            # Each point in its own cluster if requested clusters >= samples
            return {"labels": list(range(n)), "n_clusters": requested_k}

        # Ensure symmetry and numeric stability
        W = 0.5 * (W + W.T)
        # Clip to reasonable range to avoid weird inputs
        # (similarities are between 0 and 1 per problem statement)
        W = np.clip(W, 0.0, np.inf)

        # Build normalized affinity A = D^{-1/2} W D^{-1/2}
        deg = np.sum(W, axis=1)
        deg_safe = deg.copy()
        # Replace zeros to avoid divide-by-zero; isolated nodes get zeroed rows in A
        deg_safe[deg_safe <= 0] = 1.0
        inv_sqrt = 1.0 / np.sqrt(deg_safe)
        D_outer = np.outer(inv_sqrt, inv_sqrt)
        A = W * D_outer

        k = int(requested_k)

        # Compute leading k eigenvectors of A (largest eigenvalues)
        try:
            vecs = None
            if _HAS_SPARSE_EIG and n > 500 and k < n - 1:
                # Iterative solver using LinearOperator avoids building sparse matrices
                from scipy.sparse.linalg import LinearOperator, eigsh  # type: ignore

                linop = LinearOperator((n, n), matvec=lambda v: A.dot(v))
                vals, vecs = eigsh(linop, k=k, which="LA", tol=1e-6)
                order = np.argsort(-vals)
                vecs = vecs[:, order]
            if vecs is None:
                if _HAS_SCIPY_EIGH and 0 < k < n // 2:
                    # Request only the subset of eigenpairs when it's efficient
                    vals, vecs = scipy_eigh(A, subset_by_index=(n - k, n - 1))
                else:
                    # Fall back to numpy dense eigen-decomposition and slice
                    vals_all, vecs_all = np.linalg.eigh(A)
                    vecs = vecs_all[:, -k:]
            # Ensure correct shape (n, k)
            if vecs.ndim == 1:
                vecs = vecs.reshape(-1, 1)
            if vecs.shape[1] < k:
                rng = np.random.default_rng(0)
                extra = rng.normal(scale=1e-8, size=(n, k - vecs.shape[1]))
                vecs = np.hstack([vecs, extra])
        except Exception:
            # Fallback embedding to avoid crashing; deterministic with fixed seed
            rng = np.random.default_rng(0)
            vecs = rng.normal(size=(n, k))

        # Row-normalize embedding (each sample becomes unit vector)
        row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 0, row_norms, 1.0)
        embedding = vecs / row_norms

        # Run k-means on the embedding
        labels = self._kmeans(embedding, k, random_state=0)

        # If k-means collapsed to a single cluster, try iterative splitting
        if requested_k > 1 and np.unique(labels).size == 1 and n >= 2:
            labels = self._iterative_split(embedding, requested_k, random_state=0)

        return {"labels": labels.tolist(), "n_clusters": requested_k}

    def _kmeans(self, X: np.ndarray, n_clusters: int, random_state: int = 0,
                max_iter: int = 300, tol: float = 1e-6) -> np.ndarray:
        """
        Lightweight k-means with k-means++ initialization.

        Vectorized updates:
         - works on float32 views of X for faster dot products
         - updates centers using np.add.at to avoid Python loops
         - uses multiple inits (n_init) and returns best inertia labeling
        """
        n, d = X.shape
        if n_clusters <= 1:
            return np.zeros(n, dtype=int)
        if n_clusters >= n:
            # Put each point in its own cluster (deterministic)
            return np.arange(n, dtype=int)

        rng_master = np.random.default_rng(random_state)
        if n <= 200:
            n_init = 8
        elif n <= 1000:
            n_init = 4
        else:
            n_init = 2

        best_inertia = np.inf
        best_labels = np.zeros(n, dtype=int)

        # Work in float32 where possible for better performance on large data
        Xf = X.astype(np.float32, copy=False)
        X_sq = np.sum(Xf * Xf, axis=1).astype(np.float32)

        for _ in range(n_init):
            seed = int(rng_master.integers(0, 2 ** 31 - 1))
            centers = self._init_centers_kmeanspp(Xf, n_clusters, seed)
            centers = centers.astype(np.float32, copy=False)

            labels = np.full(n, -1, dtype=np.int32)

            for it in range(max_iter):
                centers_sq = np.sum(centers * centers, axis=1).astype(np.float32)
                # Efficient distance computation using ||x||^2 + ||c||^2 - 2 x.c
                dots = Xf.dot(centers.T)
                dists = X_sq[:, None] + centers_sq[None, :] - 2.0 * dots
                new_labels = np.argmin(dists, axis=1).astype(np.int32)

                if it > 0 and np.array_equal(new_labels, labels):
                    break
                labels = new_labels

                old_centers = centers.copy()

                # Vectorized center update
                centers.fill(0.0)
                np.add.at(centers, labels, Xf)
                counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)

                zero_mask = counts == 0
                counts_safe = counts.copy()
                counts_safe[zero_mask] = 1.0  # avoid div-by-zero
                centers = centers / counts_safe[:, None]

                # Reinitialize any empty clusters
                if zero_mask.any():
                    reinit_idx = rng_master.integers(0, n, size=int(zero_mask.sum()))
                    centers[zero_mask] = Xf[reinit_idx]

                center_shift = float(np.sum((centers - old_centers) ** 2))
                if center_shift <= tol:
                    break

            # compute inertia from last distances (guaranteed to exist)
            inertia = float(dists[np.arange(n), labels].sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

        return best_labels

    def _init_centers_kmeanspp(self, X: np.ndarray, k: int, seed: int) -> np.ndarray:
        """
        Deterministic k-means++ initialization using numpy RNG.
        """
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        centers = np.empty((k, X.shape[1]), dtype=X.dtype)

        first_idx = int(rng.integers(0, n))
        centers[0] = X[first_idx]
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for i in range(1, k):
            total = float(closest_sq.sum())
            if not np.isfinite(total) or total <= 0:
                idx = int(rng.integers(0, n))
                centers[i] = X[idx]
                new_dist = np.sum((X - centers[i]) ** 2, axis=1)
                closest_sq = np.minimum(closest_sq, new_dist)
                continue

            probs = closest_sq / total
            try:
                idx = int(rng.choice(n, p=probs))
            except Exception:
                r = float(rng.random())
                cum = np.cumsum(probs)
                idx = int(np.searchsorted(cum, r, side="right"))
                if idx >= n:
                    idx = n - 1

            centers[i] = X[idx]
            new_dist = np.sum((X - centers[i]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_dist)

        return centers

    def _iterative_split(self, embedding: np.ndarray, target_k: int, random_state: int = 0) -> np.ndarray:
        """
        Iteratively split the largest cluster until reaching target_k clusters.
        Splitting is performed using 2-means on the cluster's embedding; if 2-means fails
        to split, fall back to median split on principal direction.
        """
        n = embedding.shape[0]
        labels = np.zeros(n, dtype=int)
        current_k = 1
        rng = np.random.default_rng(random_state)

        while current_k < target_k:
            sizes = np.bincount(labels, minlength=current_k)
            largest = int(np.argmax(sizes))
            indices = np.nonzero(labels == largest)[0]
            if indices.size <= 1:
                break  # cannot split further

            sub_X = embedding[indices]
            # Attempt a 2-means split
            sub_seed = int(rng.integers(0, 2 ** 31 - 1))
            try:
                sub_labels = self._kmeans(sub_X, 2, random_state=sub_seed)
            except Exception:
                sub_labels = np.zeros(indices.size, dtype=int)

            # If kmeans failed to split (all in one cluster), do PCA/median split
            if np.unique(sub_labels).size == 1:
                # Project onto first principal component and split at median
                try:
                    Xm = sub_X - sub_X.mean(axis=0)
                    # compute first right singular vector
                    _, _, vh = np.linalg.svd(Xm, full_matrices=False)
                    pc = vh[0]
                    proj = Xm.dot(pc)
                    med = np.median(proj)
                    sub_labels = (proj > med).astype(int)
                except Exception:
                    # Random balanced split fallback
                    rvals = rng.random(indices.size)
                    sub_labels = (rvals > 0.5).astype(int)

            # Assign new label id to the '1' group from sub_labels
            new_label = current_k
            labels[indices[sub_labels == 1]] = new_label
            # labels[indices[sub_labels == 0]] keep as 'largest'
            current_k += 1

        return labels