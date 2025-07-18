import numpy as np  

class Solver:  
    def solve(self, problem, **kwargs):  
        S = problem.get("similarity_matrix")  
        n_clusters = problem.get("n_clusters")  
        if S is None:  
            raise ValueError("Missing similarity_matrix")  
        if n_clusters is None:  
            raise ValueError("Missing n_clusters")  
        # cast to float32 for faster ops  
        S = np.asarray(S, dtype=np.float32)  
        n = S.shape[0]  
        if S.ndim != 2 or S.shape[1] != n:  
            raise ValueError("Invalid similarity matrix")  
        if not isinstance(n_clusters, int) or n_clusters < 1:  
            raise ValueError("Invalid number of clusters")  
        # trivial cases  
        if n == 0:  
            return {"labels": [], "n_clusters": n_clusters}  
        if n_clusters == 1:  
            return {"labels": [0] * n, "n_clusters": n_clusters}  
        if n_clusters >= n:  
            return {"labels": list(range(n)), "n_clusters": n_clusters}  

        # compute D^{-1/2}  
        D = S.sum(axis=1)  
        inv_sqrt_D = np.where(D > 0, 1.0 / np.sqrt(D), 0.0).astype(np.float32)  

        # spectral embedding  
        full_thresh = 100  
        if n <= full_thresh:  
            # exact eigen-decomposition of normalized matrix  
            M = inv_sqrt_D[:, None] * S * inv_sqrt_D[None, :]  
            _, eigvecs = np.linalg.eigh(M)  
            U = eigvecs[:, -n_clusters:]  
        else:  
            # Nystrom approximation  
            m0 = max(n_clusters * 10, 50)  
            m = min(n - 1, m0)  
            idx = np.random.permutation(n)[:m]  
            idx_inv = inv_sqrt_D[idx]  
            # C = D^{-1/2} * S[:, idx] * D^{-1/2}[idx]  
            C = S[:, idx] * idx_inv[None, :]  
            C = C * inv_sqrt_D[:, None]  
            # W submatrix  
            W = S[np.ix_(idx, idx)] * (idx_inv[:, None] * idx_inv[None, :])  
            w_vals, w_vecs = np.linalg.eigh(W)  
            sel = np.argsort(w_vals)[-n_clusters:]  
            vals = w_vals[sel]  
            vecs = w_vecs[:, sel]  
            # extend eigenvectors to full set  
            U = C.dot(vecs / np.sqrt(vals)[None, :])  

        # row-normalize embedding  
        norms = np.linalg.norm(U, axis=1)  
        nz = norms > 0  
        U[nz] /= norms[nz, None]  

        # assign clusters by max dimension  
        labels = np.argmax(U, axis=1)  
        # ensure every cluster label appears  
        present = set(labels.tolist())  
        for j in range(n_clusters):  
            if j not in present:  
                labels[np.argmax(U[:, j])] = j  
                present.add(j)  

        # to native list of ints  
        labels = labels.tolist()  
        return {"labels": labels, "n_clusters": n_clusters}