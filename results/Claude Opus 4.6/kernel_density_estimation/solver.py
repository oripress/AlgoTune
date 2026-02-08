import numpy as np
from scipy.special import logsumexp, gammaln


class Solver:
    def solve(self, problem, **kwargs):
        X = np.asarray(problem["data_points"], dtype=np.float64)
        X_q = np.asarray(problem["query_points"], dtype=np.float64)
        kernel = problem["kernel"]
        h = float(problem["bandwidth"])

        n, d = X.shape
        m = X_q.shape[0]

        if m == 0:
            return {"log_density": []}

        # Precompute squared norms for efficient distance computation
        X_sqn = np.einsum('ij,ij->i', X, X)
        Xq_sqn = np.einsum('ij,ij->i', X_q, X_q)

        # Chunk size to limit memory (~160MB per chunk)
        max_elems = 20_000_000
        cs = max(1, max_elems // max(n, 1))
        ld = np.empty(m)

        if kernel == 'gaussian':
            log_norm = -np.log(n) - d * np.log(h) - 0.5 * d * np.log(2 * np.pi)
            neg_inv_2h2 = -0.5 / (h * h)
            for s in range(0, m, cs):
                e = min(s + cs, m)
                D = Xq_sqn[s:e, None] + X_sqn[None, :] - 2.0 * (X_q[s:e] @ X.T)
                np.maximum(D, 0, out=D)
                D *= neg_inv_2h2
                ld[s:e] = logsumexp(D, axis=1) + log_norm

        elif kernel == 'tophat':
            log_Vd = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1)
            log_base = -np.log(n) - d * np.log(h) - log_Vd
            h_sq = h * h
            for s in range(0, m, cs):
                e = min(s + cs, m)
                D = Xq_sqn[s:e, None] + X_sqn[None, :] - 2.0 * (X_q[s:e] @ X.T)
                counts = np.sum(D < h_sq, axis=1).astype(np.float64)
                with np.errstate(divide='ignore'):
                    ld[s:e] = np.log(counts) + log_base

        elif kernel == 'epanechnikov':
            log_Vd = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1)
            log_norm = np.log(d + 2) - np.log(2) - log_Vd - np.log(n) - d * np.log(h)
            inv_h2 = 1.0 / (h * h)
            for s in range(0, m, cs):
                e = min(s + cs, m)
                D = Xq_sqn[s:e, None] + X_sqn[None, :] - 2.0 * (X_q[s:e] @ X.T)
                np.maximum(D, 0, out=D)
                D *= inv_h2
                np.subtract(1.0, D, out=D)
                np.maximum(D, 0, out=D)
                total = D.sum(axis=1)
                with np.errstate(divide='ignore'):
                    ld[s:e] = np.log(total) + log_norm

        elif kernel == 'exponential':
            log_Sd = np.log(2) + 0.5 * d * np.log(np.pi) - gammaln(0.5 * d)
            log_norm = -np.log(n) - d * np.log(h) - log_Sd - gammaln(d)
            neg_inv_h = -1.0 / h
            for s in range(0, m, cs):
                e = min(s + cs, m)
                D = Xq_sqn[s:e, None] + X_sqn[None, :] - 2.0 * (X_q[s:e] @ X.T)
                np.maximum(D, 0, out=D)
                np.sqrt(D, out=D)
                D *= neg_inv_h
                ld[s:e] = logsumexp(D, axis=1) + log_norm

        elif kernel == 'linear':
            log_Vd = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1)
            log_norm = np.log(d + 1) - log_Vd - np.log(n) - d * np.log(h)
            inv_h = 1.0 / h
            for s in range(0, m, cs):
                e = min(s + cs, m)
                D = Xq_sqn[s:e, None] + X_sqn[None, :] - 2.0 * (X_q[s:e] @ X.T)
                np.maximum(D, 0, out=D)
                np.sqrt(D, out=D)
                D *= inv_h
                np.subtract(1.0, D, out=D)
                np.maximum(D, 0, out=D)
                total = D.sum(axis=1)
                with np.errstate(divide='ignore'):
                    ld[s:e] = np.log(total) + log_norm

        elif kernel == 'cosine':
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(kernel='cosine', bandwidth=h)
            kde.fit(X)
            ld = kde.score_samples(X_q)

        else:
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(kernel=kernel, bandwidth=h)
            kde.fit(X)
            ld = kde.score_samples(X_q)

        return {"log_density": ld.tolist()}