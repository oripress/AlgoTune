import numpy as np
import numba

@numba.njit(fastmath=True, parallel=True)
def gaussian_kde_numba(X_q, X, h2, d):
    n_query = X_q.shape[0]
    n_data = X.shape[0]
    log_dens = np.empty(n_query, dtype=np.float64)

    for i in range(n_query):
        # logsumexp trick for numerical stability
        m = -np.inf
        for j in range(n_data):
            sq_dist = 0.0
            for k in range(d):
                diff = X_q[i, k] - X[j, k]
                sq_dist += diff * diff
            val = -0.5 * sq_dist / h2
            if val > m:
                m = val
        
        if m == -np.inf:
            log_dens[i] = -np.inf
            continue

        s = 0.0
        for j in range(n_data):
            sq_dist = 0.0
            for k in range(d):
                diff = X_q[i, k] - X[j, k]
                sq_dist += diff * diff
            s += np.exp(-0.5 * sq_dist / h2 - m)
        
        log_dens[i] = m + np.log(s)
    return log_dens

@numba.njit(fastmath=True, parallel=True)
def tophat_kde_numba(X_q, X, h2, d):
    n_query = X_q.shape[0]
    n_neighbors = np.empty(n_query, dtype=np.int64)

    for i in range(n_query):
        count = 0
        for j in range(X.shape[0]):
            sq_dist = 0.0
            for k in range(d):
                diff = X_q[i, k] - X[j, k]
                sq_dist += diff * diff
            if sq_dist <= h2:
                count += 1
        n_neighbors[i] = count
    return n_neighbors

@numba.njit(fastmath=True, parallel=True)
def exponential_kde_numba(X_q, X, h, d):
    n_query = X_q.shape[0]
    log_dens = np.empty(n_query, dtype=np.float64)

    for i in range(n_query):
        # logsumexp trick
        m = -np.inf
        for j in range(X.shape[0]):
            sq_dist = 0.0
            for k in range(d):
                diff = X_q[i, k] - X[j, k]
                sq_dist += diff * diff
            val = -np.sqrt(sq_dist) / h
            if val > m:
                m = val
        
        if m == -np.inf:
            log_dens[i] = -np.inf
            continue

        s = 0.0
        for j in range(X.shape[0]):
            sq_dist = 0.0
            for k in range(d):
                diff = X_q[i, k] - X[j, k]
                sq_dist += diff * diff
            s += np.exp(-np.sqrt(sq_dist) / h - m)
        
        log_dens[i] = m + np.log(s)
    return log_dens

@numba.njit(fastmath=True, parallel=True)
def sum_kde_numba(X_q, X, h, h2, d, kernel_id):
    n_query = X_q.shape[0]
    sum_vals = np.empty(n_query, dtype=np.float64)

    for i in range(n_query):
        s = 0.0
        if kernel_id == 0:  # epanechnikov
            for j in range(X.shape[0]):
                sq_dist = 0.0
                for k in range(d):
                    diff = X_q[i, k] - X[j, k]
                    sq_dist += diff * diff
                if sq_dist <= h2:
                    s += 1.0 - sq_dist / h2
        elif kernel_id == 1:  # linear
            for j in range(X.shape[0]):
                sq_dist = 0.0
                for k in range(d):
                    diff = X_q[i, k] - X[j, k]
                    sq_dist += diff * diff
                if sq_dist <= h2:
                    s += 1.0 - np.sqrt(sq_dist) / h
        elif kernel_id == 2:  # cosine
            for j in range(X.shape[0]):
                sq_dist = 0.0
                for k in range(d):
                    diff = X_q[i, k] - X[j, k]
                    sq_dist += diff * diff
                if sq_dist <= h2:
                    s += np.cos(np.pi / 2.0 * np.sqrt(sq_dist) / h)
        sum_vals[i] = s
    return sum_vals