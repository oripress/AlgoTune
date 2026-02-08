import numpy as np
from typing import Any
import hdbscan
from hdbscan._hdbscan_tree import condense_tree, compute_stability, get_clusters
from hdbscan._hdbscan_linkage import label as label_mst
import faiss
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix, lil_matrix
from numba import njit


@njit(cache=True)
def _prim_mst_full(dataset, core_distances, n, dim):
    """
    Prim's MST on the full mutual reachability graph.
    O(n^2) but with good constants using numba.
    """
    in_tree = np.zeros(n, dtype=np.bool_)
    min_edge_weight = np.full(n, np.inf, dtype=np.float64)
    min_edge_from = np.full(n, -1, dtype=np.int64)
    
    mst_edges = np.empty((n - 1, 3), dtype=np.float64)
    
    # Start from node 0
    in_tree[0] = True
    cd_0 = core_distances[0]
    
    for j in range(1, n):
        d_sq = 0.0
        for dd in range(dim):
            diff = dataset[0, dd] - dataset[j, dd]
            d_sq += diff * diff
        d = d_sq ** 0.5
        mr = cd_0
        if core_distances[j] > mr:
            mr = core_distances[j]
        if d > mr:
            mr = d
        min_edge_weight[j] = mr
        min_edge_from[j] = 0
    
    for step in range(n - 1):
        min_val = np.inf
        min_node = -1
        for j in range(n):
            if not in_tree[j] and min_edge_weight[j] < min_val:
                min_val = min_edge_weight[j]
                min_node = j
        
        if min_node < 0:
            break
        
        in_tree[min_node] = True
        mst_edges[step, 0] = min_edge_from[min_node]
        mst_edges[step, 1] = min_node
        mst_edges[step, 2] = min_val
        
        cd_new = core_distances[min_node]
        for j in range(n):
            if in_tree[j]:
                continue
            d_sq = 0.0
            for dd in range(dim):
                diff = dataset[min_node, dd] - dataset[j, dd]
                d_sq += diff * diff
            d = d_sq ** 0.5
            mr = cd_new
            if core_distances[j] > mr:
                mr = core_distances[j]
            if d > mr:
                mr = d
            if mr < min_edge_weight[j]:
                min_edge_weight[j] = mr
                min_edge_from[j] = min_node
    
    return mst_edges


@njit(cache=True)
def _build_mr_edges(knn_idx, knn_dists, core_distances, n, n_neighbors):
    """Build mutual reachability edges from kNN graph."""
    max_edges = n * n_neighbors * 2
    rows = np.empty(max_edges, dtype=np.int32)
    cols = np.empty(max_edges, dtype=np.int32)
    vals = np.empty(max_edges, dtype=np.float64)
    
    idx = 0
    for i in range(n):
        cd_i = core_distances[i]
        for j_pos in range(1, n_neighbors + 1):
            j = knn_idx[i, j_pos]
            if j < 0 or j >= n:
                continue
            d = knn_dists[i, j_pos]
            mr = cd_i
            if core_distances[j] > mr:
                mr = core_distances[j]
            if d > mr:
                mr = d
            # Add both directions
            rows[idx] = i
            cols[idx] = j
            vals[idx] = mr
            idx += 1
            rows[idx] = j
            cols[idx] = i
            vals[idx] = mr
            idx += 1
    
    return rows[:idx], cols[:idx], vals[:idx]


# Trigger compilation
_d = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
_c = np.array([1.0, 1.0, 1.0], dtype=np.float64)
_prim_mst_full(_d, _c, 3, 2)

_ki = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.int64)
_kd = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64)
_build_mr_edges(_ki, _kd, _c, 3, 2)


def _fast_hdbscan_small(dataset, min_cluster_size, min_samples):
    """Fast HDBSCAN for small-medium datasets using FAISS kNN + numba Prim's."""
    n, dim = dataset.shape
    
    if n < 2:
        return np.full(n, -1), np.zeros(n), []
    
    k = min_samples
    
    # Compute core distances using FAISS
    dataset_f32 = np.ascontiguousarray(dataset, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(dataset_f32)
    
    sq_dists, _ = index.search(dataset_f32, k + 1)
    core_distances = np.sqrt(np.maximum(sq_dists[:, k], 0.0)).astype(np.float64)
    
    # Prim's MST
    dataset_c = np.ascontiguousarray(dataset, dtype=np.float64)
    mst_edges = _prim_mst_full(dataset_c, core_distances, n, dim)
    
    return _finish_hdbscan(mst_edges, min_cluster_size, n)


def _fast_hdbscan_large(dataset, min_cluster_size, min_samples):
    """Fast HDBSCAN for large datasets using FAISS kNN + scipy MST."""
    n, dim = dataset.shape
    
    if n < 2:
        return np.full(n, -1), np.zeros(n), []
    
    k = min_samples
    
    # Compute kNN using FAISS
    dataset_f32 = np.ascontiguousarray(dataset, dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(dataset_f32)
    
    n_neighbors = min(n - 1, max(2 * k + 1, 2 * min_cluster_size, 64))
    
    sq_dists, knn_idx = index.search(dataset_f32, n_neighbors + 1)
    knn_dists = np.sqrt(np.maximum(sq_dists, 0.0)).astype(np.float64)
    knn_idx_int = knn_idx.astype(np.int64)
    
    core_distances = knn_dists[:, k].copy()
    
    # Build sparse mutual reachability graph
    rows, cols, vals = _build_mr_edges(knn_idx_int, knn_dists, core_distances, n, n_neighbors)
    
    graph = csr_matrix((vals, (rows.astype(np.int64), cols.astype(np.int64))), shape=(n, n))
    
    # Compute MST
    mst_sparse = minimum_spanning_tree(graph)
    mst_coo = mst_sparse.tocoo()
    
    n_edges = len(mst_coo.data)
    
    if n_edges < n - 1:
        # Handle disconnected components
        n_components, component_labels = connected_components(graph, directed=False)
        
        if n_components > 1:
            # Find closest pair between components using FAISS
            # For each pair of components, find the closest points
            # Simple approach: for each component, find its centroid and connect to nearest other component centroid
            
            # Better: use brute force between component representatives
            # Fallback to full Prim's for small enough datasets
            if n <= 10000:
                dataset_c = np.ascontiguousarray(dataset, dtype=np.float64)
                mst_edges = _prim_mst_full(dataset_c, core_distances, n, dim)
                return _finish_hdbscan(mst_edges, min_cluster_size, n)
            
            # For very large datasets, try increasing neighbors
            n_neighbors2 = min(n - 1, n_neighbors * 4)
            if n_neighbors2 > n_neighbors:
                sq_dists2, knn_idx2 = index.search(dataset_f32, n_neighbors2 + 1)
                knn_dists2 = np.sqrt(np.maximum(sq_dists2, 0.0)).astype(np.float64)
                knn_idx2_int = knn_idx2.astype(np.int64)
                
                rows2, cols2, vals2 = _build_mr_edges(knn_idx2_int, knn_dists2, core_distances, n, n_neighbors2)
                graph2 = csr_matrix((vals2, (rows2.astype(np.int64), cols2.astype(np.int64))), shape=(n, n))
                mst_sparse = minimum_spanning_tree(graph2)
                mst_coo = mst_sparse.tocoo()
                n_edges = len(mst_coo.data)
                
                if n_edges < n - 1:
                    return None  # Fall back to library
            else:
                return None
    
    # Convert to edge array
    edges = np.column_stack([
        mst_coo.row.astype(np.float64),
        mst_coo.col.astype(np.float64),
        mst_coo.data
    ])
    
    order = np.argsort(edges[:, 2])
    sorted_edges = edges[order].copy()
    
    return _finish_hdbscan(sorted_edges, min_cluster_size, n)


def _finish_hdbscan(sorted_edges, min_cluster_size, n):
    """Complete HDBSCAN from sorted MST edges."""
    # Build single linkage tree
    single_linkage_tree = label_mst(sorted_edges)
    
    # Condense tree and extract clusters
    condensed_tree_arr = condense_tree(single_linkage_tree, min_cluster_size)
    stability_dict = compute_stability(condensed_tree_arr)
    labels, probabilities, stabilities = get_clusters(condensed_tree_arr, stability_dict)
    
    # Compute cluster persistence
    unique_labels = sorted(set(int(l) for l in labels if l >= 0))
    
    if unique_labels:
        persistence = []
        for cluster_label in unique_labels:
            cluster_id = cluster_label + n
            parent_mask = condensed_tree_arr['parent'] == cluster_id
            if np.any(parent_mask):
                lambdas = condensed_tree_arr['lambda_val'][parent_mask]
                child_mask = condensed_tree_arr['child'] == cluster_id
                if np.any(child_mask):
                    birth_lambda = condensed_tree_arr['lambda_val'][child_mask][0]
                else:
                    birth_lambda = lambdas.min()
                death_lambda = lambdas.max()
                persistence.append(float(death_lambda - birth_lambda))
            else:
                persistence.append(0.0)
    else:
        persistence = []
    
    return labels, probabilities, persistence


class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        dataset = np.array(problem["dataset"], dtype=np.float64)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        
        n, dim = dataset.shape
        
        try:
            if n <= 4000:
                result = _fast_hdbscan_small(dataset, min_cluster_size, min_samples)
            else:
                result = _fast_hdbscan_large(dataset, min_cluster_size, min_samples)
            
            if result is not None:
                labels, probabilities, persistence = result
                labels = np.asarray(labels)
                probabilities = np.asarray(probabilities)
                solution = {
                    "labels": labels.tolist(),
                    "probabilities": probabilities.tolist(),
                    "cluster_persistence": persistence if isinstance(persistence, list) else list(persistence),
                    "num_clusters": len(set(int(l) for l in labels if l >= 0)),
                    "num_noise_points": int(np.sum(labels == -1)),
                }
                return solution
        except Exception:
            pass
        
        # Fallback to standard HDBSCAN
        if dim <= 30:
            algorithm = 'boruvka_kdtree'
        else:
            algorithm = 'boruvka_balltree'
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algorithm=algorithm,
            core_dist_n_jobs=-1,
            leaf_size=40,
            gen_min_span_tree=False,
            prediction_data=False,
        )
        clusterer.fit(dataset)
        
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_
        
        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": len(set(labels[labels != -1])),
            "num_noise_points": int(np.sum(labels == -1)),
        }
        return solution