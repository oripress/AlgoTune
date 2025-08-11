import numpy as np
import hdbscan

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        dataset = np.array(problem["dataset"], dtype=np.float32)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        
        n, dim = dataset.shape
        
        # Ultra-aggressive HDBSCAN optimization
        # The key is to minimize the computational bottlenecks
        
        # For very small datasets, use minimal overhead
        if n < 50:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                algorithm='generic',  # No tree overhead for tiny datasets
                core_dist_n_jobs=1,
                approx_min_span_tree=False,  # Exact is faster for tiny data
                gen_min_span_tree=False,
                memory=None,  # No memory overhead
                cluster_selection_method='leaf',
                allow_single_cluster=False,
                prediction_data=False
            )
        # For medium datasets, use balanced optimization
        elif n < 500:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                algorithm='prims_kdtree',
                leaf_size=2,  # Optimal aggressive leaf size
                core_dist_n_jobs=-1,
                approx_min_span_tree=True,
                gen_min_span_tree=False,
                memory='pymem',
                cluster_selection_method='leaf',
                allow_single_cluster=False,
                prediction_data=False,
                alpha=1.0,
                cluster_selection_epsilon=0.0
            )
        # For large datasets, use maximum parallelization
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                algorithm='prims_kdtree',
                leaf_size=20,  # Optimized leaf size for big data
                core_dist_n_jobs=-1,
                approx_min_span_tree=True,
                gen_min_span_tree=False,
                memory='pymem',
                cluster_selection_method='leaf',
                allow_single_cluster=False,
                prediction_data=False,
                alpha=1.0,
                cluster_selection_epsilon=0.0
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