import unittest

import numpy as np

try:
    from AlgoTuneTasks.kd_tree.kd_tree import KDTree
except Exception as import_error:  # pragma: no cover - environment-dependent
    KDTree = None
    KD_TREE_IMPORT_ERROR = import_error
else:
    KD_TREE_IMPORT_ERROR = None


@unittest.skipIf(KDTree is None, f"KDTree import unavailable: {KD_TREE_IMPORT_ERROR}")
class KDTreeValidatorTests(unittest.TestCase):
    def test_reference_solver_passes_validation(self):
        task = KDTree()
        problem = task.generate_problem(n=4, random_seed=0)

        self.assertTrue(task.is_solution(problem, task.solve(problem)))

    def test_validator_rejects_fixed_sample_shortcut(self):
        task = KDTree()
        problem = task.generate_problem(n=7, random_seed=0)
        exact_solution = task.solve(problem)

        hacked_indices = np.array(exact_solution["indices"], copy=True)
        hacked_distances = np.array(exact_solution["distances"], copy=True)

        n_queries = len(problem["queries"])
        sample_size = min(64, n_queries)
        sample_idx = np.random.default_rng(0).choice(n_queries, sample_size, replace=False)
        sampled_rows = set(sample_idx.tolist())

        placeholder_indices = np.arange(problem["k"], dtype=int)
        placeholder_distances = np.arange(problem["k"], dtype=float)
        hacked_rows = 0

        for row_idx in range(n_queries):
            if row_idx in sampled_rows:
                continue
            hacked_indices[row_idx] = placeholder_indices
            hacked_distances[row_idx] = placeholder_distances
            hacked_rows += 1

        self.assertGreater(hacked_rows, 0)
        hacked_solution = {
            "indices": hacked_indices.tolist(),
            "distances": hacked_distances.tolist(),
        }

        self.assertFalse(task.is_solution(problem, hacked_solution))


if __name__ == "__main__":
    unittest.main()
