import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_pca_module():
    module_path = Path(__file__).resolve().parents[2] / "AlgoTuneTasks" / "pca" / "pca.py"
    fake_package = types.ModuleType("AlgoTuneTasks")
    fake_base = types.ModuleType("AlgoTuneTasks.base")

    class _Task:
        def __init__(self, **kwargs):
            pass

    def _register_task(name):
        def _decorator(cls):
            return cls

        return _decorator

    fake_base.Task = _Task
    fake_base.register_task = _register_task
    fake_package.base = fake_base

    original_package = sys.modules.get("AlgoTuneTasks")
    original_base = sys.modules.get("AlgoTuneTasks.base")
    try:
        sys.modules["AlgoTuneTasks"] = fake_package
        sys.modules["AlgoTuneTasks.base"] = fake_base
        spec = importlib.util.spec_from_file_location("test_pca_task", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_package is not None:
            sys.modules["AlgoTuneTasks"] = original_package
        else:
            sys.modules.pop("AlgoTuneTasks", None)
        if original_base is not None:
            sys.modules["AlgoTuneTasks.base"] = original_base
        else:
            sys.modules.pop("AlgoTuneTasks.base", None)


PCA = _load_pca_module().PCA


class PCAValidatorTests(unittest.TestCase):
    def test_reference_solver_passes_validation(self):
        task = PCA()
        problem = task.generate_problem(n=3, random_seed=0)

        self.assertTrue(task.is_solution(problem, task.solve(problem)))

    def test_validator_accepts_sign_flipped_reference_components(self):
        task = PCA()
        problem = task.generate_problem(n=3, random_seed=1)
        solution = np.array(task.solve(problem), dtype=float)
        solution[1::2] *= -1.0

        self.assertTrue(task.is_solution(problem, solution.tolist()))

    def test_validator_rejects_rotated_principal_subspace_shortcut(self):
        task = PCA()
        problem = {
            "X": [
                [1.0, 2.0, 0.0, 1.0, 3.0],
                [2.0, 0.0, 1.0, 4.0, 1.0],
                [0.0, 1.0, 3.0, 2.0, 2.0],
            ],
            "n_components": 2,
        }
        reference = np.array(task.solve(problem), dtype=float)
        rotation = np.array([[1.0, 1.0], [-1.0, 1.0]], dtype=float) / np.sqrt(2.0)
        rotated = rotation @ reference

        self.assertFalse(task.is_solution(problem, rotated.tolist()))


if __name__ == "__main__":
    unittest.main()
