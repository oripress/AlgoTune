import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_minimum_volume_ellipsoid_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "AlgoTuneTasks"
        / "minimum_volume_ellipsoid"
        / "minimum_volume_ellipsoid.py"
    )
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
        spec = importlib.util.spec_from_file_location("test_minimum_volume_ellipsoid_task", module_path)
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


MinimumVolumeEllipsoid = _load_minimum_volume_ellipsoid_module().MinimumVolumeEllipsoid


class MinimumVolumeEllipsoidValidatorTests(unittest.TestCase):
    def test_reference_solver_passes_validation(self):
        task = MinimumVolumeEllipsoid()
        problem = task.generate_problem(n=4, random_seed=0)

        self.assertTrue(task.is_solution(problem, task.solve(problem)))

    def test_validator_rejects_broadcastable_wrong_shape_center(self):
        task = MinimumVolumeEllipsoid()
        problem = task.generate_problem(n=2, random_seed=0)
        solution = task.solve(problem)
        center = np.asarray(solution["ellipsoid"]["Y"], dtype=float)
        hacked_solution = {
            "objective_value": solution["objective_value"],
            "ellipsoid": {
                "X": solution["ellipsoid"]["X"],
                "Y": center.reshape(1, 1),
            },
        }

        self.assertFalse(task.is_solution(problem, hacked_solution))


if __name__ == "__main__":
    unittest.main()
