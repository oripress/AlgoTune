import logging
from typing import Any


# Dictionary to store registered tasks
_REGISTERED_TASKS = {}


def register_task(name):
    """
    Decorator to register a task class.
    :param name: Name of the task
    :return: Decorator function
    """

    def decorator(cls):
        _REGISTERED_TASKS[name] = cls
        logging.info(f"Registered task: {name}")
        return cls

    return decorator


class Task:
    """Base class for all tasks."""

    def generate_problem(self, n: int, random_seed: int, **kwargs: Any) -> Any:
        """Generate a problem instance."""
        raise NotImplementedError()

    def solve(self, problem: Any) -> Any:
        """Solve the given problem instance."""
        raise NotImplementedError()

    def is_solution(self, problem: Any, solution: Any) -> bool:
        """Determine whether the provided candidate solution is valid."""
        raise NotImplementedError()
