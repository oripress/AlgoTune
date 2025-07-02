"""
AlgoTuner/task_lists.py - Task list management for sequential processing

This module provides predefined task lists that can be used for sequential
evaluation modes. Users can customize these lists or create their own.
"""

from typing import List, Dict, Any
from pathlib import Path


# Predefined task lists
TASK_LISTS = {
    "fast": [
        "svm",
        "max_common_subgraph",
        "cyclic_independent_set",
    ],
    
    "medium": [
        "svm",
        "max_common_subgraph", 
        "cyclic_independent_set",
        "nmf",
        "communicability",
        "eigenvalues_real",
    ],
    
    "all": [
        "svm",
        "max_common_subgraph",
        "count_riemann_zeta_zeros", 
        "ode_stiff_robertson",
        "nmf",
        "cyclic_independent_set",
        "communicability",
        "eigenvalues_real",
    ],
    
    "compute_heavy": [
        "count_riemann_zeta_zeros",
        "ode_stiff_robertson", 
        "nmf",
        "eigenvalues_real",
    ],
    
    "graph_tasks": [
        "max_common_subgraph",
        "cyclic_independent_set", 
        "communicability",
    ]
}


def get_task_list(name: str) -> List[str]:
    """
    Get a predefined task list by name.
    
    Args:
        name: Name of the task list ("fast", "medium", "all", etc.)
        
    Returns:
        List of task names
        
    Raises:
        KeyError: If task list name doesn't exist
    """
    if name not in TASK_LISTS:
        available = ", ".join(TASK_LISTS.keys())
        raise KeyError(f"Unknown task list '{name}'. Available: {available}")
    
    return TASK_LISTS[name].copy()


def get_available_task_lists() -> List[str]:
    """Get list of available predefined task lists."""
    return list(TASK_LISTS.keys())


def filter_existing_tasks(task_list: List[str], tasks_root: Path) -> List[str]:
    """
    Filter a task list to only include tasks that exist in the tasks directory.
    
    Args:
        task_list: List of task names
        tasks_root: Root directory containing task definitions
        
    Returns:
        Filtered list of existing tasks
    """
    if not tasks_root.exists():
        return []
    
    existing_tasks = []
    for task_name in task_list:
        task_dir = tasks_root / task_name
        if task_dir.is_dir() and (task_dir / "description.txt").exists():
            existing_tasks.append(task_name)
    
    return existing_tasks


def load_custom_task_list(file_path: Path) -> List[str]:
    """
    Load a custom task list from a text file.
    
    Args:
        file_path: Path to text file with one task name per line
        
    Returns:
        List of task names
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Task list file not found: {file_path}")
    
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            task = line.strip()
            if task and not task.startswith('#'):
                tasks.append(task)
    
    return tasks


def create_task_list_file(tasks: List[str], file_path: Path) -> None:
    """
    Create a task list file from a list of task names.
    
    Args:
        tasks: List of task names
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write("# Custom task list\n")
        f.write("# One task name per line\n")
        f.write("# Lines starting with # are ignored\n\n")
        for task in tasks:
            f.write(f"{task}\n")


# Example usage
if __name__ == "__main__":
    # Print available task lists
    print("Available task lists:")
    for name in get_available_task_lists():
        tasks = get_task_list(name)
        print(f"  {name}: {tasks}")
    
    # Create example custom task list
    custom_tasks = ["svm", "nmf"]
    create_task_list_file(custom_tasks, Path("my_tasks.txt"))
    print(f"\nCreated example task list: my_tasks.txt") 