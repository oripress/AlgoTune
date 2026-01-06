import os
import re

def check_task_names():
    tasks_dir = "AlgoTuneTasks"
    errors = []

    for task_dir in os.listdir(tasks_dir):
        task_path = os.path.join(tasks_dir, task_dir)
        if not os.path.isdir(task_path):
            continue

        task_file = None
        for file in os.listdir(task_path):
            if file.endswith(".py") and file != "__init__.py":
                task_file = os.path.join(task_path, file)
                break

        if task_file is None:
            continue

        with open(task_file, "r") as f:
            content = f.read()

        match = re.search(r"@register_task\(\"([a-zA-Z0-9_]+)\"\)", content)
        if not match:
            errors.append(f"Could not find register_task in {task_file}")
            continue

        task_name = match.group(1)

        if task_name != task_dir:
            errors.append(f"Task name {task_name} does not match directory name {task_dir} in {task_file}")

    if errors:
        print("Errors:")
        for error in errors:
            print(error)
        exit(1)
    else:
        print("Task names check completed successfully.")

if __name__ == "__main__":
    check_task_names()
