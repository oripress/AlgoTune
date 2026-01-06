import argparse
import json
import logging
import time

from utils import find_task_class

logging.basicConfig(level=logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(description="Generate, solve, and time a HardProblemBench task for a given n.")
    parser.add_argument("task_name", type=str, help="The name of the task to run (e.g., 'aes_gcm_encryption').")
    parser.add_argument("n_value", type=int, help="The integer input parameter 'n' for the task's generate_problem method.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for problem generation (default: use n_value).")
    args = parser.parse_args()

    task_name = args.task_name
    n_value = args.n_value
    # Use n_value as seed if not provided, for consistency with test_timing.py
    random_seed = args.seed if args.seed is not None else hash(n_value)

    task_instance = find_task_class(task_name)()
    problem_input = task_instance.generate_problem(n=n_value, random_seed=random_seed)
    logging.debug(f"Generated problem input: {problem_input}")

    start_time = time.perf_counter()
    solution = task_instance.solve(problem_input)
    end_time = time.perf_counter()
    solve_duration = end_time - start_time
    logging.debug(f"Problem solved in {solve_duration:.6f} seconds.")

    is_valid = task_instance.is_solution(problem_input, solution)

    print(json.dumps({
        "solve_duration_seconds": solve_duration,
        "is_valid": is_valid
    }))

if __name__ == "__main__":
    main()
