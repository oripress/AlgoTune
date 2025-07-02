#!/usr/bin/env python3

"""
algotune.py - Unified entry point for AlgoTune timing evaluations

This is the main script for running timing evaluations. It provides clear
subcommands for different modes and automatically handles environment setup.

Usage:
  # Auto-detection mode (SLURM if available, standalone otherwise)
  python3 algotune.py timing --target-time-ms 100
  
  # Explicit standalone mode
  python3 algotune.py timing --target-time-ms 100 --standalone
  
  # Multiple specific tasks
  python3 algotune.py timing --target-time-ms 100 --tasks svm kmeans qr_factorization
  
  # Sequential processing
  python3 algotune.py timing --target-time-ms 100 --standalone --sequential
  
  # Single task
  python3 algotune.py single --task svm --target-time-ms 100
  
  # List available tasks/task-lists
  python3 algotune.py list-tasks
  python3 algotune.py list-task-lists
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
import os
import re

# Add project to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies() -> bool:
    """Check if required dependencies are available in current environment."""
    try:
        import orjson
        import numpy
        return True
    except ImportError:
        return False

def run_with_singularity(script_path: str, args: list) -> None:
    """Run a Python script inside Singularity container."""
    project_root = Path(__file__).parent.parent
    
    # Try to load configuration for Singularity settings
    try:
        # Try unified config first
        config_file = project_root / "config.env"
        if not config_file.exists():
            config_file = project_root / "slurm/run_config.env"
        
        if config_file.exists():
            print(f"📋 Loading config from: {config_file}")
            # Parse the config file to get SINGULARITY_IMAGE and paths
            env_vars = {}
            for line in config_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.replace("export ", "").strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
            
            singularity_image = env_vars.get("SINGULARITY_IMAGE")
            data_dir = env_vars.get("DATA_DIR")
            temp_dir = env_vars.get("TEMP_DIR_STORAGE")
            
            if singularity_image and Path(singularity_image).exists():
                print(f"🐍 Running via Singularity: {singularity_image}")
                
                # Build bind mounts
                bind_mounts = [f"{project_root}:/app"]
                
                # Add DATA_DIR if it exists and is different from project
                if data_dir and data_dir != str(project_root):
                    # Create parent directories if they don't exist (host side)
                    try:
                        Path(data_dir).mkdir(parents=True, exist_ok=True)
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not create DATA_DIR {data_dir}: {e}")
                    bind_mounts.append(f"{data_dir}:{data_dir}")
                
                # Add TEMP_DIR_STORAGE if it exists and is different 
                if temp_dir and temp_dir not in ["/tmp", str(project_root)] and temp_dir != data_dir:
                    try:
                        Path(temp_dir).mkdir(parents=True, exist_ok=True)
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not create TEMP_DIR {temp_dir}: {e}")
                    bind_mounts.append(f"{temp_dir}:{temp_dir}")
                
                # Build command with all bind mounts
                cmd = ["singularity", "exec"]
                for bind in bind_mounts:
                    cmd.extend(["--bind", bind])
                
                # Add environment variables
                cmd.extend(["--env", f"PYTHONPATH=/app"])
                cmd.extend(["--env", "CODE_DIR=/app"])
                
                if data_dir:
                    cmd.extend(["--env", f"DATA_DIR={data_dir}"])
                else:
                    cmd.extend(["--env", "DATA_DIR=/app/data"])
                    
                if temp_dir:
                    cmd.extend(["--env", f"TEMP_DIR_STORAGE={temp_dir}"])
                else:
                    cmd.extend(["--env", "TEMP_DIR_STORAGE=/tmp"])
                
                # Add image and command
                cmd.extend([singularity_image, "python3", f"/app/{script_path}"] + args)
                
                print(f"🔗 Bind mounts: {bind_mounts}")
                result = subprocess.run(cmd)
                sys.exit(result.returncode)
    except Exception as e:
        print(f"Warning: Could not use Singularity: {e}")
    
    # Fallback: try to run directly
    print("❌ Dependencies not available and Singularity not configured.")
    print("Please either:")
    print("1. Install dependencies: pip install -e .")
    print("2. Configure Singularity in config.env or slurm/run_config.env")
    print("3. Run from inside the container")
    sys.exit(1)

def run_timing_command(args):
    """Run the main timing evaluation."""
    # For SLURM mode, we need to run on the host (where sbatch is available)
    # Only use Singularity for standalone mode when dependencies are missing
    
    # Check if we can import required modules
    if not check_dependencies():
        # Dependencies not available
        if not args.standalone:
            # SLURM mode - try to run on host anyway (sbatch needs to be on host)
            print("⚠️  Dependencies not available but SLURM mode requires host execution")
            print("    Attempting to run SLURM submission on host...")
            # Fall through to run directly (SLURM jobs will use Singularity)
        else:
            # Standalone mode - run via Singularity
            timing_args = []
            
            if args.target_time_ms is not None:
                timing_args.extend(["--target-time-ms", str(args.target_time_ms)])
            
            if args.standalone:
                timing_args.append("--standalone")
            
            if args.sequential:
                timing_args.append("--sequential")
                
            if args.task:
                timing_args.extend(["--task", args.task])
                
            if args.tasks:
                timing_args.extend(["--tasks"] + args.tasks)
                
            if args.task_list:
                timing_args.extend(["--task-list", args.task_list])
                
            if args.task_list_file:
                timing_args.extend(["--task-list-file", str(args.task_list_file)])
                
            if args.data_dir:
                timing_args.extend(["--data-dir", str(args.data_dir)])
            
            if args.lazy:
                timing_args.append("--lazy")
            
            run_with_singularity("scripts/submit_generate_python.py", timing_args)
            return
    
    # Dependencies available OR SLURM mode - run directly via subprocess
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts/submit_generate_python.py"
    
    # Build command arguments  
    cmd_args = [sys.executable, str(script_path)]
    
    if args.target_time_ms is not None:
        cmd_args.extend(["--target-time-ms", str(args.target_time_ms)])
    
    if args.standalone:
        cmd_args.append("--standalone")
    
    if args.sequential:
        cmd_args.append("--sequential")
        
    if args.task:
        cmd_args.extend(["--task", args.task])
        
    if args.tasks:
        cmd_args.extend(["--tasks"] + args.tasks)
        
    if args.task_list:
        cmd_args.extend(["--task-list", args.task_list])
        
    if args.task_list_file:
        cmd_args.extend(["--task-list-file", str(args.task_list_file)])
        
    if args.data_dir:
        cmd_args.extend(["--data-dir", str(args.data_dir)])
    
    if args.lazy:
        cmd_args.append("--lazy")
    
    # Run the script as subprocess
    result = subprocess.run(cmd_args, cwd=project_root)
    sys.exit(result.returncode)


def run_single_command(args):
    """Run single task evaluation."""
    # Check if we can import required modules
    if not check_dependencies():
        # Try to run via Singularity
        core_args = [
            "--task", args.task,
            "--target-time-ms", str(args.target_time_ms),
            "--data-dir", str(args.data_dir)
        ]
        
        if args.num_runs:
            core_args.extend(["--num-runs", str(args.num_runs)])
            
        if args.override_k:
            core_args.extend(["--override-k", str(args.override_k)])
            
        if args.force_regenerate:
            core_args.append("--force-regenerate")
            
        if args.output:
            core_args.extend(["--output", str(args.output)])
        
        # Use timing_core directly - create a temporary script that calls timing_core.main()
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write("""#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')
from AlgoTuner.timing_core import main
if __name__ == "__main__":
    main()
""")
            temp_script = tmp.name
        try:
            run_with_singularity(temp_script, core_args)
        finally:
            import os
            os.unlink(temp_script)
        return
    
    # Dependencies available - run timing_core directly
    project_root = Path(__file__).parent.parent
    
    # Build command arguments  
    cmd_args = [
        sys.executable, "-m", "AlgoTuner.timing_core",
        "--task", args.task,
        "--target-time-ms", str(args.target_time_ms),
        "--data-dir", str(args.data_dir)
    ]
    
    if args.num_runs:
        cmd_args.extend(["--num-runs", str(args.num_runs)])
        
    if args.override_k:
        cmd_args.extend(["--override-k", str(args.override_k)])
        
    if args.force_regenerate:
        cmd_args.append("--force-regenerate")
        
    if args.output:
        cmd_args.extend(["--output", str(args.output)])
    
    # Run the script as subprocess
    result = subprocess.run(cmd_args, cwd=project_root)
    sys.exit(result.returncode)


def list_tasks_command(args):
    """List available tasks."""
    project_root = Path(__file__).parent.parent
    tasks_root = project_root / "AlgoTuneTasks"
    
    if not tasks_root.exists():
        print("❌ AlgoTuneTasks directory not found")
        return
    
    # Simple directory-based task discovery (avoid importing submit_agent)
    valid_tasks = []
    for task_dir in tasks_root.iterdir():
        if task_dir.is_dir() and (task_dir / "description.txt").exists():
            valid_tasks.append(task_dir.name)
    
    print("Available tasks:")
    for task in sorted(valid_tasks):
        print(f"  {task}")
    print(f"\nTotal: {len(valid_tasks)} tasks")


def list_task_lists_command(args):
    """List available predefined task lists."""
    # Hardcode the task lists to avoid import issues
    task_lists = {
        "fast": ["svm", "max_common_subgraph", "cyclic_independent_set"],
        "medium": ["svm", "max_common_subgraph", "cyclic_independent_set", "nmf", "communicability", "eigenvalues_real"],
        "all": ["svm", "max_common_subgraph", "count_riemann_zeta_zeros", "ode_stiff_robertson", "nmf", "cyclic_independent_set", "communicability", "eigenvalues_real"],
        "compute_heavy": ["count_riemann_zeta_zeros", "ode_stiff_robertson", "nmf", "eigenvalues_real"],
        "graph_tasks": ["max_common_subgraph", "cyclic_independent_set", "communicability"]
    }
    
    print("Available task lists:")
    for name, tasks in task_lists.items():
        print(f"  {name}: {tasks}")


def auto_detect_and_run(args):
    """Auto-detect environment and run timing evaluation via submit_agent.py."""
    project_root = Path(__file__).parent.parent
    python_script = project_root / "scripts/submit_agent.py"
    
    # Build command arguments
    cmd_args = [sys.executable, str(python_script), "--target-time-ms", str(args.target_time_ms)]
    
    if args.standalone:
        cmd_args.append("--standalone")
    
    if args.sequential:
        cmd_args.append("--sequential")
        
    if args.task:
        cmd_args.extend(["--task", args.task])
        
    if args.tasks:
        cmd_args.extend(["--tasks"] + args.tasks)
        
    if args.task_list:
        cmd_args.extend(["--task-list", args.task_list])
        
    if args.task_list_file:
        cmd_args.extend(["--task-list-file", str(args.task_list_file)])
        
    if args.data_dir:
        cmd_args.extend(["--data-dir", str(args.data_dir)])
    
    # Run the python script
    result = subprocess.run(cmd_args, cwd=project_root)
    sys.exit(result.returncode)


def run_agent_command(args):
    """Run AI agent on tasks."""
    if not check_dependencies():
        # Dependencies not available - run via Singularity
        project_root = Path(__file__).parent.parent
        
        # Get list of tasks
        tasks = args.tasks if args.tasks else []
        
        # If no tasks specified, get all available tasks
        if not tasks:
            tasks_root = project_root / "AlgoTuneTasks"
            if tasks_root.exists():
                for task_dir in tasks_root.iterdir():
                    if task_dir.is_dir() and (task_dir / "description.txt").exists():
                        tasks.append(task_dir.name)
        
        if not tasks:
            print("❌ No tasks found")
            sys.exit(1)
        
        print(f"📋 Running agent on {len(tasks)} task(s) via Singularity...")
        
        # Run each task via Singularity
        for task in tasks:
            print(f"\n🎯 Running task: {task}")
            agent_args = ["--model", args.model, "--task", task]
            run_with_singularity("AlgoTuner/main.py", agent_args)
    else:
        # Dependencies available - run directly
        project_root = Path(__file__).parent.parent
        
        # Set up CODE_DIR if not set
        if not os.environ.get("CODE_DIR"):
            import tempfile
            code_dir = tempfile.mkdtemp(prefix="algotune_agent_")
            os.environ["CODE_DIR"] = code_dir
            print(f"📁 Created temporary CODE_DIR: {code_dir}")
        
        # Get list of tasks
        tasks = args.tasks if args.tasks else []
        
        # If no tasks specified, error out (don't run all tasks by default in standalone)
        if not tasks:
            print("❌ No tasks specified. Please specify task names.")
            print("   Example: algotune.py agent --model o4-mini svm kmeans")
            sys.exit(1)
        
        print(f"📋 Running agent on {len(tasks)} task(s)...")
        
        # Run each task
        for task in tasks:
            print(f"\n🎯 Running task: {task}")
            cmd_args = [
                sys.executable, "-m", "AlgoTuner.main",
                "--model", args.model,
                "--task", task
            ]
            
            result = subprocess.run(cmd_args, cwd=project_root)
            if result.returncode != 0:
                print(f"❌ Task {task} failed with return code {result.returncode}")
                sys.exit(result.returncode)
        
        print("\n✅ All tasks completed")

def run_test_command(args):
    """Run the test suite with dummy LLM."""
    # For tests, we should always use Singularity since that's where all the 
    # specialized dependencies (cvxpy, ortools, sklearn, etc.) are properly installed
    # and where the Python version supports modern syntax
    
    if not getattr(args, 'standalone', False):
        # SLURM mode - create a test submission script using Singularity
        print("🤖 Running tests via SLURM with Singularity")
        submit_test_jobs_slurm(args)
        return
    else:
        # Standalone mode - run via Singularity directly
        print("🐍 Running tests via Singularity")
        run_tests_with_singularity(args)
        return

def submit_test_jobs_slurm(args):
    """Submit test jobs to SLURM for all tasks in tests/inputs."""
    project_root = Path(__file__).parent.parent
    tests_input_dir = project_root / "AlgoTuner/tests/inputs"
    
    if not tests_input_dir.exists():
        print(f"❌ Tests input directory not found: {tests_input_dir}")
        sys.exit(1)
    
    # Get all test input files
    test_files = list(tests_input_dir.glob("*.txt"))
    if not test_files:
        print(f"❌ No test files found in {tests_input_dir}")
        sys.exit(1)
    
    print(f"📋 Found {len(test_files)} test files")
    
    # Clean up old test logs
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        test_log_pattern = "*_dummy_*.log"
        old_logs = list(logs_dir.glob(test_log_pattern))
        if old_logs:
            print(f"🧹 Cleaning up {len(old_logs)} old test logs...")
            for log_file in old_logs:
                try:
                    log_file.unlink()
                except OSError as e:
                    print(f"Warning: Could not delete {log_file}: {e}")
        else:
            print("🧹 No old test logs found to clean up")
    
    # Clean up old SLURM outputs and errors
    slurm_outputs_dir = project_root / "slurm/outputs"
    slurm_errors_dir = project_root / "slurm/errors"
    
    total_cleaned = 0
    for cleanup_dir, desc in [(slurm_outputs_dir, "outputs"), (slurm_errors_dir, "errors")]:
        if cleanup_dir.exists():
            old_slurm_files = list(cleanup_dir.glob("test_*"))
            if old_slurm_files:
                print(f"🧹 Cleaning up {len(old_slurm_files)} old SLURM test {desc}...")
                for file in old_slurm_files:
                    try:
                        file.unlink()
                        total_cleaned += 1
                    except OSError as e:
                        print(f"Warning: Could not delete {file}: {e}")
    
    if total_cleaned == 0:
        print("🧹 No old SLURM test files found to clean up")
    
    # Load configuration to get SLURM partition
    config = {}
    config_file = project_root / "config.env"
    if not config_file.exists():
        config_file = project_root / "slurm/run_config.env"
    
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.replace("export ", "").strip()
                value = value.strip().strip('"').strip("'")
                config[key] = value
    else:
        print("❌ No configuration file found (config.env or slurm/run_config.env)")
        sys.exit(1)
    
    # Get SLURM partition
    slurm_partition = config.get("SLURM_PARTITIONS_DEFAULT", "cpu")
    
    # Create necessary directories
    project_root.joinpath("slurm/outputs").mkdir(parents=True, exist_ok=True)
    project_root.joinpath("slurm/errors").mkdir(parents=True, exist_ok=True)
    project_root.joinpath("AlgoTuner/tests/logs").mkdir(parents=True, exist_ok=True)
    
    # Submit jobs for each test
    job_ids = []
    submitted_tasks = set()  # Track submitted tasks to prevent duplicates
    
    for test_file in test_files:
        try:
            task_name = test_file.stem
            
            # Validate and sanitize task name to prevent corruption
            # Task names should only contain letters, numbers, and underscores
            original_task_name = task_name
            
            # Remove any job ID patterns (numbers at the end)
            task_name = re.sub(r'_\d+$', '', task_name)
            
            # Remove multiple 'test_' prefixes if present
            task_name = re.sub(r'^(test_)+', '', task_name)
            
            # Ensure task name contains only valid characters
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', task_name):
                print(f"❌ Invalid task name: {original_task_name} -> {task_name}")
                continue
                
            # Verify the task exists in the test inputs
            expected_input_file = project_root / "AlgoTuner/tests/inputs" / f"{task_name}.txt"
            if not expected_input_file.exists():
                print(f"❌ No test input file found for task: {task_name} (expected: {expected_input_file})")
                continue
            
            if original_task_name != task_name:
                print(f"🔧 Sanitized task name: {original_task_name} -> {task_name}")
            
            # Skip if we've already submitted this task
            if task_name in submitted_tasks:
                print(f"⚠️  Skipping duplicate submission for {task_name}")
                continue
                
            # Check if there are already running test jobs for this task
            try:
                check_cmd = ["squeue", "-u", os.getenv("USER", ""), "--noheader", "--format", "%j"]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                if check_result.returncode == 0:
                    # Parse job names and check if any test job for this task is running
                    running_jobs = check_result.stdout.strip().split('\n') if check_result.stdout.strip() else []
                    test_job_name = f"test_{task_name}"
                    if any(job_name.strip() == test_job_name for job_name in running_jobs):
                        print(f"⚠️  Test job already running for {task_name}, skipping")
                        continue
            except Exception:
                # If squeue fails, continue with submission
                pass
            
            # Create a temporary script that runs the test via Singularity
            test_script = project_root / f"run_test_{task_name}.sh"
            
            try:
                create_test_script(test_script, task_name, test_file, args.max_samples)
                
                # Submit to SLURM
                cmd = [
                    "sbatch",
                    "--job-name", f"test_{task_name}",
                    "--output", f"slurm/outputs/test_{task_name}_%j.txt",
                    "--error", f"slurm/errors/test_{task_name}_%j.err",
                    "--partition", slurm_partition,
                    "--time", "12:00:00",
                    "--mem", "8G",
                    "--cpus-per-task", "4",
                    str(test_script)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    submitted_tasks.add(task_name)
                    print(f"✅ Submitted test for {task_name}: job {job_id}")
                else:
                    print(f"❌ Failed to submit test for {task_name}: {result.stderr}")
                    
            finally:
                # Clean up temporary script
                if test_script.exists():
                    test_script.unlink()
            
        except Exception as e:
            print(f"❌ Error processing {test_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if job_ids:
        print(f"\n🎯 Submitted {len(job_ids)} test jobs: {' '.join(job_ids)}")
        print("📊 Results will be saved to AlgoTuner/tests/outputs/")
    else:
        print("❌ No tests were successfully submitted")

def create_test_script(script_path: Path, task_name: str, test_file: Path, max_samples: int = None):
    """Create a temporary SLURM script to run a single test."""
    project_root = Path(__file__).parent.parent
    
    # Load config to get the actual temp directory path
    config = {}
    config_file = project_root / "config.env"
    if not config_file.exists():
        config_file = project_root / "slurm/run_config.env"
    
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.replace("export ", "").strip()
                value = value.strip().strip('"').strip("'")
                config[key] = value
    
    # Get the actual temp directory path
    temp_dir_storage = config.get("TEMP_DIR_STORAGE", "/tmp")
    data_dir = config.get("DATA_DIR", f"{project_root}/data")
    singularity_image = config.get("SINGULARITY_IMAGE", "")
    
    # Create the full CODE_DIR path with unique ID
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    code_dir_path = f"{temp_dir_storage}/task_{task_name}_code_{unique_id}"
    
    # Add max_samples to the command if provided
    max_samples_arg = f" --max-samples {max_samples}" if max_samples is not None else ""
    
    script_content = f"""#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -e

# Get project root and load config
PROJECT_ROOT="{project_root}"
cd "$PROJECT_ROOT"

echo "🔧 DEBUG: Current working directory: $(pwd)"
echo "🔧 DEBUG: Contents of current directory:"
ls -la
echo "🔧 DEBUG: Contents of AlgoTuner/tests/:"
ls -la AlgoTuner/tests/
echo "🔧 DEBUG: File permissions for run_tests.py:"
ls -la AlgoTuner/tests/run_tests.py
echo "🔧 DEBUG: First few lines of run_tests.py:"
head -5 AlgoTuner/tests/run_tests.py

# Source configuration
if [ -f "config.env" ]; then
    source config.env
    echo "🔧 DEBUG: Sourced config.env"
elif [ -f "slurm/run_config.env" ]; then
    source slurm/run_config.env
    echo "🔧 DEBUG: Sourced slurm/run_config.env"
else
    echo "❌ No configuration file found"
    exit 1
fi

echo "🔧 DEBUG: SINGULARITY_IMAGE=$SINGULARITY_IMAGE"
echo "🔧 DEBUG: DATA_DIR=$DATA_DIR"
echo "🔧 DEBUG: TEMP_DIR_STORAGE=$TEMP_DIR_STORAGE"

# Create output directory
mkdir -p AlgoTuner/tests/outputs

# Show the exact command we're about to run
echo "🔧 DEBUG: About to run singularity command for {task_name}..."
echo "🔧 DEBUG: Task file: {test_file.relative_to(project_root)}"
echo "🔧 DEBUG: CODE_DIR will be: {code_dir_path}"

# Run test via Singularity with proper quoting
echo "🧪 Running test for {task_name}..."
singularity exec \
    --pwd /app \
    --bind "$PROJECT_ROOT:/app" \
    --bind "{data_dir}:{data_dir}" \
    --bind "{temp_dir_storage}:{temp_dir_storage}" \
    --env PYTHONPATH="/app" \
    --env AGENT_MODE="1" \
    --env CODE_DIR="{code_dir_path}" \
    --env DATA_DIR="{data_dir}" \
    --env TEMP_DIR_STORAGE="{temp_dir_storage}" \
    --env SKIP_DATASET_GEN="1" \
    --env MAX_SAMPLES="{max_samples if max_samples is not None else 10}" \
    "{singularity_image}" \
    bash -c 'mkdir -p "{code_dir_path}" && cd /app && python3 -u /app/AlgoTuner/tests/run_tests.py --model "dummy" --task "{task_name}" --input "/app/{test_file.relative_to(project_root)}"{max_samples_arg}'

echo "✅ Test completed for {task_name}"
"""
    
    script_path.write_text(script_content)
    script_path.chmod(0o755)

def run_tests_with_singularity(args):
    """Run tests via Singularity in standalone mode."""
    project_root = Path(__file__).parent.parent
    tests_input_dir = project_root / "AlgoTuner/tests/inputs"
    
    if not tests_input_dir.exists():
        print(f"❌ Tests input directory not found: {tests_input_dir}")
        sys.exit(1)
    
    # Get all test input files
    test_files = list(tests_input_dir.glob("*.txt"))
    if not test_files:
        print(f"❌ No test files found in {tests_input_dir}")
        sys.exit(1)
    
    print(f"📋 Running {len(test_files)} tests via Singularity...")
    
    # Clean up old test logs
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        test_log_pattern = "*_dummy_*.log"
        old_logs = list(logs_dir.glob(test_log_pattern))
        if old_logs:
            print(f"🧹 Cleaning up {len(old_logs)} old test logs...")
            for log_file in old_logs:
                try:
                    log_file.unlink()
                except OSError as e:
                    print(f"Warning: Could not delete {log_file}: {e}")
        else:
            print("🧹 No old test logs found to clean up")
    
    # Load configuration
    try:
        config_file = project_root / "config.env"
        if not config_file.exists():
            config_file = project_root / "slurm/run_config.env"
        
        if config_file.exists():
            env_vars = {}
            for line in config_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.replace("export ", "").strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
            
            singularity_image = env_vars.get("SINGULARITY_IMAGE")
            data_dir = env_vars.get("DATA_DIR", str(project_root / "data"))
            temp_dir = env_vars.get("TEMP_DIR_STORAGE", "/tmp")
            
            if not singularity_image or not Path(singularity_image).exists():
                print(f"❌ Singularity image not found: {singularity_image}")
                sys.exit(1)
                
        else:
            print("❌ No configuration file found")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = project_root / "AlgoTuner/tests/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    results = []
    for test_file in test_files:
        task_name = test_file.stem
        print(f"🧪 Running test: {task_name}")
        
        try:
            # Add max_samples to the command if provided
            max_samples_arg = f" --max-samples {args.max_samples}" if args.max_samples is not None else ""
            
            # Build command
            cmd = ["singularity", "exec"]
            cmd.extend(["--pwd", "/app"])
            cmd.extend(["--bind", f"{project_root}:/app"])
            cmd.extend(["--bind", f"{data_dir}:{data_dir}"])
            cmd.extend(["--bind", f"{temp_dir}:{temp_dir}"])
            cmd.extend(["--env", "PYTHONPATH=/app"])
            cmd.extend(["--env", "AGENT_MODE=1"])
            cmd.extend(["--env", f"CODE_DIR={temp_dir}/task_{task_name}_code"])
            cmd.extend(["--env", f"DATA_DIR={data_dir}"])
            cmd.extend(["--env", f"TEMP_DIR_STORAGE={temp_dir}"])
            cmd.extend(["--env", f"MAX_SAMPLES={args.max_samples if args.max_samples is not None else 10}"])
            cmd.extend([singularity_image])
            cmd.extend(["bash", "-c"])
            cmd.extend([f"mkdir -p {temp_dir}/task_{task_name}_code && python3 -u /app/AlgoTuner/tests/run_tests.py --model \"dummy\" --task \"{task_name}\" --input \"/app/{test_file.relative_to(project_root)}\"{max_samples_arg}"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✅ {task_name}: PASSED")
                results.append((task_name, True, result.stdout))
            else:
                print(f"❌ {task_name}: FAILED")
                print(f"   Error: {result.stderr}")
                results.append((task_name, False, result.stderr))
                
        except Exception as e:
            print(f"❌ {task_name}: ERROR - {e}")
            results.append((task_name, False, str(e)))
    
    # Generate summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"\n📊 Test Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"test_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write(f"AlgoTune Test Report - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Summary: {passed}/{total} tests passed\n\n")
        
        for task_name, success, output in results:
            status = "PASSED" if success else "FAILED"
            f.write(f"{task_name}: {status}\n")
            if output:
                f.write(f"Output:\n{output}\n")
            f.write("-" * 30 + "\n")
    
    print(f"📄 Detailed report saved to: {report_file}")

def run_tests_directly(args):
    """Run tests directly with available dependencies."""
    import tempfile
    
    project_root = Path(__file__).parent.parent
    
    # Clean up old test logs
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        test_log_pattern = "*_dummy_*.log"
        old_logs = list(logs_dir.glob(test_log_pattern))
        if old_logs:
            print(f"🧹 Cleaning up {len(old_logs)} old test logs...")
            for log_file in old_logs:
                try:
                    log_file.unlink()
                except OSError as e:
                    print(f"Warning: Could not delete {log_file}: {e}")
        else:
            print("🧹 No old test logs found to clean up")
    
    # Set up temporary CODE_DIR for tests (like SLURM mode does)
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    temp_code_dir = tempfile.mkdtemp(prefix=f"algotune_test_code_{unique_id}_")
    print(f"📁 Created temporary CODE_DIR for tests: {temp_code_dir}")
    
    # Change to project directory and run test runner
    original_cwd = Path.cwd()
    original_env = os.environ.copy()
    try:
        os.chdir(project_root)
        
        # Set environment variables for tests
        os.environ["CODE_DIR"] = temp_code_dir
        os.environ["AGENT_MODE"] = "1"
        
        # Import and run the test runner
        sys.path.insert(0, str(project_root))
        from AlgoTuner.tests.test_runner import main as run_test_main
        
        print("🧪 Running tests with dummy LLM...")
        run_test_main()
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(temp_code_dir)
            print(f"🧹 Cleaned up temporary CODE_DIR: {temp_code_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_code_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Timing command
    timing_parser = subparsers.add_parser("timing", help="Run timing evaluations")
    timing_parser.add_argument("--target-time-ms", type=int, help="Target time in milliseconds (optional, uses cached values if not provided)")
    timing_parser.add_argument("--standalone", action="store_true", help="Force standalone mode (no SLURM)")
    timing_parser.add_argument("--sequential", action="store_true", help="Process tasks sequentially")
    timing_parser.add_argument("--task", type=str, help="Single task to run")
    timing_parser.add_argument("--tasks", nargs="+", help="Multiple specific tasks to run")
    timing_parser.add_argument("--task-list", type=str, help="Predefined task list name")
    timing_parser.add_argument("--task-list-file", type=Path, help="File containing task list")
    timing_parser.add_argument("--data-dir", type=Path, help="Override data directory")
    timing_parser.add_argument("--lazy", action="store_true", help="Only generate datasets for explicitly requested tasks")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run test suite with dummy LLM")
    test_parser.add_argument("--standalone", action="store_true", help="Force standalone mode (no SLURM)")
    test_parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to evaluate")
    
    # Single command
    single_parser = subparsers.add_parser("single", help="Run single task evaluation")
    single_parser.add_argument("--task", type=str, required=True, help="Task to run")
    single_parser.add_argument("--target-time-ms", type=int, default=50, help="Target time in milliseconds")
    single_parser.add_argument("--data-dir", type=Path, help="Override data directory")
    
    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run AI agent on tasks")
    agent_parser.add_argument("--model", type=str, required=True, help="Model name to use")
    agent_parser.add_argument("--standalone", action="store_true", help="Force standalone mode (no SLURM)")
    agent_parser.add_argument("tasks", nargs="*", help="Specific tasks to run (all if none specified)")
    
    # List commands
    subparsers.add_parser("list-tasks", help="List available tasks")
    subparsers.add_parser("list-task-lists", help="List available task lists")

    args = parser.parse_args()
    
    if not args.command:
        # Auto-detect mode for backward compatibility
        auto_detect_and_run(args)
    elif args.command == "timing":
        run_timing_command(args)
    elif args.command == "test":
        run_test_command(args)
    elif args.command == "single":
        run_single_command(args)
    elif args.command == "agent":
        run_agent_command(args)
    elif args.command == "list-tasks":
        list_tasks_command(args)
    elif args.command == "list-task-lists":
        list_task_lists_command(args)


if __name__ == "__main__":
    main()