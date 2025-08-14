#!/usr/bin/env python3
"""
Evaluate results code against baselines with robust compilation handling.

This script reads generation.json for baseline timings, discovers models from 
the ./results directory, runs both baseline and optimized code on test datasets,
and generates evaluate_summary.json with speedup results.

Features:
- Robust compilation handling (Cython, Pythran, DaCe, C extensions)
- Process isolation for clean benchmarks
- Automatic model discovery from results directory
- Test dataset evaluation with proper baseline comparison
- Error handling and graceful degradation
"""

import argparse
import json
import os
import sys
import subprocess
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Only import what we need for argument parsing and basic functionality
# Heavy imports will be done inside functions that need them


@dataclass
class EvaluationResult:
    """Result of evaluating a single model on a task."""
    task_name: str
    model_name: str
    display_model_name: str
    baseline_time_ms: Optional[float]
    optimized_time_ms: Optional[float]
    speedup: Optional[float]
    success: bool
    error_message: Optional[str] = None
    compilation_needed: bool = False
    compilation_success: bool = True


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def normalize_model_name(model_name: str) -> str:
    """Normalize model names to match trajectories_to_html display names."""
    model_display_names = {
        "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
        "anthropic/claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
        "anthropic/claude-3-haiku": "Claude 3 Haiku",
        "anthropic/claude-3-opus": "Claude 3 Opus",
        "anthropic/claude-3.5-haiku": "Claude 3.5 Haiku",
        "gemini/gemini-1.5-pro": "Gemini 1.5 Pro",
        "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "DeepSeek-R1": "DeepSeek R1",
        "deepseek-ai/DeepSeek-R1": "DeepSeek R1",
        "claude-opus-4-20250514": "Claude Opus 4",
        "o4-mini": "o4-mini",
        "deepseek/deepseek-reasoner": "DeepSeek R1",
        "deepseek-reasoner": "DeepSeek R1",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4o": "GPT-4o",
        "gpt-5": "GPT-5",
        "gpt-5-mini": "GPT-5 Mini",
    }
    return model_display_names.get(model_name, model_name)


def discover_models_and_tasks(results_dir: Path) -> Dict[str, List[str]]:
    """Discover available models and their tasks from results directory."""
    models_tasks = {}
    
    if not results_dir.exists():
        logging.warning(f"Results directory does not exist: {results_dir}")
        return models_tasks
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        tasks = []
        
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            # Check if task has solver.py or other code files
            code_files = list(task_dir.glob("*.py")) + list(task_dir.glob("*.pyx")) + list(task_dir.glob("*.c"))
            if code_files:
                tasks.append(task_dir.name)
        
        if tasks:
            models_tasks[model_name] = sorted(tasks)
    
    return models_tasks


def load_generation_data(generation_file: Path) -> Dict[str, Dict]:
    """Load baseline timing data from generation.json."""
    try:
        with open(generation_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load generation data from {generation_file}: {e}")
        return {}


def filter_used_files(code_dir: Path) -> List[str]:
    """
    Return only files that are actually used by solver.py (same logic as trajectories_to_html.py).
    
    Starts with solver.py and recursively finds all imported files to avoid
    compiling unused artifacts like leftover Cython files.
    """
    import re
    
    # Check if solver.py exists
    solver_path = code_dir / "solver.py"
    if not solver_path.exists():
        return []
    
    # Get all files in directory  
    all_files = {f.name: f.read_text(encoding='utf-8', errors='ignore') 
                 for f in code_dir.iterdir() if f.is_file()}
    
    if "solver.py" not in all_files:
        return []
    
    import_pattern = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)", re.MULTILINE)
    
    # Extensions we consider as source files for a Python import
    _SOURCE_EXTS = (".py", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp")
    
    used_files = {"solver.py"}
    queue = ["solver.py"]
    
    while queue:
        fname = queue.pop()
        code_text = all_files.get(fname, "")
        if not isinstance(code_text, str):
            continue
        
        # Look for top-level import statements and map them to candidate files
        for match in import_pattern.finditer(code_text):
            module_path = match.group(1).lstrip(".")  # strip any relative dots
            base_mod = module_path.split(".")[0]
            for ext in _SOURCE_EXTS:
                candidate = f"{base_mod}{ext}"
                if candidate in all_files and candidate not in used_files:
                    used_files.add(candidate)
                    queue.append(candidate)
    
    # Include setup.py only if we detected any non-Python source files (Cython/C/C++)
    compilation_exts = (".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp")
    needs_setup = any(fname.endswith(compilation_exts) for fname in used_files)
    if needs_setup and "setup.py" in all_files:
        used_files.add("setup.py")
    
    return list(used_files)


def needs_compilation(code_dir: Path) -> Tuple[bool, List[str]]:
    """Check if code directory contains files that need compilation (only for actually used files)."""
    # First filter to only used files
    used_files = filter_used_files(code_dir)
    if not used_files:
        return False, []
    
    compilation_indicators = []
    
    # Check for setup.py or pyproject.toml (only if they're actually used)
    if "setup.py" in used_files and (code_dir / "setup.py").exists():
        compilation_indicators.append("setup.py")
    if "pyproject.toml" in used_files and (code_dir / "pyproject.toml").exists():
        compilation_indicators.append("pyproject.toml")
    
    # Check for Cython files (only used ones)
    for fname in used_files:
        if fname.endswith(('.pyx', '.pxd')):
            if (code_dir / fname).exists():
                compilation_indicators.append(fname)
    
    # Check for Pythran files (contains "pythran export") - only used ones
    for fname in used_files:
        if fname.endswith('.py'):
            py_file = code_dir / fname
            if py_file.exists():
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    if "pythran export" in content.lower():
                        compilation_indicators.append(f"{fname} (Pythran)")
                except Exception:
                    pass
    
    # Check for DaCe files (contains "@dace.program") - only used ones
    for fname in used_files:
        if fname.endswith('.py'):
            py_file = code_dir / fname
            if py_file.exists():
                try:
                    content = py_file.read_text(encoding='utf-8', errors='ignore')
                    if "@dace.program" in content:
                        compilation_indicators.append(f"{fname} (DaCe)")
                except Exception:
                    pass
    
    return len(compilation_indicators) > 0, compilation_indicators




def compile_code_if_needed(source_code_dir: Path, task_name: str) -> Tuple[bool, str]:
    """
    Compile code if needed using simplified compilation logic.
    
    For evaluation, we compile in-place since we're working with a copy in results/
    The evaluation runs in isolated processes anyway.
    """
    needs_comp, indicators = needs_compilation(source_code_dir)
    if not needs_comp:
        return True, "No compilation needed"
    
    logging.info(f"Compilation needed for {task_name}: {indicators}")
    
    try:
        # Handle setup.py/pyproject.toml
        if (source_code_dir / "setup.py").exists() or (source_code_dir / "pyproject.toml").exists():
            logging.info(f"Running pip install for {task_name}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", ".", "--no-deps", "--force-reinstall", "--no-cache-dir"],
                cwd=source_code_dir,
                timeout=1800,  # 30 minutes
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return False, f"Setup install failed: {result.stderr}"
            logging.info(f"Setup install successful for {task_name}")
            
        # Handle Pythran files  
        for py_file in source_code_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if "pythran export" in content.lower():
                    logging.info(f"Compiling Pythran file: {py_file}")
                    result = subprocess.run(
                        ["pythran", "-O3", "-march=native", str(py_file)],
                        cwd=source_code_dir,
                        timeout=300,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        logging.warning(f"Pythran compilation failed: {result.stderr}")
                        # Non-fatal - continue with other files
            except Exception as e:
                logging.warning(f"Error checking/compiling Pythran file {py_file}: {e}")
        
        # Handle DaCe files
        for py_file in source_code_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if "@dace.program" in content:
                    logging.info(f"DaCe file detected: {py_file}")
                    # DaCe compiles JIT, but we can pre-compile by importing
                    # This ensures any compilation errors are caught early
                    import_name = py_file.stem
                    logging.info(f"Pre-compiling DaCe module: {import_name}")
                    # Add the directory to sys.path temporarily
                    sys.path.insert(0, str(source_code_dir))
                    try:
                        __import__(import_name)
                        logging.info(f"DaCe module {import_name} loaded successfully")
                    except Exception as e:
                        logging.warning(f"DaCe pre-compilation warning: {e}")
                        # Non-fatal - DaCe will compile JIT during execution
                    finally:
                        sys.path.pop(0)
            except Exception as e:
                logging.warning(f"Error checking DaCe file {py_file}: {e}")
                
        return True, "Compilation completed"
        
    except Exception as e:
        logging.error(f"Compilation error for {task_name}: {e}")
        return False, f"Compilation error: {e}"


class AgentCompatibleEvaluator:
    """Evaluator that uses identical methodology to the agent."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def evaluate_task(self, task_name: str, model_name: str, code_dir: Path) -> EvaluationResult:
        """Single task evaluation using agent's exact pipeline."""
        display_model_name = normalize_model_name(model_name)
        
        result = EvaluationResult(
            task_name=task_name,
            model_name=model_name,
            display_model_name=display_model_name,
            baseline_time_ms=None,
            optimized_time_ms=None,
            speedup=None,
            success=False
        )
        
        try:
            # Step 1: Create task instance (same as agent)
            from AlgoTuneTasks.factory import TaskFactory
            from AlgoTuner.config.loader import load_config
            
            config = load_config()
            task_config = config.get('tasks', {}).get(task_name, {})
            oracle_time_limit = task_config.get('oracle_time_limit')
            
            task_instance = TaskFactory(task_name, oracle_time_limit=oracle_time_limit, data_dir=str(self.data_dir))
            
            # Step 2: Create BaselineManager and generate baseline times
            from AlgoTuner.utils.evaluator.baseline_manager import BaselineManager
            
            baseline_manager = BaselineManager(task_instance)
            logging.info(f"Created BaselineManager for {task_name}")
            
            # Generate baseline times
            old_skip_gen = os.environ.get('SKIP_DATASET_GEN')
            try:
                os.environ['SKIP_DATASET_GEN'] = '1'
                baseline_times_raw = baseline_manager.get_baseline_times(
                    subset="test", 
                    force_regenerate=False,  # Don't force regeneration - use cached if available
                    test_mode=False,
                    max_samples=None
                )
                logging.info(f"Generated {len(baseline_times_raw)} baseline times with keys: {list(baseline_times_raw.keys())[:5]}")
            finally:
                if old_skip_gen is None:
                    os.environ.pop('SKIP_DATASET_GEN', None)
                else:
                    os.environ['SKIP_DATASET_GEN'] = old_skip_gen
            
            # Step 3: Load optimized solver from code_dir
            sys.path.insert(0, str(code_dir))
            optimized_solve = None
            try:
                # Try to import solve function directly
                from solver import solve as optimized_solve
            except ImportError:
                try:
                    # Try to import Solver class and instantiate it
                    from solver import Solver
                    solver_instance = Solver()
                    optimized_solve = solver_instance.solve
                except ImportError as e:
                    result.error_message = f"Failed to import optimized solver: {e}"
                    return result
            finally:
                sys.path.pop(0)
            
            if optimized_solve is None:
                result.error_message = "Could not load optimized solver"
                return result
            
            # Step 4: Load test dataset and transform baseline times to index-based keys
            old_skip_gen = os.environ.get('SKIP_DATASET_GEN')
            try:
                os.environ['SKIP_DATASET_GEN'] = '1'
                _, test_iter = task_instance.load_dataset()
                test_problems = list(test_iter)
                
                if not test_problems:
                    result.error_message = "No test problems found"
                    return result
                    
                logging.info(f"Loaded {len(test_problems)} test problems for {task_name}")
                
            finally:
                if old_skip_gen is None:
                    os.environ.pop('SKIP_DATASET_GEN', None)
                else:
                    os.environ['SKIP_DATASET_GEN'] = old_skip_gen
            
            # Step 5: Create temporary directory and copy code (like agent does)
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp(prefix=f"eval_{task_name}_")
            temp_code_dir = Path(temp_dir)
            
            try:
                # Copy all code files to temporary directory
                logging.info(f"Copying code from {code_dir} to temporary CODE_DIR: {temp_code_dir}")
                shutil.copytree(code_dir, temp_code_dir, dirs_exist_ok=True)
                
                # Check if compilation is needed and compile in the temp directory
                needs_comp, indicators = needs_compilation(temp_code_dir)
                
                
                if needs_comp:
                    logging.info(f"Compiling {task_name} in temporary directory: {indicators}")
                    comp_success, comp_message = compile_code_if_needed(temp_code_dir, task_name)
                    if not comp_success:
                        result.error_message = f"Compilation failed: {comp_message}"
                        result.compilation_needed = True
                        result.compilation_successful = False
                        return result
                    result.compilation_needed = True
                    
                
                # Step 6: Evaluate optimized code using same function as agent
                from AlgoTuner.utils.evaluator.main import evaluate_code_on_dataset
                
                # Set CODE_DIR environment variable to point to the temporary directory
                os.environ['CODE_DIR'] = str(temp_code_dir)
                logging.info(f"CODE_DIR set to temporary directory: {temp_code_dir}")
                
                # Set AGENT_MODE to ensure isolated execution (same as agent)
                old_agent_mode = os.environ.get('AGENT_MODE')
                os.environ['AGENT_MODE'] = '1'
                os.environ['ISOLATED_EVAL'] = '1'
                logging.info("Set AGENT_MODE=1 and ISOLATED_EVAL=1 for consistent timing with agent")
            
                logging.info(f"Running optimized evaluation for {task_name} using evaluate_code_on_dataset")
                
                # Pre-generate baseline times and transform them to index-based keys
                baseline_times_raw = baseline_manager.get_baseline_times(
                    subset="test",
                    force_regenerate=False,
                    test_mode=False
                )
                logging.info(f"Got {len(baseline_times_raw)} baseline times from BaselineManager")
                
                # No transformation needed - main.py uses the same key extraction logic as BaselineManager
                # Both use dataset IDs (e.g., "142", "143") with fallback to index-based keys
                logging.info(f"Using raw baseline times from BaselineManager: {len(baseline_times_raw)} entries")
                
                # evaluate_code_on_dataset will:
                # 1. Load and evaluate the optimized solver from CODE_DIR
                # 2. Use the pre-generated baseline times we provide
                results = evaluate_code_on_dataset(
                    task_obj=task_instance,
                    dataset_iterable=test_problems,
                    baseline_times=baseline_times_raw,  # Pass raw baseline times with dataset ID keys
                    data_subset="test",
                    test_mode=False
                )
                
                # Step 6: Extract speedups from results (same format as agent)
                logging.info(f"Results type: {type(results)}, has aggregate_metrics: {hasattr(results, 'aggregate_metrics')}")
                
                # Check if we have aggregate metrics from the agent evaluation
                if hasattr(results, 'aggregate_metrics') and results.aggregate_metrics:
                    agg = results.aggregate_metrics
                    logging.info(f"Agent aggregate metrics: mean_speedup={getattr(agg, 'mean_speedup', 'N/A')}, num_valid={getattr(agg, 'num_valid', 'N/A')}, num_evaluated={getattr(agg, 'num_evaluated', 'N/A')}")
                
                if isinstance(results, list) and results:
                    # Log first result to understand format
                    if results:
                        logging.info(f"First result keys: {list(results[0].keys())}")
                        logging.info(f"First result sample: success={results[0].get('success')}, is_valid={results[0].get('is_valid')}, baseline_time_ms={results[0].get('baseline_time_ms')}, solver_min_time_ms={results[0].get('solver_min_time_ms')}, min_time_ms={results[0].get('min_time_ms')}")
                    
                    # Calculate aggregate metrics from individual results
                    valid_speedups = []
                    total_baseline_time = 0
                    total_optimized_time = 0
                    
                    critical_error_found = False
                    for res in results:
                        # Check for critical errors that should stop evaluation
                        error_type = res.get('error_type')
                        if error_type in ['benchmark_error', 'solver_exception', 'runtime_error', 'import_error', 'memory_error']:
                            critical_error_found = True
                            logging.error(f"Critical error found: {error_type}")
                            result.error_message = f"Critical error: {error_type}"
                            break
                        
                        if res.get('success', False) and res.get('is_valid', False):
                            # Try different field names that might contain timing info
                            baseline_ms = res.get('baseline_time_ms')
                            optimized_ms = (res.get('solver_min_time_ms') or 
                                          res.get('min_time_ms') or 
                                          res.get('optimized_time_ms'))
                            
                            if baseline_ms and optimized_ms and baseline_ms > 0 and optimized_ms > 0:
                                speedup = baseline_ms / optimized_ms
                                valid_speedups.append(speedup)
                                total_baseline_time += baseline_ms
                                total_optimized_time += optimized_ms
                                if len(valid_speedups) <= 3:  # Log first few
                                    logging.info(f"Found valid speedup: {baseline_ms:.3f}ms / {optimized_ms:.3f}ms = {speedup:.4f}x")
                    
                    # Count total valid results
                    total_evaluated = len(results)
                    num_valid = len(valid_speedups)
                    success_rate = (num_valid / total_evaluated * 100) if total_evaluated > 0 else 0
                    
                    logging.info(f"Validation stats: {num_valid}/{total_evaluated} valid ({success_rate:.1f}%)")
                    
                    # Agent evaluation requires 100% validity for speedup
                    if critical_error_found:
                        # Critical errors mean no speedup regardless of partial results
                        logging.info(f"❌ {task_name}/{display_model_name}: No speedup due to critical error")
                    elif valid_speedups and success_rate == 100:
                        import statistics
                        overall_speedup = statistics.mean(valid_speedups)
                        avg_baseline_time = total_baseline_time / len(valid_speedups)
                        avg_optimized_time = total_optimized_time / len(valid_speedups)
                        
                        result.baseline_time_ms = avg_baseline_time
                        result.optimized_time_ms = avg_optimized_time
                        result.speedup = overall_speedup
                        result.success = True
                        
                        logging.info(f"✅ {task_name}/{display_model_name}: {overall_speedup:.4f}x speedup ({avg_baseline_time:.3f}ms → {avg_optimized_time:.3f}ms) across {len(valid_speedups)} problems")
                        logging.info(f"Speedup calculation details: valid_speedups={len(valid_speedups)}, min={min(valid_speedups):.4f}, max={max(valid_speedups):.4f}, mean={overall_speedup:.4f}, median={statistics.median(valid_speedups):.4f}")
                    elif valid_speedups and success_rate < 100:
                        result.error_message = f"Speedup N/A due to invalid results: {num_valid}/{total_evaluated} valid ({success_rate:.1f}%)"
                        logging.info(f"❌ {task_name}/{display_model_name}: Speedup N/A - only {success_rate:.1f}% valid results")
                        
                        # Check if we have aggregate metrics to compare
                        if hasattr(results, 'aggregate_metrics') and results.aggregate_metrics:
                            agent_mean_speedup = results.aggregate_metrics.get('mean_speedup') if isinstance(results.aggregate_metrics, dict) else getattr(results.aggregate_metrics, 'mean_speedup', None)
                            if agent_mean_speedup is not None:
                                logging.info(f"Comparison: Agent's mean_speedup={agent_mean_speedup:.4f} vs our calculation={overall_speedup:.4f}, difference={(agent_mean_speedup - overall_speedup):.4f}")
                    else:
                        result.error_message = "No valid speedup calculations from agent evaluation"
                else:
                    result.error_message = f"Unexpected results format from evaluate_code_on_dataset: {type(results)}"
                    
            finally:
                # Clean up temporary directory
                logging.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            result.error_message = f"Agent-compatible evaluation error: {str(e)}"
            logging.error(f"❌ {task_name}/{display_model_name}: {e}")
            logging.debug(traceback.format_exc())
        
        return result




def evaluate_single_task(args_tuple: Tuple) -> EvaluationResult:
    """Evaluate a single task for a model using agent-compatible evaluation."""
    (task_name, model_name, display_model_name, code_dir_str, generation_data, data_dir_str, num_runs) = args_tuple
    
    code_dir = Path(code_dir_str)
    data_dir = Path(data_dir_str)
    
    logging.info(f"Evaluating {task_name} for {display_model_name}")
    
    # Compilation will be handled inside AgentCompatibleEvaluator in the temporary directory
    
    # Use AgentCompatibleEvaluator for the actual evaluation
    evaluator = AgentCompatibleEvaluator(data_dir)
    result = evaluator.evaluate_task(task_name, model_name, code_dir)
    
    # Compilation flags are already set by AgentCompatibleEvaluator
    
    return result


def handle_slurm_mode(args) -> None:
    """Handle SLURM mode - discover models/tasks and submit jobs."""
    import subprocess
    import logging
    
    # Setup basic logging for SLURM mode
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load SLURM configuration from config.env
    config_env = Path(__file__).parent.parent / "config.env"
    if config_env.exists():
        logging.info(f"Loading SLURM config from: {config_env}")
        result = subprocess.run(['bash', '-c', f'source {config_env} && env'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '=' in line and any(var in line for var in ['SLURM_', 'SINGULARITY_', 'TEMP_DIR_', 'DATA_DIR']):
                key, value = line.split('=', 1)
                os.environ[key] = value
                logging.info(f"Set {key}={value}")
    
    # Set default data_dir if not provided (same logic as main())
    if args.data_dir is None:
        args.data_dir = Path(os.environ.get('DATA_DIR', project_root.parent / 'data'))
    
    # Discover models and tasks (simple version without full imports)
    models_tasks = discover_models_and_tasks_simple(args.results_dir, args.models, args.tasks)
    if not models_tasks:
        logging.error("No models/tasks found")
        sys.exit(1)
    
    submit_slurm_evaluation_jobs(models_tasks, args)


def discover_models_and_tasks_simple(results_dir: Path, filter_models: List[str] = None, filter_tasks: List[str] = None) -> Dict[str, List[str]]:
    """Simple model/task discovery without heavy imports."""
    models_tasks = {}
    
    if not results_dir.exists():
        return models_tasks
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        if filter_models and model_name not in filter_models:
            continue
            
        tasks = []
        for task_dir in model_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                # Filter by task if specified
                if filter_tasks and task_name not in filter_tasks:
                    continue
                # Simple check - if directory exists, assume it's a valid task
                tasks.append(task_name)
        
        if tasks:
            models_tasks[model_name] = sorted(tasks)
    
    return models_tasks


def submit_slurm_evaluation_jobs(models_tasks: Dict[str, List[str]], args) -> None:
    """Submit SLURM array jobs for evaluation, similar to submit_agent.sh."""
    import subprocess
    
    # Create evaluation task list
    eval_tasks = []
    for model_name, tasks in models_tasks.items():
        for task_name in tasks:
            eval_tasks.append((model_name, task_name))
    
    logging.info(f"Submitting {len(eval_tasks)} evaluation jobs to SLURM")
    
    # Create temporary directory for task file
    tmp_dir = Path("tmp/eval_tasks")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create array task file
    task_file = tmp_dir / "eval_tasks.txt"
    with open(task_file, 'w') as f:
        for model_name, task_name in eval_tasks:
            f.write(f"{model_name}\t{task_name}\n")
    
    # Submit SLURM array job
    slurm_script = Path(__file__).parent / "slurm_jobs" / "evaluate.sh"
    if not slurm_script.exists():
        logging.error(f"SLURM evaluation script not found: {slurm_script}")
        sys.exit(1)
    
    env_vars = {
        "EVAL_RESULTS_DIR": str(args.results_dir),
        "EVAL_GENERATION_FILE": str(args.generation_file),
        "EVAL_OUTPUT_FILE": str(args.output),
        "EVAL_NUM_RUNS": str(args.num_runs),
        "EVAL_DATA_DIR": str(args.data_dir)
    }
    
    cmd = [
        "sbatch",
        f"--array=1-{len(eval_tasks)}",
        "--job-name=algotune-eval",
        f"--partition={os.environ.get('SLURM_PARTITIONS_DEFAULT', 'cpu')}",
        "--time=47:59:00",
        "--output=slurm/outputs/eval-%A_%a.out",
        "--error=slurm/errors/eval-%A_%a.err",
        f"--export=ALL,EVAL_TASK_FILE={task_file}," + ",".join(f"{k}={v}" for k, v in env_vars.items()),
        str(slurm_script)
    ]
    
    logging.info(f"SLURM command: {' '.join(cmd)}")
    logging.info(f"Task file created: {task_file}")
    logging.info(f"Environment variables: {env_vars}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        logging.info(f"✅ Submitted SLURM array job: {job_id}")
        logging.info(f"Monitor with: squeue -j {job_id}")
        logging.info(f"Results will be written directly to: {args.output}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to submit SLURM jobs: {e}")
        logging.error(f"Command: {' '.join(cmd)}")
        logging.error(f"Error output: {e.stderr}")
        logging.error(f"STDOUT: {e.stdout}")
        sys.exit(1)


def calculate_harmonic_mean(speedups: List[float]) -> float:
    """Calculate harmonic mean of speedups, following stats/generate_speedup_hmean_table.py logic."""
    if not speedups:
        return 1.0
    denom = 0.0
    for v in speedups:
        # Avoid division by zero; treat extremely small numbers as epsilon
        denom += 1.0 / max(v, 1e-12)
    return len(speedups) / denom


def update_single_result(result: EvaluationResult, summary_file: Path) -> None:
    """Update evaluation summary file with a single result, using file locking."""
    import fcntl
    import time
    import random
    
    lock_file = summary_file.with_suffix('.json.lock')
    max_attempts = 60
    
    for attempt in range(max_attempts):
        try:
            # Try to create lock file
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            
            try:
                # Load existing summary
                summary_data = {}
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary_data = json.load(f)
                    except:
                        summary_data = {}
                
                # Update with this result
                if result.task_name not in summary_data:
                    summary_data[result.task_name] = {}
                
                if result.success and result.speedup is not None:
                    speedup_str = f"{result.speedup:.4f}"
                else:
                    speedup_str = "N/A"
                
                summary_data[result.task_name][result.display_model_name] = {
                    "final_speedup": speedup_str
                }
                
                # Write back
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                    
                logging.info(f"Updated summary for {result.task_name}/{result.display_model_name}: {speedup_str}")
                return
                
            finally:
                # Release lock
                try:
                    os.remove(str(lock_file))
                except:
                    pass
                    
        except FileExistsError:
            # Lock exists, wait and retry
            time.sleep(random.uniform(0.1, 0.5))
            continue
            
    logging.error(f"Failed to acquire lock after {max_attempts} attempts")


def update_agent_summary(results: List[EvaluationResult], summary_file: Path, generation_data: Dict) -> None:
    """Update evaluation summary file with evaluation results and harmonic means."""
    # Load existing summary or create new one
    summary_data = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning(f"Could not load existing summary file {summary_file}, creating new one")
            summary_data = {}
    
    all_tasks = set(generation_data.keys())
    total_tasks = len(all_tasks)
    logging.info(f"Total expected tasks: {total_tasks}")
    
    # Group results by model
    model_results = {}
    for result in results:
        if result.model_name not in model_results:
            model_results[result.model_name] = {}
        model_results[result.model_name][result.task_name] = result
    
    # Calculate harmonic mean for each model
    model_harmonic_means = {}
    
    for model_name, task_results in model_results.items():
        display_name = normalize_model_name(model_name)
        speedups = []
        
        for task_name in all_tasks:
            if task_name in task_results:
                result = task_results[task_name]
                if result.success and result.speedup is not None:
                    # Cap speedup below 1.0 to 1.0 (following stats script logic)
                    speedups.append(max(result.speedup, 1.0))
                else:
                    # Failed or N/A tasks count as 1x speedup
                    speedups.append(1.0)
            else:
                # Missing tasks count as 1x speedup
                speedups.append(1.0)
        
        # Calculate harmonic mean
        harmonic_mean = calculate_harmonic_mean(speedups)
        model_harmonic_means[model_name] = harmonic_mean
        
        completed_tasks = len([task_results[t] for t in task_results if task_results[t].success])
        logging.info(f"{display_name}: Harmonic mean = {harmonic_mean:.4f}x "
                    f"(completed {completed_tasks}/{total_tasks} tasks)")
    
    # Create ordered dictionary with harmonic means first
    from collections import OrderedDict
    ordered_summary = OrderedDict()
    
    # Add harmonic means section first (sorted by score descending)
    sorted_models = sorted(model_harmonic_means.items(), key=lambda x: (-x[1], x[0]))
    ordered_summary["_overall_scores"] = OrderedDict()
    for model_name, mean in sorted_models:
        ordered_summary["_overall_scores"][normalize_model_name(model_name)] = {
            "harmonic_mean": f"{mean:.4f}",
            "model_name": model_name
        }
    
    # Then add individual task results
    for result in results:
        if result.task_name not in ordered_summary:
            ordered_summary[result.task_name] = {}
        
        if result.success and result.speedup is not None:
            speedup_str = f"{result.speedup:.4f}"
        else:
            speedup_str = "N/A"
        
        ordered_summary[result.task_name][result.model_name] = {
            "final_speedup": speedup_str
        }
    
    # Write ordered summary
    summary_file.parent.mkdir(exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(ordered_summary, f, indent=2)
    
    logging.info(f"Updated evaluation summary: {summary_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate results code against baselines")
    parser.add_argument("--results-dir", type=Path, default="results", 
                       help="Results directory (default: results)")
    parser.add_argument("--generation-file", type=Path, default="reports/generation.json",
                       help="Generation file with baseline timings (default: reports/generation.json)")
    parser.add_argument("--data-dir", type=Path, 
                       help="Data directory (default: from environment or ../data)")
    parser.add_argument("--output", type=Path, default="reports/evaluate_summary.json",
                       help="Output file (default: reports/evaluate_summary.json)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of benchmark runs (default: 10, same as agent mode)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum worker processes (default: 4)")
    parser.add_argument("--models", nargs="+",
                       help="Specific models to evaluate (default: all)")
    parser.add_argument("--tasks", nargs="+",
                       help="Specific tasks to evaluate (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    parser.add_argument("--slurm", action="store_true",
                       help="Run in SLURM array mode")
    
    args = parser.parse_args()
    
    # Handle SLURM mode early - just submit jobs and exit
    if args.slurm:
        print("Entering SLURM mode...")
        handle_slurm_mode(args)
        print("SLURM submission complete.")
        return
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Set up paths
    if args.data_dir is None:
        args.data_dir = Path(os.environ.get('DATA_DIR', project_root.parent / 'data'))
    
    # Don't check if data_dir exists here - the Task class will handle finding
    # task-specific subdirectories and will provide better error messages
    logging.info(f"Using data directory: {args.data_dir}")
    
    # Debug: Check if we can see the data directory
    import subprocess
    try:
        result = subprocess.run(['ls', '-la', str(args.data_dir)], capture_output=True, text=True, timeout=5)
        logging.info(f"Data directory listing (first 10 lines):")
        for line in result.stdout.split('\n')[:10]:
            logging.info(f"  {line}")
    except Exception as e:
        logging.warning(f"Could not list data directory: {e}")
    
    if not args.generation_file.exists():
        logging.error(f"Generation file does not exist: {args.generation_file}")
        sys.exit(1)
    
    logging.info(f"Starting evaluation with results_dir={args.results_dir}, data_dir={args.data_dir}")
    
    # Load generation data
    generation_data = load_generation_data(args.generation_file)
    if not generation_data:
        logging.error("No generation data loaded")
        sys.exit(1)
    
    logging.info(f"Loaded generation data for {len(generation_data)} tasks")
    
    # Discover models and tasks
    models_tasks = discover_models_and_tasks(args.results_dir)
    if not models_tasks:
        logging.error(f"No models found in results directory: {args.results_dir}")
        sys.exit(1)
    
    logging.info(f"Discovered {len(models_tasks)} models: {list(models_tasks.keys())}")
    
    # Filter models and tasks
    if args.models:
        models_tasks = {k: v for k, v in models_tasks.items() if k in args.models}
    
    if args.tasks:
        for model in models_tasks:
            models_tasks[model] = [t for t in models_tasks[model] if t in args.tasks]
        # Remove models with no tasks
        models_tasks = {k: v for k, v in models_tasks.items() if v}
    
    # Filter to only tasks with generation data
    for model in models_tasks:
        models_tasks[model] = [t for t in models_tasks[model] if t in generation_data]
    models_tasks = {k: v for k, v in models_tasks.items() if v}
    
    if not models_tasks:
        logging.error("No valid model/task combinations found")
        sys.exit(1)
    
    # Handle SLURM mode
    if args.slurm:
        logging.info("Running in SLURM array mode - submitting jobs")
        submit_slurm_evaluation_jobs(models_tasks, args)
        return
    
    # Prepare evaluation tasks
    eval_tasks = []
    for model_name, tasks in models_tasks.items():
        display_model_name = normalize_model_name(model_name)
        for task_name in tasks:
            code_dir = args.results_dir / model_name / task_name
            eval_tasks.append((
                task_name,
                model_name,
                display_model_name,
                str(code_dir),
                generation_data,
                str(args.data_dir),
                args.num_runs
            ))
    
    logging.info(f"Evaluating {len(eval_tasks)} model/task combinations")
    
    # Run evaluations
    results = []
    if args.max_workers == 1:
        # Sequential execution for debugging
        for task_args in eval_tasks:
            results.append(evaluate_single_task(task_args))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {executor.submit(evaluate_single_task, task_args): task_args for task_args in eval_tasks}
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task_args = future_to_task[future]
                    logging.error(f"Evaluation failed for {task_args[0]}/{task_args[2]}: {e}")
    
    # Summary statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    compiled = [r for r in results if r.compilation_needed]
    
    logging.info(f"Evaluation complete: {len(successful)}/{len(results)} successful")
    logging.info(f"Compilation needed: {len(compiled)} tasks")
    
    if successful:
        speedups = [r.speedup for r in successful if r.speedup is not None]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            logging.info(f"Average speedup: {avg_speedup:.4f}x")
    
    if failed:
        logging.warning(f"Failed evaluations: {len(failed)}")
        for result in failed[:5]:  # Show first 5 failures
            logging.warning(f"  {result.task_name}/{result.display_model_name}: {result.error_message}")
    
    # Update summary - use single result update for SLURM array jobs
    if len(results) == 1:
        # Single task evaluation (typical in SLURM array job)
        update_single_result(results[0], args.output)
    else:
        # Multiple tasks - calculate harmonic means
        update_agent_summary(results, args.output, generation_data)
    
    logging.info(f"Results written to: {args.output}")


if __name__ == "__main__":
    # CRITICAL: Initialize multiprocessing support before anything else
    # This prevents issues when using ProcessPoolExecutor
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set the multiprocessing start method early for consistency
    try:
        multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass
    
    # Set NUMBA threading layer for fork safety
    if "NUMBA_THREADING_LAYER" not in os.environ:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    
    main()