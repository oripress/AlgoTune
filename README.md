<p align="center">
  <a href="https://algotune.io/">
    <img
      src="https://github.com/oripress/AlgoTune/blob/main/assets/algotune_title_orange.png?raw=true"
      alt="AlgoTune banner"
      width="800"
    />
  </a>
</p>
<p align="center">
  <a href="https://algotune.io/"><strong>Website</strong></a>&nbsp; | &nbsp;
  <a href="https://algotune.io/paper"><strong>Paper</strong></a>
</p>

How good are language models at coming up with new algorithms? To try to answer this, we built a benchmark, AlgoTune, comprised of 155 widely used math, physics, and computer science functions. For each function, the goal is to write code that produces the same outputs as the original function, while being faster. In addition to the benchmark, we also provide an agent, AlgoTuner, which allows language models to easily optimize code. For contributions to the AlgoTune benchmark, see the [contribution guide](#-contributing-new-problems-to-algotune).

<p align="center">
  <a href="https://algotune.io/">
    <img
      src="https://github.com/oripress/AlgoTune/blob/main/assets/algotune_banner.png?raw=true"
      alt="AlgoTune banner"
      width="800"
    />
  </a>
</p>

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -e .  # or, if you prefer conda:
# conda create -n algotune python=3.10
# conda activate algotune
# pip install -e .

# 2. Add your API key
echo "OPENAI_API_KEY=your_key_here" > .env
# OR
echo "CLAUDE_API_KEY=your_key_here" > .env
```

### Run AlgoTune (stand-alone mode)

```bash
# 1️⃣  Measure baseline speed for two tasks (100 ms target)
./algotune.sh --standalone generate --target-time-ms 100 --tasks svm kmeans

# 2️⃣  Ask an LM to optimise the same tasks with model "o4-mini"
./algotune.sh --standalone agent o4-mini svm kmeans

# 3️⃣  View the aggregated speed-up report
cat reports/agent_summary.json
```

### Running on SLURM (cluster)  
When `sbatch` is available the launcher auto-detects SLURM. Use the same two-step workflow:

```bash
# One-time image build (only once per cluster)
sudo singularity build algotune.sif slurm/image.def

# Generate baseline datasets for all tasks
./algotune.sh generate --target-time-ms 100

# Run the agent on all tasks
./algotune.sh agent o4-mini

# Results are summarised in:
cat reports/agent_summary.json
```

---

## 🔧 Commands & Options

### Essential commands

```bash
# List all available tasks
./algotune.sh list-tasks

# Generate baseline for specific tasks (stand-alone)
./algotune.sh --standalone generate --tasks svm kmeans

# Run agent on specific tasks (stand-alone)
./algotune.sh --standalone agent o4-mini svm kmeans

# Run agent on specific tasks (SLURM)
./algotune.sh agent o4-mini svm kmeans
```

---

## 🛠️ Contributing New Problems to AlgoTune

We welcome contributions of new optimization problems! Each problem requires:
- A textual description
- A solver for generating optimal solutions  
- An `is_solution` function for verification

We prefer problems with existing solvers in common libraries (e.g., NumPy, SciPy, FAISS). For a complete example, see the [QR factorization task](AlgoTuneTasks/qr_factorization).

**Example problems:** KD-Trees, Vector Quantization, K-Means, PageRank, Approximate FFT

We especially seek real-world problems from active research and industry domains without known optimal solutions.

### Technical Requirements

Implement the following functions with the specified signatures:

* **`generate_problem(n, random_seed=1)`**
  - **`n`**: [Integer] Controls problem difficulty.
  - **`random_seed`**: Controls any random seeds used in the generation process.
  - **Returns**: A problem instance (e.g., list, list of lists, or string). The problem should scale in difficulty as `n` grows.

  _Note_: The `n` parameter allows us to control problem difficulty and scale solve times to a target duration (e.g., 10 seconds) to enable meaningful speedup comparisons.

* **`solve(problem)`**
  - **`problem`**: An instance produced by `generate_problem`.
  - **Returns**: The optimal solution in the format specific to your task.

* **`is_solution(problem, solution)`**
  - **`problem`**: The instance produced by `generate_problem`.
  - **`solution`**: A proposed solution.
  - **Returns**: A boolean value: True if the solution is valid and optimal, False otherwise.

Ensure that:
- All functions (except `__init__`) include explicit type annotations.
- The function signatures are exactly as specified (including `self, problem, solution` for `is_solution`)

### Example Task
Example: QR Decomposition. Full implementation is [here](AlgoTuneTasks/qr_factorization).

```python
    def generate_problem(self, n: int, random_seed: int = 1) -> Dict[str, np.ndarray]:
        np.random.seed(random_seed)
        A = np.random.randn(n, n + 1)
        problem = {"matrix": A}
        return problem

    def solve(
        self, problem: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, List[List[float]]]]:
        A = problem["matrix"]
        Q, R = np.linalg.qr(A, mode="reduced")
        solution = {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
        return solution

    def is_solution(
        self,
        problem: Dict[str, np.ndarray],
        solution: Dict[str, Dict[str, List[List[float]]]]
    ) -> bool:
        # (Additional checks for solution properties are omitted for brevity).

        # Check orthonormality of Q: Q^T @ Q should be approximately the identity matrix.
        if not np.allclose(Q.T @ Q, np.eye(n), atol=1e-6):
            logging.error("Matrix Q does not have orthonormal columns.")
            return False

        # Check if the product Q @ R reconstructs A within tolerance.
        if not np.allclose(A, Q @ R, atol=1e-6):
            logging.error("Reconstructed matrix does not match the original matrix within tolerance.")
            return False

        # All checks passed
        return True
```

- Each task requires a `description.txt` file defining the task, input, and output. It must contain the following headers in order:
  - `Input:`
  - `Example input:`
  - `Output:`
  - `Example output:`

Provide raw examples for inputs and outputs.

#### Guidelines:
1. Prefer tasks where `solve()` is a concise wrapper around a function from an established, well-tested library (e.g., NumPy, SciPy). New libraries are acceptable if they are popular and tested.
2. Before contributing, check existing tasks in the `AlgoTuneTasks/` directory to ensure novelty and maintain format consistency.

## 📤 How to Submit

Submit a pull request with your implementation, ensuring it follows the specified format and includes all required files. Tests can be run locally via provided scripts or will be executed automatically via GitHub Actions upon PR submission.

## Development Setup

### Installation

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/oripress/AlgoTune.git
cd AlgoTune

# Install the package with development dependencies
pip install -e ".[dev]"
```

This will install all required dependencies listed in the pyproject.toml file, including pre-commit and ruff.

### Pre-commit Hooks (Optional)

This repository uses GitHub Actions to automatically run ruff checks on all pull requests. Additionally, you can optionally set up pre-commit hooks to run these checks locally before committing:

1. After installing the development dependencies, install the git hook scripts:
   ```bash
   pre-commit install
   ```

2. (Optional) Run against all files:
   ```bash
   pre-commit run --all-files
   ```

Using pre-commit hooks is completely optional - if you don't install them, the GitHub workflow will still run ruff checks when you submit a pull request.

## Running Workflow Tests

You can run workflow tests (formatting and consistency checks) in two ways:

**Push to GitHub:** Push your changes to a branch or create a pull request. The checks will run automatically via GitHub Actions.

or:

**Run Tests Locally:** Use the `pre-commit run --all-files` command to run Ruff formatting and linting checks on your local machine before pushing.
And then, to run the validation and consistency tests for your changes:

```bash
# Run basic task unit tests (checks generate/solve/is_solution functionality)
python .github/scripts/test_tasks.py                                     # <1 minute for all tasks together

# Run timing and consistency tests for specific tasks (or all if omitted)
# This finds the problem size 'n' where solve time exceeds a threshold and validates results.
python .github/scripts/test_timing_and_consistency.py <task_a,task_b>    # ~1 minute/task

# Run the is_solution return type check for all tasks
# This ensures the is_solution method returns a standard Python boolean.
python .github/scripts/test_is_solution_return_type.py                   # <1 minute for all tasks together
```

## FAQ

**Q: What's the difference between "generate" and "agent"?**
- `generate` measures how fast your original code runs (baseline)
- `agent` uses AI to create optimized versions of that code

**Q: Do I need to run on all 120+ tasks?**
No! Start with just a few tasks like `svm` or `kmeans` to test things out.

**Q: What does `--standalone` do?**
It runs the code directly on your machine instead of submitting to a compute cluster.

**Q: Why does generation take so long?**
We need accurate baseline measurements. The `--target-time-ms` parameter controls how long we measure each function (default: 100 ms per measurement, multiple trials).

**Q: Can I use my own OpenAI/Anthropic API key?**
Yes! Just add it to your `.env` file as shown in the setup.

## Questions?
Contact us with any questions: ori.press@bethgelab.org
