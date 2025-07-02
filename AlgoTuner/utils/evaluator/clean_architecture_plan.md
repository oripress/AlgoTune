# AlgoTuner Clean Architecture Refactoring Plan

## Overview
This plan refactors the evaluation system to eliminate spaghetti code while maintaining backward compatibility.

## Implementation Steps

### Step 1: Create Core Data Classes (Week 1)
- [ ] Create `evaluation_types.py` with immutable dataclasses
- [ ] Add type hints throughout the codebase
- [ ] No changes to existing code yet

### Step 2: Extract Single-Responsibility Components (Week 2)
- [ ] Create `solver_executor.py` - Just runs solvers, no validation
- [ ] Create `validation_pipeline.py` - Single validation point
- [ ] Create `memory_optimizer.py` - Handles all stripping
- [ ] Create `result_aggregator.py` - Metrics and formatting

### Step 3: Create Orchestrator with Adapter Pattern (Week 3)
- [ ] Create `evaluation_orchestrator.py` as new entry point
- [ ] Add adapters to convert old data structures to new
- [ ] Existing code continues to work unchanged

### Step 4: Migrate Incrementally (Week 4-5)
- [ ] Update `main.py` to use orchestrator for new code paths
- [ ] Keep old paths working with deprecation warnings
- [ ] Add feature flags to switch between old/new

### Step 5: Update Tests and Documentation (Week 6)
- [ ] Write comprehensive tests for new components
- [ ] Update documentation
- [ ] Add migration guide

### Step 6: Deprecate Old Code (Week 7-8)
- [ ] Mark old functions as deprecated
- [ ] Provide clear migration paths
- [ ] Remove dead code after grace period

## Key Files to Create

### 1. evaluation_types.py
```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class TimingMetrics:
    mean_ms: float
    min_ms: float
    max_ms: float
    values_ms: List[float]
    warmup_ms: float

@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    output: Any
    timing: TimingMetrics
    stdout: str = ""
    stderr: str = ""
    error: Optional[Exception] = None
    traceback: Optional[str] = None

@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    failure_context: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
@dataclass(frozen=True)
class EvaluationResult:
    problem_id: str
    execution: ExecutionResult
    validation: ValidationResult
    speedup: Optional[float] = None
    baseline_time_ms: Optional[float] = None
```

### 2. solver_executor.py
```python
class SolverExecutor:
    """Executes solver code in isolation."""
    
    def __init__(self, runner_config: RunnerConfig):
        self.config = runner_config
        
    def execute(self, solver: Solver, problem: Any) -> ExecutionResult:
        """Execute solver on problem, return execution result only."""
        # Run in isolated process
        # Capture timing, output, errors
        # NO validation here
        pass
```

### 3. validation_pipeline.py
```python
class ValidationPipeline:
    """Single point of validation for all solutions."""
    
    def __init__(self, context_manager: ValidationContextManager):
        self.context_manager = context_manager
        
    def validate(self, task: Task, problem: Any, solution: Any) -> ValidationResult:
        """Validate solution and capture context if invalid."""
        try:
            is_valid = task.is_solution(problem, solution)
            if not is_valid:
                context = self._capture_context(task, problem, solution)
                return ValidationResult(
                    is_valid=False,
                    failure_context=context,
                    error_type="invalid_solution"
                )
            return ValidationResult(is_valid=True)
        except Exception as e:
            # Handle validation errors
            pass
    
    def _capture_context(self, task, problem, solution) -> str:
        """Capture failure context before solution is stripped."""
        # Use failure analyzer here
        pass
```

### 4. memory_optimizer.py
```python
class MemoryOptimizer:
    """Handles solution stripping and memory management."""
    
    STRIPPED_MARKER = "__stripped__"
    
    def strip_solution(self, solution: Any) -> Dict[str, Any]:
        """Strip solution to minimal representation."""
        return {
            self.STRIPPED_MARKER: True,
            "type": type(solution).__name__,
            "size_bytes": self._estimate_size(solution)
        }
    
    def should_strip(self, solution: Any) -> bool:
        """Determine if solution should be stripped."""
        # Check size thresholds
        pass
```

### 5. evaluation_orchestrator.py
```python
class EvaluationOrchestrator:
    """Main entry point for all evaluations."""
    
    def __init__(self):
        self.executor = SolverExecutor()
        self.validator = ValidationPipeline()
        self.memory_opt = MemoryOptimizer()
        self.aggregator = ResultAggregator()
    
    def evaluate_dataset(self, task: Task, dataset: List[Any], 
                        solver: Solver) -> DatasetResults:
        """Evaluate solver on entire dataset."""
        results = []
        
        for problem in dataset:
            # 1. Execute solver
            exec_result = self.executor.execute(solver, problem)
            
            # 2. Validate if execution succeeded
            val_result = ValidationResult(is_valid=False)
            if exec_result.success:
                val_result = self.validator.validate(
                    task, problem, exec_result.output
                )
            
            # 3. Strip solution after validation
            if self.memory_opt.should_strip(exec_result.output):
                exec_result = self._strip_execution_result(exec_result)
            
            # 4. Create evaluation result
            eval_result = EvaluationResult(
                problem_id=problem.id,
                execution=exec_result,
                validation=val_result,
                speedup=self._calculate_speedup(exec_result, baseline)
            )
            results.append(eval_result)
        
        # 5. Aggregate results
        return self.aggregator.aggregate(results)
```

## Migration Strategy

### Phase 1: Parallel Implementation
- New code lives alongside old code
- No breaking changes
- Feature flags control which path is used

### Phase 2: Gradual Migration
```python
def evaluate_code_on_dataset(...):
    if USE_NEW_PIPELINE:
        # Use new orchestrator
        orchestrator = EvaluationOrchestrator()
        new_results = orchestrator.evaluate_dataset(...)
        # Convert to old format for compatibility
        return adapt_new_to_old(new_results)
    else:
        # Use existing implementation
        return _legacy_evaluate_code_on_dataset(...)
```

### Phase 3: Full Migration
- Update all callers to use new API
- Remove adaptation layer
- Delete legacy code

## Benefits

1. **Clear Separation of Concerns**
   - Execution, validation, memory management are separate
   - Easy to test each component independently

2. **Single Point of Truth**
   - Validation happens in one place only
   - Solution stripping happens in one place only
   - No duplicate code paths

3. **Better Error Handling**
   - Clear error types and contexts
   - Failure context preserved properly

4. **Easier Maintenance**
   - Each component is small and focused
   - Clear interfaces between components
   - Easy to add new features

5. **Performance**
   - Solutions stripped only once
   - No redundant validation
   - Better memory management

## Testing Strategy

1. **Unit Tests**: Each component tested in isolation
2. **Integration Tests**: Test component interactions
3. **Regression Tests**: Ensure old behavior preserved
4. **Performance Tests**: Verify no performance regression
5. **Memory Tests**: Ensure memory usage improved