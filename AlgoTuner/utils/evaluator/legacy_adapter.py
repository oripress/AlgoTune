"""
Adapters to maintain backward compatibility with the old evaluation system.
This allows gradual migration to the new clean architecture.
"""

import logging
from typing import Any, Dict, List, Optional

from AlgoTuner.utils.evaluator.evaluation_types import (
    DatasetResults,
    RunnerConfig,
    ErrorType,
)
from AlgoTuner.utils.evaluator.evaluation_orchestrator import EvaluationOrchestrator


class AttributedList(list):
    """Legacy AttributedList for backward compatibility."""
    pass


class LegacyAdapter:
    """Adapts between old and new evaluation interfaces."""
    
    def __init__(self):
        """Initialize the legacy adapter."""
        self.logger = logging.getLogger(__name__)
    
    def create_runner_config(
        self,
        num_runs: int = 10,
        warmup_runs: int = 1,
        timeout_multiplier: float = 10.0,
        capture_output: bool = False,
        **kwargs
    ) -> RunnerConfig:
        """
        Create RunnerConfig from legacy parameters.
        
        Args:
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            timeout_multiplier: Timeout multiplier
            capture_output: Whether to capture stdout/stderr
            **kwargs: Additional legacy parameters
            
        Returns:
            RunnerConfig for new architecture
        """
        # Extract feature flags
        feature_flags = {}
        if "use_new_pipeline" in kwargs:
            feature_flags["use_new_pipeline"] = kwargs.pop("use_new_pipeline")
        
        # Map legacy parameters
        config = RunnerConfig(
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            timeout_multiplier=timeout_multiplier,
            capture_output=capture_output,
            validate_in_process=kwargs.get("validate_in_process", True),
            strip_solutions=kwargs.get("strip_solutions", True),
            use_isolated_execution=kwargs.get("use_isolated_execution", True),
            feature_flags=feature_flags
        )
        
        return config
    
    def adapt_dataset_results(self, dataset_results: DatasetResults) -> Any:
        """
        Convert DatasetResults to legacy format.
        
        Args:
            dataset_results: Results from new architecture
            
        Returns:
            Legacy AttributedList with attached metrics, or dict for early exit errors
        """
        # Check if this was an early exit due to critical error
        if dataset_results.results and dataset_results.results[-1].execution.error_type in [
            ErrorType.MEMORY_ERROR,
            ErrorType.IMPORT_ERROR,
            ErrorType.EXECUTION_ERROR,
            ErrorType.TIMEOUT,
        ]:
            # Return early exit error format
            last_result = dataset_results.results[-1]
            return {
                "success": False,
                "problem_id": last_result.problem_id,
                "error": last_result.execution.error,
                "error_type": last_result.execution.error_type.value,
                "traceback": last_result.execution.traceback,
                "code_context": last_result.execution.error,  # Error already includes context
                "evaluation_stopped_early": True,
                "evaluation_type": "dataset",
                "num_evaluated": len(dataset_results.results),
                "total_problems": len(dataset_results.results)  # We don't know total on early exit
            }
        
        # Convert to legacy format
        legacy_data = dataset_results.to_legacy_format()
        
        # Create AttributedList with results
        attributed_list = AttributedList(legacy_data.get("results", []))
        
        # Attach aggregate metrics
        attributed_list.aggregate_metrics = legacy_data.get("aggregate_metrics", {})
        
        # Attach invalid solution analysis
        if "invalid_solution_analysis" in legacy_data:
            attributed_list.invalid_solution_analysis = legacy_data["invalid_solution_analysis"]
        
        return attributed_list
    
    def wrap_evaluate_function(
        self,
        task_instance: Any,
        dataset: List[Any],
        solver_func: Any,
        config: RunnerConfig,
        **kwargs
    ) -> Any:
        """
        Wrap evaluation using new architecture but return legacy format.
        
        This is the main integration point for gradual migration.
        
        Args:
            task_instance: Task instance
            dataset: Dataset to evaluate
            solver_func: Solver function
            config: Runner configuration
            **kwargs: Additional legacy parameters
            
        Returns:
            Legacy formatted results
        """
        # Check feature flag
        use_new_pipeline = config.feature_flags.get("use_new_pipeline", False)
        
        if not use_new_pipeline:
            # Use legacy evaluation (would call old evaluate_with_runner_on_task)
            self.logger.debug("Using legacy evaluation pipeline")
            raise NotImplementedError("Legacy pipeline should be called directly")
        
        # Use new evaluation pipeline
        self.logger.info("Using new clean architecture evaluation pipeline")
        
        # Create orchestrator
        orchestrator = EvaluationOrchestrator(config)
        
        # Run evaluation
        dataset_results = orchestrator.evaluate_dataset(
            task_instance=task_instance,
            dataset=dataset,
            solver_func=solver_func,
            task_name=kwargs.get("task_name"),
            baseline_func=kwargs.get("baseline_func"),
            progress_callback=kwargs.get("progress_callback")
        )
        
        # Convert to legacy format
        return self.adapt_dataset_results(dataset_results)
    
    def extract_failed_instances(
        self,
        results: Any
    ) -> Dict[str, List[Any]]:
        """
        Extract failed instances for post-evaluation analysis.
        
        Args:
            results: Evaluation results (old or new format)
            
        Returns:
            Dictionary mapping task names to failed instances
        """
        failed_instances = {}
        
        if isinstance(results, DatasetResults):
            # New format
            task_name = results.task_name
            failed = []
            
            for result in results.results:
                if not result.is_valid and result.validation:
                    # In new architecture, we already have the context
                    # No need to store the full solution
                    failed.append((
                        task_name,
                        result.problem_id,
                        result.validation.context
                    ))
            
            if failed:
                failed_instances[task_name] = failed
                
        elif isinstance(results, AttributedList):
            # Legacy format - would need to handle old structure
            pass
        
        return failed_instances


def create_legacy_compatible_runner(**kwargs) -> Any:
    """
    Create a runner that's compatible with legacy code but uses new architecture.
    
    This function can be used as a drop-in replacement for old runner creation.
    
    Args:
        **kwargs: Legacy runner parameters
        
    Returns:
        Runner object (adapter or legacy depending on feature flags)
    """
    adapter = LegacyAdapter()
    config = adapter.create_runner_config(**kwargs)
    
    if config.feature_flags.get("use_new_pipeline", False):
        # Return a wrapper that uses new architecture
        class NewArchitectureRunner:
            def __init__(self, config, adapter):
                self.config = config
                self.adapter = adapter
                self.orchestrator = EvaluationOrchestrator(config)
            
            def evaluate(self, task_instance, dataset, solver_func, **kwargs):
                # Use new architecture
                dataset_results = self.orchestrator.evaluate_dataset(
                    task_instance=task_instance,
                    dataset=dataset,
                    solver_func=solver_func,
                    **kwargs
                )
                # Convert to legacy format
                return self.adapter.adapt_dataset_results(dataset_results)
        
        return NewArchitectureRunner(config, adapter)
    else:
        # Return legacy runner (would import and return old runner)
        raise NotImplementedError("Legacy runner should be imported from old code")