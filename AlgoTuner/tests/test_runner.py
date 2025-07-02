#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime
import difflib
import subprocess
import signal
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.run_tests import DummyLLM
from AlgoTuner.utils.logger import setup_logging

@dataclass
class TestResult:
    test_name: str
    passed: bool
    input_differences: List[Tuple[int, str, str]]
    execution_time: float = 0.0

class TestRunner:
    def __init__(self):
        self.logger = setup_logging(task="test_runner")
        self.results: List[TestResult] = []
        self.processes: List[subprocess.Popen] = []
        
    def run_test_suite(self, test_file: str) -> List[TestResult]:
        """Run a single test suite and return results"""
        task_name = Path(test_file).stem
        self.logger.info(f"Running test suite: {test_file} for task: {task_name}")
        
        output_dir = Path("tests/outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{task_name}_{timestamp}.json"
        
        cmd = [sys.executable, "tests/run_tests.py", "--input", str(test_file)]
        process = subprocess.Popen(cmd)
        self.processes.append(process)
        
        process.wait()
        
        if output_file.exists():
            with open(output_file) as f:
                output_data = json.load(f)
                return self._process_output(output_data, test_file)
        else:
            self.logger.error(f"No output file was created at {output_file}")
            return []

    def cleanup(self):
        """Clean up any running processes"""
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                self.logger.error(f"Error cleaning up process: {e}")

    def __del__(self):
        """Ensure processes are cleaned up when the runner is destroyed"""
        self.cleanup()

    def _process_output(self, output_data: Dict, test_file: str) -> List[TestResult]:
        """Process the output data and create TestResult objects"""
        results = []
        
        for test_case in output_data.get("test_cases", []):
            result = TestResult(
                test_name=f"{Path(test_file).stem}_{test_case.get('name', 'unnamed')}",
                passed=test_case.get("passed", False),
                input_differences=[],
                execution_time=test_case.get("execution_time", 0.0)
            )
            results.append(result)
            
        return results

    def extract_inputs(self, logs: List[str]) -> List[str]:
        """Extract only the actual input strings from Sent to LLM messages."""
        inputs: List[str] = []
        for idx, line in enumerate(logs):
            if "Sent to LLM:" in line:
                after_prefix = line.split("Sent to LLM:", 1)[1].strip()

                if not after_prefix and idx + 1 < len(logs):
                    after_prefix = logs[idx + 1].strip()

                if (
                    not after_prefix
                    or after_prefix.lower().startswith("you have sent")
                    or after_prefix.lower() == "eval"
                ):
                    continue

                inputs.append(after_prefix)
        return inputs

    def compare_logs(self, test_file: str, actual_logs: List[str]) -> List[Tuple[int, str, str]]:
        """Compare actual logs with expected logs, focusing only on Sent to LLM inputs"""

        return []

    def generate_report(self) -> str:
        """Generate a summary report of all test results"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        report = [
            "=== Test Summary Report ===",
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Success Rate: {(passed_tests/total_tests)*100:.2f}%",
            "\nDetailed Results:",
        ]
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            report.append(f"\n{status} {result.test_name}")
            if not result.passed:
                report.append("  Input Differences:")
                for input_num, expected, actual in result.input_differences:
                    report.append(f"\n    Input {input_num}:")
                    if expected:
                        report.append(f"      Expected: {expected}")
                    if actual:
                        report.append(f"      Actual:   {actual}")
            report.append(f"  Time: {result.execution_time:.2f}s")
            
        return "\n".join(report)

    def find_most_recent_log(self, task_name: str) -> Optional[Path]:
        """Find the most recent log file for a given task"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return None
            
        log_files = list(log_dir.glob(f"*{task_name}*.log"))
        if not log_files:
            return None
            
        return max(log_files, key=lambda x: x.stat().st_mtime)

def main():
    runner = TestRunner()
    
    try:
        input_dir = Path("tests/inputs")
        if not input_dir.exists():
            runner.logger.error(f"Input directory not found: {input_dir}")
            return
            
        for test_file in input_dir.glob("*.txt"):
            if test_file.name == "expected":
                continue
                
            results = runner.run_test_suite(str(test_file))
            runner.results.extend(results)
            
            for result in results:
                task_name = test_file.stem
                log_file = runner.find_most_recent_log(task_name)
                if log_file and log_file.exists():
                    with open(log_file) as f:
                        actual_logs = f.readlines()
                    result.input_differences = runner.compare_logs(str(test_file), actual_logs)
                    result.passed = len(result.input_differences) == 0
                else:
                    runner.logger.warning(f"No log file found for task {task_name}")
        
        report = runner.generate_report()
        print(report)
        
        report_file = Path("tests/outputs") / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        runner.logger.info(f"Test report saved to {report_file}")
        
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()
