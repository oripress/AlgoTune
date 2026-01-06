# GitHub Workflow Scripts

This directory contains scripts used by GitHub Actions workflows for testing and validating tasks.

## Scripts

- **test_tasks.py**: Validates that all Task implementations follow the required conventions and structure
- **test_consistency.py**: Verifies that tasks produce consistent results across multiple runs

These scripts are called from the workflow files in `../.github/workflows/`. 