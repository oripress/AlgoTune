import os
import logging
from pathlib import Path
from typing import Optional, Union, List


class FileManager:
    """
    Centralized file operations manager.
    Handles file reading, path validation, and workspace management.
    """

    def __init__(self, workspace_root: Union[str, Path]):
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {self.workspace_root}")

    def load_file_content(self, file_path: Union[str, Path]) -> str:
        """
        Loads and returns the content of a given file.

        Args:
            file_path: The path to the file to be read, relative to workspace root

        Returns:
            The content of the file as a string

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file path is outside workspace
        """
        try:
            full_path = self._get_safe_file_path(file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logging.critical(f"File not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
        except Exception as e:
            logging.critical(f"Error reading file {file_path}: {e}")
            raise

    def save_file_content(
        self, file_path: Union[str, Path], content: str, create_dirs: bool = True
    ) -> None:
        """
        Save content to a file, optionally creating parent directories.

        Args:
            file_path: The path to save to, relative to workspace root
            content: The content to save
            create_dirs: Whether to create parent directories if they don't exist

        Raises:
            ValueError: If the file path is outside workspace
            OSError: If file operations fail
        """
        try:
            full_path = self._get_safe_file_path(file_path)
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Error saving file {file_path}: {e}")
            raise

    def read_lines(
        self,
        file_path: Union[str, Path],
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> List[str]:
        """
        Read specific lines from a file.

        Args:
            file_path: The path to read from, relative to workspace root
            start_line: The line to start reading from (1-based)
            end_line: The line to end reading at (inclusive, 1-based)

        Returns:
            List of lines read

        Raises:
            ValueError: If line numbers are invalid or file path is outside workspace
            FileNotFoundError: If the file doesn't exist
        """
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if end_line is not None and end_line < start_line:
            raise ValueError(
                f"end_line ({end_line}) must be >= start_line ({start_line})"
            )

        try:
            full_path = self._get_safe_file_path(file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if end_line is None or end_line > len(lines):
                end_line = len(lines)

            return [line.rstrip("\n") for line in lines[start_line - 1 : end_line]]
        except Exception as e:
            logging.error(f"Error reading lines from {file_path}: {e}")
            raise

    def list_files(self, pattern: Optional[str] = None) -> List[Path]:
        """
        List all files in the workspace, optionally filtered by a glob pattern.

        Args:
            pattern: Optional glob pattern to filter files (e.g., "*.py")

        Returns:
            List of Path objects relative to workspace root
        """
        try:
            if pattern:
                files = list(self.workspace_root.glob(pattern))
            else:
                files = list(self.workspace_root.rglob("*"))

            # Filter to only files (not directories) and convert to relative paths
            return [f.relative_to(self.workspace_root) for f in files if f.is_file()]
        except Exception as e:
            logging.error(f"Error listing files: {e}")
            raise

    def ensure_dir(self, dir_path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            dir_path: Directory path relative to workspace root

        Returns:
            Path object for the directory

        Raises:
            ValueError: If path is outside workspace
        """
        try:
            full_path = self._get_safe_file_path(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
            return full_path.relative_to(self.workspace_root)
        except Exception as e:
            logging.error(f"Error ensuring directory exists: {e}")
            raise

    def delete_file(
        self, file_path: Union[str, Path], missing_ok: bool = False
    ) -> None:
        """
        Delete a file from the workspace.

        Args:
            file_path: Path to the file to delete
            missing_ok: Whether to ignore if the file doesn't exist

        Raises:
            FileNotFoundError: If the file doesn't exist and missing_ok is False
            ValueError: If the file path is outside workspace
        """
        try:
            full_path = self._get_safe_file_path(file_path)
            try:
                full_path.unlink()
            except FileNotFoundError:
                if not missing_ok:
                    raise
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            raise

    def _get_safe_file_path(self, file_path: Union[str, Path]) -> Path:
        """
        Safely convert a file path to an absolute Path object with validation.
        Ensures the file path is within the workspace and properly formatted.

        Args:
            file_path: The path to validate, relative to workspace root

        Returns:
            Resolved absolute Path object

        Raises:
            ValueError: If path attempts to escape workspace root
        """
        try:
            # Convert to Path object
            path = Path(file_path)

            # Resolve the absolute path to handle any '..' or '.' in the path
            abs_path = (self.workspace_root / path).resolve()

            # Check if the resolved path is within the workspace
            if not str(abs_path).startswith(str(self.workspace_root)):
                raise ValueError(
                    f"File path {file_path} attempts to access location outside workspace"
                )

            return abs_path

        except Exception as e:
            logging.error(f"Invalid file path {file_path}: {e}")
            raise ValueError(f"Invalid file path {file_path}: {e}")

    def exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists within the workspace."""
        try:
            return self._get_safe_file_path(file_path).exists()
        except ValueError:
            return False

    def is_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a path points to a file within the workspace."""
        try:
            return self._get_safe_file_path(file_path).is_file()
        except ValueError:
            return False

    def get_relative_path(self, file_path: Union[str, Path]) -> Path:
        """Convert an absolute or relative path to workspace-relative path."""
        abs_path = self._get_safe_file_path(file_path)
        return abs_path.relative_to(self.workspace_root)


# Legacy function for backward compatibility
def load_file_content(file_path: Union[str, Path]) -> str:
    """
    Legacy function that creates a FileManager instance for a single operation.
    For better performance, create and reuse a FileManager instance.
    """
    workspace_root = os.getcwd()
    manager = FileManager(workspace_root)
    return manager.load_file_content(file_path)
