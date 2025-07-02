from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Pattern
import re
from enum import Enum, unique
from AlgoTuner.interfaces.error_message import GENERIC_ERROR_MESSAGE


@dataclass
class CommandFormat:
    """Defines the format and validation rules for a command."""

    name: str
    pattern: Pattern  # Compiled regex pattern
    description: str
    example: str
    error_message: str


@dataclass
class ParsedCommand:
    """Represents a parsed command with its arguments."""

    command: str
    args: Dict[str, Any]
    total_blocks: Optional[int] = None
    raw_text: Optional[str] = None


@dataclass
class EditCommandFormat:
    """Centralized definition of edit command format."""

    TEMPLATE = """edit
file: filename  # Existing or new file
lines: start-end  # 0-0 for new files, 0-N to prepend+replace, 1-N to replace lines
---
content
---"""

    FILE_PATTERN = re.compile(r"^[a-zA-Z0-9_./\-]+$")  # Basic file name validation
    REQUIRED_PARTS = ["file:", "lines:", "---"]
    EXAMPLE = """# Edit existing file (replace lines 1-5):
edit
file: solver.py
lines: 1-5
---
def example():
    pass
---

# Create new file:
edit
file: new_file.py
lines: 0-0
---
def new_function():
    pass
---

# Prepend to file and replace first 3 lines:
edit
file: solver.py
lines: 0-3
---
# New header comment
def modified_function():
    pass
---"""


# Command patterns for regex matching
COMMAND_FORMATS = {
    "ls": CommandFormat(
        name="ls",
        pattern=re.compile(r"^ls(?:\s*)?$"),
        description="List files in the current working directory",
        example="""
```
ls
```
""",
        error_message="Invalid ls format.",
    ),
    "view_file": CommandFormat(
        name="view_file",
        pattern=re.compile(r"^view_file\s+(\S+)(?:\s+(\d+))?\s*$"),
        description="View contents of a file, optionally starting from a specific line",
        example="""
```
view_file solver.py 11
```
""",
        error_message="Invalid view_file format.",
    ),
    "reference": CommandFormat(
        name="reference",
        pattern=re.compile(r"^reference\s+([\s\S]+)$", re.DOTALL),
        description="Query the reference solver function",
        example="""
```
reference [1,2,3]
```
""",
        error_message="Invalid reference format.",
    ),
    "eval_input": CommandFormat(
        name="eval_input",
        pattern=re.compile(r"^eval_input(?:\s+([\s\S]+))?$", re.DOTALL),
        description="Run your current solver implementation on the given input and compare it with the oracle solution.",
        example="""
```
eval_input [[0.1,-0.34],[10.9,0.64]]
```
""",
        error_message="Invalid eval_input format.",
    ),
    "revert": CommandFormat(
        name="revert",
        pattern=re.compile(r"^revert\s*$"),
        description="Revert the last edit",
        example="""
```
revert
```
""",
        error_message="Invalid revert format.",
    ),
    "edit": CommandFormat(
        name="edit",
        pattern=re.compile(r"^\s*edit\s+file:\s+(.+?)\s+lines:\s+(\d+\s*-\s*\d+)\s+---\s*([\s\S]+?)(?:\s+---)?\s*$"),
        description="Edit a file between specified line numbers",
        example="""
```
edit
file: solver.py
lines: 11-12
---
def foo(self, x):
    return x + 1
---
```
""",
        error_message=f"Invalid edit format. Expected:\n{EditCommandFormat.TEMPLATE}",
    ),
    "delete": CommandFormat(
        name="delete",
        pattern=re.compile(r"^delete\s*\nfile:\s*(\S+)\s*\nlines:\s*(\d+)-(\d+)\s*$"),
        description="Delete lines in the specified range",
        example="""
```
delete
file: solver.py
lines: 5-10
```
""",
        error_message=f"Invalid delete format.",
    ),
    "profile": CommandFormat(
        name="profile",
        pattern=re.compile(r"^profile\s+(\S+\.py)\s+(.+)$", re.DOTALL),
        description="Profile the current solution on a given input",
        example="""
```
profile solver.py
```
""",
        error_message="Invalid profile format.",
    ),
    "profile_lines": CommandFormat(
        name="profile_lines",
        pattern=re.compile(r"^profile_lines\s+(\S+\.py)\s+((?:\d+(?:-\d+)?)(?:\s*,\s*\d+(?:-\d+)?)*?)\s+(.+)$", re.DOTALL),
        description="Profile specific lines in the current solution",
        example="""
```
profile_lines solver.py 5,6,7
```
""",
        error_message="Invalid profile_lines format.",
    ),
    "eval": CommandFormat(
        name="eval",
        pattern=re.compile(r"^eval\s*$"),
        description="Run evaluation on the current solution",
        example="""
```
eval
```
""",
        error_message="Invalid eval format. Expected: eval (no arguments)",
    ),
}

# Error messages for command parsing
ERROR_MESSAGES = {
    "invalid_format": "Invalid command format. Please check command syntax and try again.",
    "empty_command": GENERIC_ERROR_MESSAGE,
    "line_number": "Invalid line number '{}'. Line numbers must be positive integers.",
    "start_line": "Start line must be greater than or equal to 0, got {}.",
    "end_line": "End line must be greater than or equal to 0, got {}.",
    "line_range": "Edit command error: start line cannot be greater than end line.",
    "prepend_range": "For prepending content, start line must be 0 and end line must be >= 0.",
    "parsing_error": "Error parsing command: {}",
    "file_permission": "Cannot access file '{}'. Check file permissions.",
    "file_not_found": "File '{}' not found. Use lines: 0-0 to create a new file, or lines: 0-N to prepend to existing file.",
    "unknown_command": "Unknown command '{}':\n{}",
    "edit_format": f"Invalid edit format. Expected:\n{EditCommandFormat.TEMPLATE}",
    "invalid_filename": "Invalid filename '{}'. Filenames can only contain letters, numbers, dots, dashes, and underscores.",
    "duplicate_specification": "Duplicate {} specification found. Each component should only be specified once.",
    "empty_content": "Edit command content cannot be empty (except when creating new files with lines: 0-0).",
    "delete_range": "Invalid delete range: end line {} cannot be smaller than start line {}.",
    "delete_out_of_bounds": "Line range {}-{} is out of bounds for file '{}' which has {} lines.",
    "delete_empty_file": "Cannot delete lines from empty file '{}'.",
    "text_after_command": GENERIC_ERROR_MESSAGE,
}

# For backward compatibility
COMMAND_PATTERNS = {name: cmd.pattern.pattern for name, cmd in COMMAND_FORMATS.items()}

# Command descriptions for help text
COMMAND_DESCRIPTIONS = {
    name: f"{cmd.description}\nUsage: {cmd.example}"
    for name, cmd in COMMAND_FORMATS.items()
}
COMMAND_DESCRIPTIONS["edit"] = (
    f"Edit a file between specified line numbers\nUsage:\n{EditCommandFormat.EXAMPLE}"
)


@unique
class EditStatus(Enum):
    SUCCESS = "edit_success"
    FAILURE = "edit_failure"


@unique
class SnapshotStatus(Enum):
    SUCCESS = "snapshot_success"
    FAILURE = "snapshot_failure"


@unique
class EvalStatus(Enum):
    SUCCESS = "eval_success"
    FAILURE = "eval_failure"
    TIMEOUT = "eval_timeout"


@unique
class FileStatus(Enum):
    SUCCESS = "file_success"
    FAILURE = "file_failure"


@unique
class ProfileStatus(Enum):
    SUCCESS = "profile_success"
    FAILURE = "profile_failure"
    TIMEOUT = "profile_timeout"


class CommandResult:
    """Represents the result of a command execution."""

    def __init__(
        self,
        success: bool,
        message: Optional[str] = None,
        error: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        stdout: Optional[str] = None,  # Output from command execution
        stderr: Optional[str] = None,  # Error output from command execution
        edit_status: Optional[str] = None,  # Can be EditStatus.SUCCESS.value or EditStatus.FAILURE.value
        snapshot_status: Optional[str] = None,  # Can be SnapshotStatus.SUCCESS.value or SnapshotStatus.FAILURE.value
        eval_status: Optional[str] = None,  # Can be EvalStatus.SUCCESS.value, EvalStatus.FAILURE.value, or EvalStatus.TIMEOUT.value
        file_status: Optional[str] = None,  # Can be FileStatus.SUCCESS.value or FileStatus.FAILURE.value
        profile_status: Optional[str] = None,  # Can be ProfileStatus.SUCCESS.value or ProfileStatus.FAILURE.value
        **kwargs: Any
    ):
        """Initialize CommandResult with required and optional attributes."""
        self.success = success
        self.message = message
        self.error = error
        self.data = data
        self.stdout = stdout
        self.stderr = stderr
        self.edit_status = edit_status
        self.snapshot_status = snapshot_status
        self.eval_status = eval_status
        self.file_status = file_status
        self.profile_status = profile_status
        
        # Add any additional keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __str__(self) -> str:
        """String representation showing success and message/error."""
        if self.success:
            return f"Command succeeded: {self.message[:100]}..."
        else:
            return f"Command failed: {self.error or 'Unknown error'}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the CommandResult to a dictionary for API responses."""
        result = {
            "success": self.success
        }
        
        # Add all non-None attributes to the dictionary
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                if value is not None and attr != 'success':  # Already added success
                    result[attr] = value
                    
        return result
