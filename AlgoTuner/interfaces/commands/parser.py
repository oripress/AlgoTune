"""
interfaces.commands.parser – robust command parser used by the playground.

This module converts a raw chat message into either a `ParsedCommand`
instance or a structured *error‑dict* that downstream code can surface
verbatim to the user.

Guarantees
~~~~~~~~~~
* **Never** returns the pair `(None, None)` – if the message does not
  contain a recognised command you still receive an error‑dict.
* Gracefully handles messy input such as stray spaces after fences or a
  missing closing ``` fence: a best‑effort fallback recognises common
  patterns so the UI shows a helpful error instead of crashing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

from AlgoTuner.interfaces.commands.types import (
    COMMAND_FORMATS,
    ERROR_MESSAGES,
    ParsedCommand,
)
from AlgoTuner.utils.code_helpers import extract_code_blocks
from AlgoTuner.utils.error_helpers import (
    get_bad_response_error_message,
    get_error_messages_cached,
)

# ──────────────────────────────────────────────────────────────
#  Helper dataclass
# ──────────────────────────────────────────────────────────────


@dataclass
class EditCommand:
    file_name: str
    start_line: int
    end_line: int
    content: str
    is_new_file: bool = False

    def __post_init__(self) -> None:
        self.is_new_file = self.start_line == 0 and self.end_line == 0


# ──────────────────────────────────────────────────────────────
#  Main parser class
# ──────────────────────────────────────────────────────────────


class CommandParser:
    """Parse chat messages into playground commands or rich error dicts."""

    _FILE_RE = re.compile(r"^[\w.\-\\/]+$")


    @staticmethod
    def _quick_fence_extract(msg: str) -> Optional[str]:
        """Return canonical command string if *msg* is a lone fenced block
        even when `extract_code_blocks` fails (e.g. stray spaces)."""
        m = re.match(r"^````?\s*(\w+)\s*\n([\s\S]*?)\n````?\s*\Z", msg.strip())
        if not m:
            return None
        lang, body = m.group(1).strip(), m.group(2)
        first = body.strip().split(maxsplit=1)[0] if body.strip() else ""
        if lang in COMMAND_FORMATS:
            return f"{lang}\n{body}" if body.strip() else lang
        if first in COMMAND_FORMATS:
            return body.strip()
        return None

    # filename & line‑range validators

    @classmethod
    def _validate_filename(cls, name: str) -> Optional[str]:
        if not name:
            return "Filename cannot be empty"
        if cls._FILE_RE.fullmatch(name) is None:
            return ERROR_MESSAGES["invalid_filename"].format(name)
        return None

    @staticmethod
    def _validate_line_spec(spec: str) -> Tuple[Optional[Tuple[int, int]], Optional[dict]]:
        try:
            start_s, end_s = spec.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start < 0 or end < 0:
                which = "start" if start < 0 else "end"
                return None, {
                    "success": False,
                    "error": f"{which.title()} line number cannot be negative",
                    "command": "edit",
                    "is_validation_error": True,
                }
            if start and start < 1: # 0 is allowed for 0-0 (new file)
                return None, {
                    "success": False,
                    "error": "Start line must be ≥ 1 (or 0-0 for new file).",
                    "command": "edit",
                    "is_validation_error": True,
                }
            if end < start:
                return None, {
                    "success": False,
                    "error": ERROR_MESSAGES["line_range"].format(end, start),
                    "command": "edit",
                    "is_validation_error": True,
                }
            return (start, end), None
        except ValueError:
            return None, {
                "success": False,
                "error": "Invalid line range; expected start-end (e.g. 1-7).",
                "command": "edit",
                "is_validation_error": True,
            }

    # fenced‑block extraction (robust)

    @classmethod
    def _extract_command_block(
        cls, msg: str
    ) -> Tuple[Optional[str], Optional[dict], int]:
        code_blocks = extract_code_blocks(msg)
        total_actual_code_blocks = len(code_blocks)

        if total_actual_code_blocks == 0:
            return None, None, 0

        first_cmd_str: Optional[str] = None
        first_cmd_block_details: Optional[tuple[str, str]] = None

        for lang, body in code_blocks:
            current_lang = lang.strip()
            current_body_stripped = body.strip()
            first_token_in_body = current_body_stripped.split(maxsplit=1)[0] if current_body_stripped else ""

            is_command = False
            if current_lang in COMMAND_FORMATS:
                is_command = True
            elif first_token_in_body in COMMAND_FORMATS:
                is_command = True

            if is_command:
                first_cmd_block_details = (current_lang, body) 
                break

        if not first_cmd_block_details:
            return None, None, total_actual_code_blocks

        lang, body = first_cmd_block_details
        # Construct cmd_str from the identified first command block
        # This logic mirrors the original construction for a single command block
        processed_lang = lang.strip()
        processed_body = body.strip()

        if processed_lang in COMMAND_FORMATS and processed_body:
            first_cmd_str = f"{processed_lang}\n{body}" # Keep original body for multi-line args
        elif processed_lang in COMMAND_FORMATS: # lang is command, body is empty
            first_cmd_str = processed_lang
        else: # lang is not command, so first token of body was command
            first_cmd_str = body.strip() # Body contains the full command and its args

        if not first_cmd_str:
             return None, {
                "success": False,
                "error": "Internal error: Failed to construct command string from the first identified block.",
                "command": "internal_error_parsing_first_block",
                "is_parsing_error": True,
            }, total_actual_code_blocks
        
        # Return 1 to signify one command block was chosen and successfully extracted for further parsing.
        # This tells CommandParser.parse that it's a fenced command.
        return first_cmd_str, None, 1

    # public parse

    @classmethod
    def parse(
        cls, message: str
    ) -> Tuple[Optional[ParsedCommand], Optional[dict], int]:
        logging.info(f"PARSER_ENTRY: Received message ({len(message)} chars):\n{message[:500]}...")
        
        # Strip thinking blocks before processing
        original_message = message
        message = cls._strip_thinking_blocks(message)
        if message != original_message:
            logging.info(f"PARSER: Stripped thinking blocks, message reduced from {len(original_message)} to {len(message)} chars")
        
        # Early rejection of system-generated content that should not be parsed as commands
        if cls._is_system_generated_content(message):
            logging.info("PARSER: Detected system-generated content, rejecting parse attempt")
            return None, {
                "success": False,
                "error": "System-generated content should not be parsed as commands",
                "command": "system_content",
                "is_parsing_error": True,
            }, 0
        
        text = message.strip()
        first_tok = text.split(maxsplit=1)[0] if text else ""

        if first_tok in COMMAND_FORMATS:
            cmd_str, err, blocks = text, None, 0
        else:
            cmd_str, err, blocks = cls._extract_command_block(message)
            if err:
                return None, err, blocks
            if cmd_str is None:
                cmd_str = cls._quick_fence_extract(message)
                if cmd_str is None:
                    return None, {
                        "success": False,
                        "error": get_bad_response_error_message(),
                        "command": "no_command_block",
                        "is_parsing_error": True,
                    }, blocks
                blocks = max(blocks, 1) # _quick_fence_extract implies at least one block-like structure

        # Enforce no trailing text for single-line commands
        if blocks == 0:
            lines = message.splitlines()
            # Locate the command line
            cmd_line = None
            for idx, line in enumerate(lines):
                if line.strip().startswith(cmd_str.strip()):
                    cmd_line = idx
                    break
            # If found, ensure all subsequent lines are blank
            if cmd_line is not None:
                for extra in lines[cmd_line+1:]:
                    if extra.strip():
                        return None, {
                            "success": False,
                            "error": ERROR_MESSAGES["empty_command"],
                            "command": first_tok,
                            "is_parsing_error": True,
                        }, blocks

        logging.info(f"PRE_TRAIL_CHECK: blocks={blocks}, cmd_str='{cmd_str[:100] if cmd_str else 'None'}'")
        keyword = cmd_str.splitlines()[0].split(maxsplit=1)[0] if cmd_str else "unknown_keyword_due_to_None_cmd_str"

        # Enforce no trailing text for single-line commands
        if blocks == 0:
            lines = message.splitlines()
            # Locate the command line
            cmd_line = None
            for idx, line in enumerate(lines):
                if line.strip().startswith(cmd_str.strip()):
                    cmd_line = idx
                    break
            # If found, ensure all subsequent lines are blank
            if cmd_line is not None:
                for extra in lines[cmd_line+1:]:
                    if extra.strip():
                        return None, {
                            "success": False,
                            "error": ERROR_MESSAGES["empty_command"],
                            "command": first_tok,
                            "is_parsing_error": True,
                        }, blocks

        # Enforce no text after the command block (allow text before)
        if blocks > 0:
            lines = message.splitlines()
            logging.info(f"TRAIL_CHECK: Full message for trailing check ({len(lines)} lines):\n{message[:500]}...")
            
            fence_lines = [i for i, l in enumerate(lines) if re.match(r'^\s*`{3,}\s*$', l)]
            logging.info(f"TRAIL_CHECK: Pure fence lines indices: {fence_lines}")

            cmd_close_line_idx = None
            if len(fence_lines) >= 2:
                cmd_close_line_idx = fence_lines[1] 
            elif fence_lines: 
                cmd_close_line_idx = fence_lines[0]
            
            logging.info(f"TRAIL_CHECK: Determined cmd_close_line_idx: {cmd_close_line_idx}")

            if cmd_close_line_idx is not None:
                logging.info(f"TRAIL_CHECK: Checking for text after line index {cmd_close_line_idx}.")
                for i, extra_line_content in enumerate(lines[cmd_close_line_idx + 1:], start=cmd_close_line_idx + 1):
                    logging.info(f"TRAIL_CHECK: Checking line {i}: '{extra_line_content[:100]}'")
                    if extra_line_content.strip():
                        logging.error(f"TRAIL_CHECK: Found trailing text on line {i}: '{extra_line_content.strip()[:100]}'. Erroring out.")
                        return None, {
                            "success": False,
                            "error": f"Found trailing text after command block: '{extra_line_content.strip()[:60]}...'", 
                            "command": keyword,
                            "is_parsing_error": True,
                        }, blocks
                logging.info(f"TRAIL_CHECK: No trailing text found after line index {cmd_close_line_idx}.")
            else:
                logging.warning("TRAIL_CHECK: Could not determine command's closing fence for trailing text check.")
        
        cmd_format = COMMAND_FORMATS.get(keyword)
        if not cmd_format:
            return None, {
                "success": False,
                "error": f"Unknown command '{keyword}':\n{get_error_messages_cached()}",
                "command": keyword,
                "is_parsing_error": True,
            }, blocks

        # EDIT
        if keyword == "edit":
            parsed, perr = cls._parse_edit(cmd_str)
            if perr:
                return None, perr, blocks
            # Ensure parsed is not None, though perr check should cover this
            if parsed is None: # Defensive check
                 return None, {
                    "success": False,
                    "error": "Internal error parsing edit command.", # Should not happen if perr is None
                    "command": "edit",
                    "is_parsing_error": True,
                }, blocks
            args = {
                "file_name": parsed.file_name,
                "start_line": parsed.start_line,
                "end_line": parsed.end_line,
                "new_content": parsed.content,
                "is_new_file": parsed.is_new_file,
            }
            return ParsedCommand(keyword, args, blocks, cmd_str), None, blocks

        # one‑liner commands
        # Strip cmd_str for one-liners as their regexes usually expect no trailing space/newlines
        # unless the pattern itself accounts for it (like for body content).
        m = cmd_format.pattern.fullmatch(cmd_str.strip())
        if not m:
            return None, {
                "success": False,
                "error": cmd_format.error_message,
                "command": keyword,
                "is_parsing_error": True,
            }, blocks

        g = m.groups()
        args: Dict[str, object] = {}

        if keyword == "view_file":
            args = {"file_name": g[0]}
            if len(g) > 1 and g[1]: # Optional line number
                try:
                    args["start_line"] = int(g[1])
                    if args["start_line"] < 1:
                         return None, {
                            "success": False,
                            "error": "Start line must be ≥ 1.",
                            "command": keyword,
                            "is_validation_error": True,
                        }, blocks
                except ValueError:
                     return None, {
                        "success": False,
                        "error": "Invalid start line number.",
                        "command": keyword,
                        "is_validation_error": True,
                    }, blocks
        elif keyword == "delete":
            s, e = int(g[1]), int(g[2])
            if s <= 0 or e <= 0 or e < s:
                return None, {
                    "success": False,
                    "error": "Invalid line range for delete: start and end must be > 0 and end >= start.",
                    "command": keyword,
                    "is_validation_error": True, # Changed from is_parsing_error
                }, blocks
            args = {"file_name": g[0], "start_line": s, "end_line": e}
        elif keyword in {"oracle", "eval_input"}:
            args = {"input_str": g[0].strip(), "body": g[0].strip()}
        elif keyword == "profile":
            args = {"filename": g[0], "input_str": (g[1] or "").strip()}
        elif keyword == "profile_lines":
            nums_str = g[1] or ""
            nums: List[int] = []
            seen_lines = set()
            if nums_str.strip():
                try:
                    for part in nums_str.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        if "-" in part:
                            start_end = part.split("-")
                            if len(start_end) != 2:
                                raise ValueError(f"Invalid range: '{part}'. Ranges must be in the form start-end, e.g. 3-7.")
                            start, end = start_end
                            if not start.strip().isdigit() or not end.strip().isdigit():
                                raise ValueError(f"Invalid range: '{part}'. Both start and end must be positive integers.")
                            start = int(start.strip())
                            end = int(end.strip())
                            if start <= 0 or end <= 0:
                                raise ValueError(f"Line numbers must be positive: '{part}'")
                            if end < start:
                                raise ValueError(f"Range start must be <= end: '{part}'")
                            for n in range(start, end + 1):
                                if n not in seen_lines:
                                    nums.append(n)
                                    seen_lines.add(n)
                        else:
                            if not part.isdigit():
                                raise ValueError(f"Invalid line number: '{part}'. Must be a positive integer.")
                            n = int(part)
                            if n <= 0:
                                raise ValueError(f"Line numbers must be positive: '{part}'")
                            if n not in seen_lines:
                                nums.append(n)
                                seen_lines.add(n)
                except ValueError as ve:
                    return None, {
                        "success": False,
                        "error": f"Invalid line numbers for profiling: {ve}. Expected comma-separated integers or ranges (e.g. 1,3-5,7).",
                        "command": keyword,
                        "is_validation_error": True,
                    }, blocks
            args = {
                "filename": g[0],
                "focus_lines": nums,
                "input_str": (g[2] or "").strip(),
            }

        return ParsedCommand(keyword, args, blocks, cmd_str), None, blocks

    # _parse_edit helper (logic unchanged)

    @classmethod
    def _parse_edit(
        cls, raw: str
    ) -> Tuple[Optional[EditCommand], Optional[dict]]:
        """Parse the body of a multi‑line `edit` command (after stripping
        its surrounding code fence).
        Expected layout::

            edit
            file: path/to/file.py
            lines: 3-7
            ---
            <new content>
            ---
        """
        lines = raw.splitlines()
        if not lines or lines[0].strip().split(maxsplit=1)[0] != "edit":
             return None, { # Should not happen if called correctly
                "success": False, "error": "Edit command format error (missing 'edit' keyword).",
                "command": "edit", "is_parsing_error": True
            }

        iterator = iter(lines[1:])  # skip the top "edit" line

        file_spec: Optional[str] = None
        line_spec_str: Optional[str] = None # Renamed to avoid clash with _validate_line_spec's return
        content: List[str] = []
        seen: Dict[str, int] = {"file": 0, "lines": 0}
        dashes = 0
        in_body = False

        for ln_idx, ln in enumerate(iterator):
            s = ln.strip()
            if s == "---":
                dashes += 1
                if dashes == 1:
                    in_body = True
                elif dashes == 2:
                    in_body = False # Body ends after second ---
                else: # More than 2 dashes
                    return None, {
                        "success": False,
                        "error": "Edit command has too many '---' delimiters.",
                        "command": "edit",
                        "is_validation_error": True,
                    }
                continue

            if in_body:
                # Add the raw line (from original splitlines) to preserve original spacing,
                # newlines will be added by join later.
                # The original `ln` from `raw.splitlines()` does not have trailing newlines.
                content.append(ln) # `ln` is the line content without trailing newline
                continue

            # Header parsing, only if not in body and dashes < 2
            if dashes >= 2: # Should not parse headers after the second '---'
                if s: # Non-empty line after content and second '---'
                    return None, {
                        "success": False,
                        "error": "Unexpected content after closing '---' delimiter.",
                        "command": "edit",
                        "is_validation_error": True,
                    }
                continue


            if s.startswith("file:"):
                if seen["file"]:
                    return None, {
                        "success": False,
                        "error": ERROR_MESSAGES["duplicate_specification"].format("file"),
                        "command": "edit",
                        "is_validation_error": True,
                    }
                seen["file"] += 1
                file_spec = s[5:].strip()
            elif s.startswith("lines:"):
                if seen["lines"]:
                    return None, {
                        "success": False,
                        "error": ERROR_MESSAGES["duplicate_specification"].format("lines"),
                        "command": "edit",
                        "is_validation_error": True,
                    }
                seen["lines"] += 1
                line_spec_str = s[6:].strip()
            elif s: # Non-empty line that is not a spec and not in body before first ---
                 return None, {
                    "success": False,
                    "error": f"Unexpected content in edit command header: '{s}'. Expected 'file:', 'lines:', or '---'.",
                    "command": "edit",
                    "is_validation_error": True,
                }


        # mandatory components present?
        missing: List[str] = []
        if file_spec is None: # Check for None explicitly, as empty string is caught by _validate_filename
            missing.append("'file:' specification")
        if line_spec_str is None:
            missing.append("'lines:' specification")
        if dashes != 2: # Must be exactly two '---'
             err_msg = "Edit command missing "
             if dashes < 2:
                 err_msg += f"{'one' if dashes == 1 else 'two'} '---' delimiter{'s' if dashes == 0 else ''}."
             else: # Should be caught by dashes > 2 check already
                 err_msg += "too many '---' delimiters."

             return None, {
                "success": False,
                "error": err_msg,
                "command": "edit",
                "is_validation_error": True,
            }

        if missing:
            return None, {
                "success": False,
                "error": f"Edit command missing {', '.join(missing)}.",
                "command": "edit",
                "is_validation_error": True,
            }

        # filename & line‑range validation
        # file_spec and line_spec_str are guaranteed to be non-None here by previous checks
        name_err = cls._validate_filename(file_spec) # type: ignore
        if name_err:
            return None, {
                "success": False,
                "error": name_err,
                "command": "edit",
                "is_validation_error": True,
            }

        line_range, range_err = cls._validate_line_spec(line_spec_str) # type: ignore
        if range_err:
            # range_err already has command: edit and is_validation_error: True
            return None, range_err
        
        # line_range is guaranteed to be non-None if range_err is None
        start, end = line_range  # type: ignore

        # Reconstruct the body with newlines
        body = "\n".join(content)

        # Content empty check:
        # For new files (0-0), empty content is allowed.
        # For existing line ranges, content should not be empty unless it's an explicit deletion
        # (which 'edit' is not; 'delete' command is for that).
        # So, if not a new file, content must exist.
        if not body and not (start == 0 and end == 0):
            return None, {
                "success": False,
                "error": ERROR_MESSAGES["empty_content"],
                "command": "edit",
                "is_validation_error": True,
            }

        return EditCommand(file_name=file_spec, start_line=start, end_line=end, content=body), None

    @classmethod
    def _strip_thinking_blocks(cls, message: str) -> str:
        """
        Remove <think>...</think> blocks from the message.
        
        These blocks contain reasoning tokens that should not be parsed as commands.
        
        Args:
            message: The message content to process
            
        Returns:
            Message with thinking blocks removed
        """
        # Remove <think>...</think> blocks (case-insensitive, multiline)
        pattern = r'<think\b[^>]*>.*?</think>'
        cleaned = re.sub(pattern, '', message, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace that might be left behind
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Reduce multiple blank lines
        
        return cleaned.strip()

    @classmethod
    def _is_system_generated_content(cls, message: str) -> bool:
        """
        Detect if the message is system-generated content that should not be parsed as commands.
        
        Args:
            message: The message content to check
            
        Returns:
            True if this appears to be system-generated content, False otherwise
        """
        lines = message.splitlines()
        
        # Check for formatted diff content (lines like "| 01: from typing import Dict, List, Any")
        if len(lines) > 5:
            diff_line_count = sum(1 for line in lines[:15] if re.match(r'^\s*[|>]\s*\d+:', line))
            if diff_line_count >= 3:  # If we see 3+ diff-style lines in the first 15 lines
                return True
        
        # Check for error message content
        if "Command parsing failed" in message and "triple backticks" in message:
            return True
            
        # Check for edit failure messages with specific patterns
        if ("Edit failed" in message and 
            ("Proposed changes" in message or "CURRENT FILE" in message)):
            return True
            
        # Check for file view content (starts with "Contents of" and has line numbers)
        if message.startswith("Contents of ") and " (lines " in message[:100]:
            return True
            
        # Check for evaluation output with specific patterns
        if ("Runtime:" in message and "Output:" in message) or "Evaluation Failed:" in message:
            return True
            
        return False