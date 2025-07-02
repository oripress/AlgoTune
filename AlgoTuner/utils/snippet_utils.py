"""Utility functions for computing snippet bounds and ranges."""


def compute_centered_snippet_bounds(
    center_line: int, total_lines: int, max_lines: int
) -> tuple[int, int]:
    """
    Given a center line, compute a snippet of at most max_lines lines around it.
    Returns (start_line, end_line) where both are 1-indexed and inclusive.
    """
    half = max_lines // 2
    start = max(1, center_line - half)
    end = min(total_lines, start + max_lines - 1)

    # If we hit the end but have room at start, use it
    if end == total_lines and (end - start + 1) < max_lines:
        start = max(1, end - max_lines + 1)
    # If we hit the start but have room at end, use it
    elif start == 1 and (end - start + 1) < max_lines:
        end = min(total_lines, start + max_lines - 1)

    return start, end


def compute_snippet_bounds(
    start_line: int, end_line: int, total_lines: int, max_lines: int
) -> tuple[int, int]:
    """
    Given a range [start_line..end_line], compute a snippet of at most max_lines lines that includes this range.
    If the range itself is bigger than max_lines, we'll return a snippet centered on the range.
    Returns (snippet_start, snippet_end) where both are 1-indexed and inclusive.
    """
    range_size = end_line - start_line + 1

    if range_size >= max_lines:
        # If range is too big, center around its midpoint
        center = (start_line + end_line) // 2
        return compute_centered_snippet_bounds(center, total_lines, max_lines)

    # Otherwise, try to show max_lines with the range in the middle
    padding = (max_lines - range_size) // 2
    snippet_start = max(1, start_line - padding)
    snippet_end = min(total_lines, snippet_start + max_lines - 1)

    # If we hit the end but have room at start, use it
    if snippet_end == total_lines and (snippet_end - snippet_start + 1) < max_lines:
        snippet_start = max(1, snippet_end - max_lines + 1)
    # If we hit the start but have room at end, use it
    elif snippet_start == 1 and (snippet_end - snippet_start + 1) < max_lines:
        snippet_end = min(total_lines, snippet_start + max_lines - 1)

    return snippet_start, snippet_end
