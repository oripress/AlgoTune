import difflib


def unified_diff(
    old: str,
    new: str,
    fromfile: str = "current",
    tofile: str = "proposed",
    n_context: int = 3
) -> str:
    """Return a unified diff string between two code blobs."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
        n=n_context,
    )
    return "".join(diff) 