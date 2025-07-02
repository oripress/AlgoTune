from pathlib import Path

"""
Module providing the generic error message loaded from messages/error_message.txt.
"""
MESSAGE_FILE = Path(__file__).parent.parent / 'messages' / 'error_message.txt'

try:
    GENERIC_ERROR_MESSAGE = MESSAGE_FILE.read_text()
except Exception:
    # Fallback to a default generic message
    GENERIC_ERROR_MESSAGE = (
        "Bad response received. Your response must include some thoughts and a "
        "_SINGLE_ command (sandwiched between two sets of triple backticks). "
        "There should be nothing after the command."
    ) 