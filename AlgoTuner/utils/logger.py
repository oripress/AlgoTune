import os
import logging
from datetime import datetime
import sys


def setup_logging(task=None, model=None):
    """
    Set up logging configuration with both file and console handlers.
    File handler will log DEBUG and above, while console will only show INFO and above.

    Args:
        task (str, optional): Task name to include in log filename
        model (str, optional): Model name to include in log filename

    Returns:
        logging.Logger: Configured logger instance

    Raises:
        OSError: If log directory cannot be created or log file cannot be written
    """
    # Create logs directory if it doesn't exist
    try:
        os.makedirs("logs", exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create logs directory: {e}")

    # Generate log filename with timestamp and task/model info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components = []
    
    # Create log file for all runs, including test runs
    if task:
        components.append(task)
        if model:
            # Extract only the part after the slash if it exists
            model_name = model.split("/")[-1] if "/" in model else model
            components.append(model_name)
    components.append(timestamp)
    log_file = os.path.join("logs", f"{'_'.join(components)}.log")

    # Print confirmation to verify this function is being called
    if log_file:
        print(f"Setting up logging to file: {log_file}")

    # Clear existing handlers to prevent duplicate handlers
    # This is critical when the function is called multiple times
    logger = logging.getLogger()
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        print("Removed existing log handlers")

    # Set root logger to DEBUG level to allow all logging
    logger.setLevel(logging.DEBUG)

    # Create handlers
    if log_file:
        # Verify log file is writable
        try:
            with open(log_file, "a"):
                pass
        except OSError as e:
            raise OSError(f"Cannot write to log file {log_file}: {e}")
            
        file_handler = logging.FileHandler(log_file, 'w')  # Use 'w' mode to create/overwrite file
        file_handler.setLevel(logging.INFO)  # Set file handler to INFO level to reduce verbosity
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set console handler to INFO level only
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))  # Simpler format for console
    logger.addHandler(console_handler)
    
    # Log initial messages
    logger.debug("Logger initialized with DEBUG level file logging")
    if log_file:
        logger.info(f"Logging to file: {log_file}")

    return logger
