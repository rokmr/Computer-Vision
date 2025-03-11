import logging
import os
from datetime import datetime
from typing import Optional

class RelativePathFormatter(logging.Formatter):
    def format(self, record):
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Convert absolute path to relative path from project root
        if hasattr(record, 'pathname'):
            record.pathname = os.path.relpath(record.pathname, project_root)
        return super().format(record)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = RelativePathFormatter(
        '%(pathname)s:%(lineno)d - %(levelname)s - %(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = RelativePathFormatter(
            '%(pathname)s:%(lineno)d - %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_log_file(module_name: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join('logs', f'{module_name}_{timestamp}.log')
    print(f"Log file will be created at: {os.path.abspath(log_path)}")
    return log_path 

LOGGER_NAME = "cricket_classifier"
logger = setup_logger(LOGGER_NAME, get_log_file(LOGGER_NAME)) 