import logging
import sys
from datetime import datetime

# ANSI escape codes for colors
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RED = "\033[31m"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelno, RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = record.levelname.ljust(8)
        log_message = record.getMessage()
        
        return f"{timestamp} | {level_color}{level_name}{RESET} | {log_message}"


def setup_logging(level=logging.INFO):
    """Configure logging with color and timestamps."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler with our formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    return logger


# Get the root logger
logger = setup_logging() 