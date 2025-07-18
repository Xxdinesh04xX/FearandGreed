"""
Logging configuration for GoQuant Sentiment Trader.
"""

import logging
import sys
from typing import Optional
from loguru import logger


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Remove default loguru handler
    logger.remove()
    
    # Add custom handler with formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler for persistent logging
    logger.add(
        "logs/goquant_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    # Create a standard library logger that forwards to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Set up standard library logger
    stdlib_logger = logging.getLogger(name)
    stdlib_logger.handlers = [InterceptHandler()]
    stdlib_logger.setLevel(getattr(logging, level.upper()))
    
    return stdlib_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified name."""
    return logging.getLogger(name)
