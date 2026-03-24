import logging

from config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    return logger
