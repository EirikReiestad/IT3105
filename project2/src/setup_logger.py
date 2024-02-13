import logging


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Open the file in write mode ('w') to overwrite the previous log
    handler = logging.FileHandler('debug.log', 'w')
    handler.setLevel(logging.DEBUG)

    # Include '%(asctime)s' to include the time of the log
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    formatter.datefmt = '%H:%M:%S'
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
