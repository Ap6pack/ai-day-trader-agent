import logging

def get_logger(name="ai_day_trader"):
    """
    Returns a logger with a standard format.
    Logs to both console and ai_day_trader.log file.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # File handler
        import os
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "ai_day_trader.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
