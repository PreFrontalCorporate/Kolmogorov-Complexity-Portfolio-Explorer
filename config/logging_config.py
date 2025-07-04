import logging

# logging_config.py

def setup_logging(log_level='DEBUG'):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger('portfolio_optimizer')
    
    # Add a file handler (optional)
    file_handler = logging.FileHandler('portfolio_optimizer.log')
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Setup logging with default log level from config.py
logger = setup_logging(log_level='DEBUG')
