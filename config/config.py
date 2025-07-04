# config.py

class Config:
    # Optimization parameters
    LAMBDA_0 = 0.1  # L0 penalty (sparsity)
    LAMBDA_1 = 0.01  # L1 penalty (magnitude)
    
    # Risk metrics thresholds
    VAR_ALPHA = 0.05  # Value-at-Risk confidence level (5%)
    CVAR_ALPHA = 0.05  # Conditional Value-at-Risk confidence level (5%)
    
    # Portfolio constraints (if any)
    MAX_WEIGHT = 0.1  # Maximum allowable weight for any asset
    
    # Path for saving model data (optional)
    MODEL_PATH = "/path/to/save/models"
    
    # Logging configuration (can be overridden in logging_config.py)
    LOG_LEVEL = "DEBUG"
