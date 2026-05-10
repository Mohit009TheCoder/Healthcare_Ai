# logger_config.py
"""
Logging configuration for healthcare prediction system.
Sets up file-based logging with rotation for application and error logs.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(app):
    """
    Configure application logging with file handlers.
    
    Creates two log files:
    - healthcare_app.log: All logs (INFO and above)
    - errors.log: Error logs only (ERROR and above)
    
    Args:
        app: Flask application instance
    """
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
        print("✅ Created logs directory")
    
    # File handler for all logs
    file_handler = RotatingFileHandler(
        'logs/healthcare_app.log',
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    
    # File handler for errors only
    error_handler = RotatingFileHandler(
        'logs/errors.log',
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]\n'
        'Exception: %(exc_info)s'
    ))
    error_handler.setLevel(logging.ERROR)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to app logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(error_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)
    
    # Log startup
    app.logger.info('=' * 60)
    app.logger.info(f'Healthcare App Started - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    app.logger.info('=' * 60)
    
    return app.logger


def log_prediction(logger, prediction_type, inputs, result, duration_ms=None):
    """
    Log a prediction with inputs and results.
    
    Args:
        logger: Application logger
        prediction_type: Type of prediction (diabetes, heart, symptoms)
        inputs: Dictionary of input values
        result: Prediction result
        duration_ms: Prediction duration in milliseconds
    """
    log_msg = f"Prediction [{prediction_type}] - Result: {result.get('prediction', 'N/A')}"
    if 'confidence' in result:
        log_msg += f", Confidence: {result['confidence']}%"
    if duration_ms:
        log_msg += f", Duration: {duration_ms}ms"
    
    logger.info(log_msg)
