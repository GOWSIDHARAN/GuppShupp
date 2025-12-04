"""
Logging Configuration
Production-ready logging setup for the GuppShupp application
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure log format
    log_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs
    log_file = log_path / f"guppshupp_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_log_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    root_logger.addHandler(error_handler)
    
    # Security audit log
    security_log_file = log_path / f"security_{datetime.now().strftime('%Y%m%d')}.log"
    security_handler = logging.handlers.RotatingFileHandler(
        security_log_file,
        maxBytes=2*1024*1024,  # 2MB
        backupCount=10
    )
    security_handler.setLevel(logging.INFO)
    security_handler.setFormatter(log_format)
    
    # Create security logger
    security_logger = logging.getLogger('security')
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)
    security_logger.propagate = False  # Don't duplicate to root logger
    
    # API request logger
    api_log_file = log_path / f"api_{datetime.now().strftime('%Y%m%d')}.log"
    api_handler = logging.handlers.RotatingFileHandler(
        api_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(log_format)
    
    # Create API logger
    api_logger = logging.getLogger('api')
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.INFO)
    api_logger.propagate = False
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")


class SecurityLogger:
    """Specialized logger for security events"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_access_attempt(self, user_id: str, ip_address: str, endpoint: str, success: bool):
        """Log API access attempts"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"ACCESS_ATTEMPT - User: {user_id}, IP: {ip_address}, "
                        f"Endpoint: {endpoint}, Status: {status}")
    
    def log_memory_access(self, user_id: str, requested_user: str, authorized: bool):
        """Log memory access attempts"""
        status = "AUTHORIZED" if authorized else "UNAUTHORIZED"
        self.logger.info(f"MEMORY_ACCESS - Requester: {user_id}, Target: {requested_user}, "
                        f"Status: {status}")
    
    def log_data_extraction(self, user_id: str, message_count: int, confidence_score: float):
        """Log memory extraction events"""
        self.logger.info(f"MEMORY_EXTRACTION - User: {user_id}, Messages: {message_count}, "
                        f"Confidence: {confidence_score:.2f}")
    
    def log_personality_switch(self, user_id: str, from_personality: str, to_personality: str):
        """Log personality changes"""
        self.logger.info(f"PERSONALITY_SWITCH - User: {user_id}, From: {from_personality}, "
                        f"To: {to_personality}")


class APILogger:
    """Specialized logger for API requests and responses"""
    
    def __init__(self):
        self.logger = logging.getLogger('api')
    
    def log_request(self, method: str, endpoint: str, user_id: str, 
                   processing_time: float, status_code: int):
        """Log API request details"""
        self.logger.info(f"REQUEST - {method} {endpoint} - User: {user_id} - "
                        f"Time: {processing_time:.2f}s - Status: {status_code}")
    
    def log_error(self, method: str, endpoint: str, user_id: str, error: str):
        """Log API errors"""
        self.logger.error(f"ERROR - {method} {endpoint} - User: {user_id} - "
                         f"Error: {error}")
    
    def log_performance(self, endpoint: str, operation: str, duration: float):
        """Log performance metrics"""
        self.logger.info(f"PERFORMANCE - {endpoint} - {operation} - "
                        f"Duration: {duration:.2f}s")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)
