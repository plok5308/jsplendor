import logging

class BaseLogger:
    _instance = None
    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            # Create logger
            cls._logger = logging.getLogger(cls.__name__)
            cls._logger.setLevel(logging.INFO)
            
            # Create console handler if none exists
            if not cls._logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(message)s')
                console_handler.setFormatter(formatter)
                cls._logger.addHandler(console_handler)
        
        return cls._logger 