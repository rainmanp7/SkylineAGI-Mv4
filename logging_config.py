# Beginning of logging_config.py
# Updated Sun Dec 8th 2024
# corrected to work with Jupyter Notebook

import logging
from logging.handlers import RotatingFileHandler
import os
import time  # For sleep
from threading import Event  # For async_ops_completed simulation

# Create a global event for asynchronous operation completion simulation
async_ops_completed = Event()

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    log_file = "app.log"
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else os.getcwd()

    # Check if log directory is writable
    if not os.access(log_dir, os.W_OK):
        log_dir = os.path.expanduser("~/logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")
        print(f"Falling back to log directory: {log_dir}")

    # Get the root logger
    logger = logging.getLogger()
    
    # Avoid duplicate handlers
    if not logger.hasHandlers():
        try:
            # Create a rotating file handler
            handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB per file
        except PermissionError:
            logging.error(f"No write access to log file: {log_file}. Falling back to console logging.")
            handler = logging.StreamHandler()
        except Exception as e:
            logging.error(f"Unexpected error while setting up file handler: {e}. Falling back to console logging.")
            handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())  # Log to console

# Call this function at the start of your application
setup_logging()

# Example usage of logging in your application
logger = logging.getLogger(__name__)

def process_data(data):
    """Process the given data."""
    logger.info("Processing data: %s", data)
    # Simulate processing logic
    if data == "error":
        raise ValueError("An error occurred while processing data.")

def main():
    """Main function to run the application."""
    logger.info("Application started.")
    try:
        # Example data processing
        process_data("sample data")
        # Uncomment to test error handling
        # process_data("error")  # This will raise an exception
    except Exception as e:
        logger.exception("An error occurred: %s", e)
    finally:
        # Verify asynchronous operation completion (if applicable)
        async_ops_completed.set()  # Simulate async operation completion

        # Short delay for system resource release (optional)
        time.sleep(0.1)

        logger.info("Operation completed.")

if __name__ == "__main__":
    main()
# End of logging_config.py
