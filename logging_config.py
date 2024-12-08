# Beginning of logging_config.py
# Updated Sun Dec 8th 2024
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    log_file = "app.log"
    log_dir = os.path.dirname(log_file)

    # Check if log directory is writable
    if not os.access(log_dir, os.W_OK):
        logging.error(f"No write access to log directory: {log_dir}")
        print(f"Error: No write access to log directory: {log_dir}")
        # Fallback to console logging only
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.addHandler(logging.StreamHandler())  # Log to console
        print("Falling back to console logging only.")
        return

    # Create a rotating file handler
    try:
        handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)  # 5 MB per file
    except PermissionError:
        logging.error(f"No write access to log file: {log_file}")
        print(f"Error: No write access to log file: {log_file}")
        # Fallback to console logging only
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.addHandler(logging.StreamHandler())  # Log to console
        print("Falling back to console logging only.")
        return
    except Exception as e:
        logging.error(f"An error occurred while setting up logging: {str(e)}")
        print(f"Error: {str(e)}")
        return

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Get the root logger
    logger = logging.getLogger()
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
        process_data("error")  # This will raise an exception
    except Exception as e:
        logger.exception("An error occurred: %s", e)
    finally:
    # Verify asynchronous operation completion (if applicable)
    async_ops_completed.wait()

    # Flush and close resources explicitly (if applicable)
    file_handler.flush()
    file_handler.close()

    # Short delay for system resource release (optional)
    time.sleep(0.1)

    logger.info("Operation completed.")

if __name__ == "__main__":
    main()
# End of logging_config.py
