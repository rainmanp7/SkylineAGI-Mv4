
# Beginning of logging_config.py
# Updated Sun Dec 8th 2024
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    # Create a rotating file handler
    handler = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=5)  # 5 MB per file
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
        logger.info("Application finished.")

if __name__ == "__main__":
    main()
# End of logging_config.py
