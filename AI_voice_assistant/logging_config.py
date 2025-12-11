import logging
import sys
import os

def setup_logging(log_filename="application.log"):
    """
    Configures logging to write to a file and NOT to the console (stdout).
    This allows keeping the main console clean for user interaction.
    """
    # Determine absolute path for the log file to ensure it's the same file the .bat watches
    # (Assuming .bat is in the same directory as this script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_dir, log_filename)

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File Handler - Overwrites file on each run ('w' mode)
    # Using 'a' (append) might be safer with the tailing script running, 
    # but 'w' should be fine if not locked. Let's try 'a' to be safe against locking issues.
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    
    # Log a startup message to confirm it's working
    logger.info(f"Logging initialized. Writing to: {log_file_path}")

