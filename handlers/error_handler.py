import logging
import traceback

logger = logging.getLogger("myapp")

def handle_error(error: Exception, context: str = ""):
    """
    Logs the error with traceback and context information.
    
    Parameters:
        error (Exception): The exception object caught.
        context (str): Optional additional context where the error occurred.
    """
    error_message = f"An error occurred in {context}: {str(error)}"
    # Log the error message and full traceback for debugging purposes
    logger.error(error_message)
    logger.error(traceback.format_exc())
    
    # Optionally, you can add additional error processing here
    # For example, sending alerts or performing clean-up actions.

    # Return the error message if needed
    return error_message

# Example usage:
if __name__ == "__main__":
    try:
        # Simulate code that may raise an exception
        1/0
    except Exception as e:
        handle_error(e, context="Main execution")