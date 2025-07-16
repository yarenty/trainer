import logging

def setup_logging(verbose=False):
    """Set up logging for the project. Use DEBUG level if verbose, else INFO."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    ) 