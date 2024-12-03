import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    file_handler = logging.FileHandler('TEA_LOG.log', encoding='utf-8', mode='w')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
