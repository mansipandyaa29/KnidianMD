#logs everything into a directory called logs to see whats been going on
import sys
import logging
import os
from datetime import datetime

# Create the logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Define the log file path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)