import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Log file name with timestamp
# The log file will be created in the logs directory with the current date and time

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # Creating a path for the logs directory
# The logs directory will be created in the current working directory

os.makedirs(logs_path,exist_ok=True) # Creating the logs directory if it does not exist
# exist_ok=True means that if the directory already exists, it will not raise an error

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE) # Full path for the log file 

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
