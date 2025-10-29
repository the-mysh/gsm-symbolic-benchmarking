import logging
import coloredlogs
from pathlib import Path
from datetime import datetime
import os


local_logger = logging.getLogger(__name__)


def install_colored_logger(log: logging.Logger | None = None, level: int | str = logging.DEBUG) -> None:
    """Set up a colored logging output.

    Args:
        log     :   The logger to install as a colored logger. If None (default), the root logger is used.
        level   :   The minimum level of messages for them to be displayed in the console logs.

    Both arguments are passed to 'coloredlogs.install'.
    """

    if log is None:
        log = logging.getLogger('')

    coloredlogs.install(
        logger=log,
        fmt='%(name)s [%(levelname)s] %(asctime)s: %(message)s',
        level=level,
        level_styles={
            'debug': {'color': 35},
            'info': {'color': 74},
            'warning': {'color': 190},
            'error': {'color': 124}
        },
        field_styles={
            'asctime': {'color': 102},
            'name': {'color': 102},
            'levelname': {'color': 231}
        }
    )
    
def setup_log_file_handler(root_logs_path: str | Path, logger: logging.Logger | None = None) -> None:
    """
    Set up a logging handler saving logs to a file.

    Define a logging formatter with time, level name, and logger name.
    Set minimum log level to be saved to DEBUG.

    Args:
        root_logs_path  :   Path to the logs directory. Log file name will be created using the current timestamp.
        logger          :   Logger instance concerned. If None (default), the root logger is used.
    """
    
    # sanitise root path
    if not isinstance(root_logs_path, Path):
        root_logs_path = Path(root_logs_path)
    root_logs_path = root_logs_path.resolve()
    
    # make sure root path exists
    if not root_logs_path.exists():
        local_logger.warning(f"Root logs path ('{root_logs_path}') does not exist. New directory will be created.")
        os.makedirs(root_logs_path)
        
    # create log file name with current timestamp
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_name = root_logs_path / f"{ts}.log"

    # define log format
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s | %(message)s')

    # set up handler
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    if logger is None:
        logger = logging.getLogger('')  # get root logger

    logger.addHandler(fh)

    local_logger.info(f"Python logs will be stored to file: {log_file_name}")
    