import logging
import coloredlogs


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