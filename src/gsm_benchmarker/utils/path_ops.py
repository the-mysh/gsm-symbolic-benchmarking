import os
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def confirm_or_create_folder(folder_path: Path | str) -> Path:
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    if not isinstance(folder_path, Path):
        raise TypeError(f"Expected a str or a Path object; got {type(folder_path)}: {folder_path}")

    if folder_path.is_file():
        raise RuntimeError(f'Expected a path to a directory; got a file path instead: {folder_path}')

    if not folder_path.exists():
        if not folder_path.parent.exists():
            logger.warning(f"Storage root does not exist and will be created")
        logger.info(f"Creating directories at {folder_path}")
        os.makedirs(folder_path)

    return folder_path


