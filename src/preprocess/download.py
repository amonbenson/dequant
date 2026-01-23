from pathlib import Path
import logging
from zipfile import ZipFile
from urllib.request import urlretrieve

logger = logging.getLogger("download")


def download_file(url: str, filename: Path):
    if filename.exists():
        logger.info(
            f"Skipping download, because the target directory '{filename}' already exists."
        )
        return

    urlretrieve(url, filename)


def unzip_file(zip_filename: Path, target_dir: Path):
    if target_dir.exists():
        logger.info(
            f"Skipping unzip, because the target directory '{target_dir}' already exists."
        )
        return

    with ZipFile(zip_filename) as f:
        f.extractall(target_dir)
