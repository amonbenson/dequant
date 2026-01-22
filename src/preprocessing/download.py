import os
import logging
from zipfile import ZipFile
from urllib.request import urlretrieve

logger = logging.getLogger("download")


def download_file(url: str, filename: os.PathLike):
    if os.path.exists(filename):
        logger.info(
            f"Skipping download, because the target directory '{filename}' already exists."
        )
        return

    urlretrieve(url, filename)


def unzip_file(zip_filename: os.PathLike, target_dir: os.PathLike):
    if os.path.exists(target_dir):
        logger.info(
            f"Skipping unzip, because the target directory '{target_dir}' already exists."
        )
        return

    with ZipFile(zip_filename) as f:
        f.extractall(target_dir)
