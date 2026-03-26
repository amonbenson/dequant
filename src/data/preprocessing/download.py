import logging
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm

logger = logging.getLogger("download")


def download_file(url: str, filename: Path):
    if filename.exists():
        logger.info(f"Skipping download, because the target directory '{filename}' already exists.")
        return

    # Show a nice progress bar
    progress: Optional[tqdm] = None

    # Hook to update the progress bar
    def reporthook(_, block_size, total_size):
        nonlocal progress
        if progress is None:
            progress = tqdm(total=total_size, unit="B", unit_scale=True)
        progress.update(block_size)

    # Run the url retrieve function with the reporthook attached
    urlretrieve(url, filename, reporthook=reporthook)

    # Close the progress bar
    if progress:
        progress.close()


def unzip_file(zip_filename: Path, target_dir: Path):
    if target_dir.exists():
        logger.info(f"Skipping unzip, because the target directory '{target_dir}' already exists.")
        return

    with ZipFile(zip_filename) as f:
        f.extractall(target_dir)


def untar_file(tar_filename: Path, target_dir: Path):
    import tarfile

    if target_dir.exists():
        logger.info(f"Skipping untar, because the target directory '{target_dir}' already exists.")
        return

    logger.info(f"Extracting '{tar_filename}' ...")
    with tarfile.open(tar_filename, "r:gz") as tar:
        tar.extractall(target_dir)
