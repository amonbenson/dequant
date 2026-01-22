import os
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .download import download_file, unzip_file
from .preprocess import extract_matrices

logger = logging.getLogger("egmd")


@dataclass
class EGMDConfig:
    midi_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip"
    meta_url: str = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.csv"
    tmp_dir: os.PathLike = ".data/tmp"
    hov_dir: os.PathLike = ".data/hov"


def preprocess(config: EGMDConfig = EGMDConfig()):
    midi_filename = os.path.join(config.tmp_dir, "egmd-midi.zip")
    meta_filename = os.path.join(config.tmp_dir, "egmd-meta.csv")

    # Create tmp directory for downloading files
    os.makedirs(config.tmp_dir, exist_ok=True)

    # Download midi and meta data
    logger.info(f"Downloading to '{config.tmp_dir}' ...")
    download_file(config.midi_url, midi_filename)
    download_file(config.meta_url, meta_filename)

    # Unzip midi dataset
    logger.info("Unzipping midi files...")
    midi_dir = os.path.splitext(midi_filename)[0]
    unzip_file(midi_filename, midi_dir)

    # Read the CSV file
    df = pd.read_csv(meta_filename)

    # filter all non 4-4
    df_filt = df[df["time_signature"] == "4-4"]
    filter_count = len(df) - len(df_filt)
    print(f"Removed {filter_count} files due to mismatching time signature")

    # split in train, test, validaton
    df_splits = {
        "train": df_filt[df_filt["split"] == "train"],
        "test": df_filt[df_filt["split"] == "test"],
        "validation": df_filt[df_filt["split"] == "validation"],
    }

    # run preprocessing for each split
    for split_name, df in df_splits.items():
        # Create split directory
        split_dir = os.path.join(config.hov_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Skip if the target already exists
        data_filename = os.path.join(split_dir, "egmd.npz")
        if os.path.exists(data_filename):
            logger.info(
                f"Skipping preprocessing, because the target file '{data_filename}' already exists."
            )
            continue

        # Convert df entries to file info list
        file_infos = [
            (
                os.path.join(midi_dir, "e-gmd-v1.0.0", row.midi_filename),
                row.bpm,
            )
            for row in df.itertuples(index=False)
        ]

        # Run the parallel preprocesser
        logger.info(f"Extracing split '{split_name}' ...")
        matrices = extract_matrices(file_infos)

        # Store the matrices as .npz
        logger.info(f"Saving '{data_filename}' ...")
        data_array = np.empty(len(matrices), dtype=object)
        for i, item in enumerate(matrices):
            data_array[i] = item
        np.savez_compressed(data_filename, data=data_array)


if __name__ == "__main__":
    # Example preprocessing default options
    preprocess()
