import logging
import pandas as pd
import numpy as np
from .download import download_file, unzip_file
from ..data.converters.hov_converter import HOVConverter, HOVConverterConfig
from ..config import CONFIG

logger = logging.getLogger("egmd")


def preprocess_egmd():
    converter = HOVConverter(
        HOVConverterConfig(
            steps_per_beat=CONFIG.model.drums.steps_per_beat,
            categories=CONFIG.model.drums.categories,
        )
    )

    midi_filename = CONFIG.data.cache_dir / "egmd-midi.zip"
    meta_filename = CONFIG.data.cache_dir / "egmd-meta.csv"

    # Create tmp directory for downloading files
    CONFIG.data.cache_dir.mkdir(parents=True, exist_ok=True)

    # Download midi and meta data
    logger.info(f"Downloading to '{CONFIG.data.cache_dir}' ...")
    download_file(CONFIG.data.egmd.midi_url, midi_filename)
    download_file(CONFIG.data.egmd.metadata_url, meta_filename)

    # Unzip midi dataset
    logger.info("Unzipping midi files...")
    midi_dir = midi_filename.with_suffix("")
    unzip_file(midi_filename, midi_dir)

    # Read the CSV file
    df = pd.read_csv(meta_filename)

    # filter all non 4-4
    df_filt = df[df["time_signature"] == "4-4"]
    filter_count = len(df) - len(df_filt)
    logger.debug(f"Removed {filter_count} files due to mismatching time signature")

    # split in train, test, validaton
    df_splits = {
        "train": df_filt[df_filt["split"] == "train"],
        "test": df_filt[df_filt["split"] == "test"],
        "validation": df_filt[df_filt["split"] == "validation"],
    }

    # run preprocessing for each split
    for split_name, df in df_splits.items():
        # Create split directory
        split_dir = CONFIG.data.dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Skip if the target already exists
        data_filename = split_dir / "egmd.npz"
        if data_filename.exists():
            logger.info(f"Skipping preprocessing, because the target file '{data_filename}' already exists.")
            continue

        # Convert df entries to file info list
        file_infos = [
            (
                midi_dir / "e-gmd-v1.0.0" / row.midi_filename,
                row.bpm,
            )
            for row in df.itertuples(index=False)
        ]

        # Run the parallel preprocesser
        logger.info(f"Extracing split '{split_name}' ...")
        matrices = converter.midi_to_hov_batch(
            file_infos,
            n_workers=CONFIG.data.num_workers,
        )

        # Store the matrices as .npz
        logger.info(f"Saving '{data_filename}' ...")
        np.savez_compressed(data_filename, data=matrices)
