import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...config import CONFIG
from ..converters.hov_converter import FileInfos, HOVConverter, HOVConverterConfig
from .download import download_file, untar_file

logger = logging.getLogger("lmd")


def _has_drum_track(path: Path) -> bool:
    """Fast check for drum content by scanning for MIDI channel 9 note_on bytes (0x99)."""
    try:
        return b"\x99" in path.read_bytes()
    except Exception:
        return False


def _find_drum_files(midi_files: list[Path], n_workers: int) -> list[Path]:
    """Scan MIDI files in parallel and return only those containing drum tracks."""
    drum_files = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_has_drum_track, path): path for path in midi_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning for drum tracks"):
            if future.result():
                drum_files.append(futures[future])
    return sorted(drum_files)


def preprocess_lmd():
    converter = HOVConverter(
        HOVConverterConfig(
            steps_per_beat=CONFIG.model.drums.steps_per_beat,
            categories=CONFIG.model.drums.categories,
            max_seq_len=CONFIG.model.max_seq_len,
        )
    )

    tar_filename = CONFIG.data.cache_dir / "lmd_full.tar.gz"
    midi_dir = CONFIG.data.cache_dir / "lmd_full"

    CONFIG.data.cache_dir.mkdir(parents=True, exist_ok=True)

    # Download
    logger.info(f"Downloading LMD to '{CONFIG.data.cache_dir}' ...")
    download_file(CONFIG.data.lmd.url, tar_filename)

    # Extract
    untar_file(tar_filename, midi_dir)

    # Find all MIDI files with drum tracks
    logger.info("Scanning for MIDI files...")
    all_midi_files = sorted(midi_dir.rglob("*.mid")) + sorted(midi_dir.rglob("*.midi"))
    logger.info(f"Found {len(all_midi_files)} MIDI files total, filtering for drum tracks...")

    n_workers = CONFIG.data.num_workers if CONFIG.data.num_workers > 0 else None
    drum_files = _find_drum_files(all_midi_files, n_workers)
    logger.info(f"{len(drum_files)} files contain drum tracks")

    # Deterministic 80/10/10 split
    rng = random.Random(42)
    rng.shuffle(drum_files)

    n = len(drum_files)
    n_train = int(n * CONFIG.data.lmd.train_split)
    n_val = int(n * CONFIG.data.lmd.val_split)

    splits: dict[str, list[Path]] = {
        "train": drum_files[:n_train],
        "validation": drum_files[n_train : n_train + n_val],
        "test": drum_files[n_train + n_val :],
    }

    for split_name, files in splits.items():
        split_dir = CONFIG.data.dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        data_filename = split_dir / "lmd.npz"
        if data_filename.exists():
            logger.info(f"Skipping '{data_filename}', already exists.")
            continue

        # BPM is extracted from each MIDI file by the converter (tempo_bpm=None)
        file_infos: FileInfos = [(path, None) for path in files]

        logger.info(f"Processing split '{split_name}' ({len(files)} files)...")
        results = converter.midi_to_hov_batch(file_infos, n_workers=CONFIG.data.num_workers)

        hovs = []
        pos_en = []
        for item in results:
            if item is None:
                continue
            hov, pos = item
            if hov is None or pos is None:
                continue
            hovs.append(hov)
            pos_en.append(pos)

        logger.info(f"Saving '{data_filename}' ({len(hovs)} tracks)...")
        np.savez_compressed(
            data_filename,
            data=np.array(hovs, dtype=object),
            pos_en=np.array(pos_en, dtype=object),
        )
