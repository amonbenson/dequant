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

BATCH_SIZE = 2000  # number of files to process before writing a batch .npz


def _has_drum_track(path: Path) -> bool:
    """Fast check for drum content by scanning for MIDI channel 9 note_on bytes (0x99)."""
    try:
        return b"\x99" in path.read_bytes()
    except Exception:
        return False


def _find_drum_files(midi_files: list[Path], n_workers: int | None) -> list[Path]:
    """Scan MIDI files in parallel and return only those containing drum tracks."""
    drum_files = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_has_drum_track, path): path for path in midi_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning for drum tracks"):
            if future.result():
                drum_files.append(futures[future])
    return sorted(drum_files)


def _process_batch(converter: HOVConverter, files: list[Path], out_path: Path, n_workers: int | None) -> int:
    """Convert a batch of MIDI files and save to a single .npz file."""
    file_infos: FileInfos = [(path, None) for path in files]
    results = converter.midi_to_hov_batch(file_infos, n_workers=n_workers)

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

    np.savez_compressed(
        out_path,
        data=np.array(hovs, dtype=object),
        pos_en=np.array(pos_en, dtype=object),
    )
    return len(hovs)


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

    # Convert 0 -> None so ProcessPoolExecutor picks the number of workers automatically
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

        batches = [files[i : i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
        logger.info(f"Processing split '{split_name}' ({len(files)} files, {len(batches)} batches)...")

        total_tracks = 0
        for batch_idx, batch in enumerate(batches):
            out_path = split_dir / f"lmd_{batch_idx:04d}.npz"
            if out_path.exists():
                logger.info(f"Skipping batch {batch_idx}, '{out_path}' already exists.")
                continue

            logger.info(f"Batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)...")
            n_saved = _process_batch(converter, batch, out_path, n_workers)
            total_tracks += n_saved
            logger.info(f"Saved {n_saved} tracks to '{out_path}'")

        logger.info(f"Split '{split_name}' complete: {total_tracks} tracks saved across {len(batches)} files")
