from dataclasses import dataclass
import numpy as np


@dataclass
class RealtimeProcessorConfig:
    pass


class RealtimeProcessor:
    """Processes a division's worth of MIDI note data.

    Takes a 128-element velocity vector and the current bar/beat position,
    returns modified velocities and per-note delays.
    """

    def __init__(self, config: RealtimeProcessorConfig | None = None):
        self._config = config or RealtimeProcessorConfig()

    def process(self, velocities: np.ndarray, bar: int, beat: int) -> tuple[np.ndarray, np.ndarray]:
        """Process a single division.

        Args:
            velocities: 128-element list, 0 = not played, >0 = velocity.
            bar: Current bar number (1-based).
            beat: Current beat number within bar (1-based).

        Returns:
            (new_velocities, delays) — 128 elements each.
            Delays are 0 (play now) or 1 (play next division).
        """
        new_velocities = (velocities > 0).astype(np.uint8) * 127  # [0] * 128
        delays = np.zeros(128, dtype=np.uint8)
        return new_velocities, delays
