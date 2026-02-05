from dataclasses import dataclass
import numpy as np


@dataclass
class DrumCategory:
    label: str
    pitches: tuple[int, ...]

    @staticmethod
    def generate_forward_lookup(categories: list["DrumCategory"]) -> np.ndarray:
        # Initialize the category lookup table
        # Maps pitch -> category id or -1 if the note has no category
        lookup = -np.ones(128, dtype=np.int8)
        for i, cat in enumerate(categories):
            for pitch in cat.pitches:
                if pitch < 0 or pitch > 127:
                    raise ValueError(f"Category {cat.label}: pitch {pitch} is out of range.")
                if lookup[pitch] != -1:
                    raise ValueError(f"Category: {cat.label}: pitch {pitch} was already mapped to another category. Category pitches must be unique!")
                lookup[pitch] = i

        return lookup

    @staticmethod
    def generate_reverse_lookup(categories: list["DrumCategory"]) -> np.ndarray:
        # Initialize the reverse-category lookup. As each category might have multiple notes
        # associated with it, only choose the first one.
        # Maps category id -> pitch
        return np.array([cat.pitches[0] for cat in categories])


# See https://musescore.org/sites/musescore.org/files/General%20MIDI%20Standard%20Percussion%20Set%20Key%20Map.pdf
# Note, the first note value is used for reverse-lookup, so we use the more common notes (36 and 38) for Kick and Snare
DEFAULT_DRUM_CATEGORIES = [
    DrumCategory("Kick", (36, 35)),
    DrumCategory("Snare", (38, 37, 39, 40)),
    DrumCategory("Floor Tom", (41, 43)),
    DrumCategory("Low Tom", (45, 47)),
    DrumCategory("High Tom", (48, 50)),
    DrumCategory("Closed Hi-Hat", (42, 44)),
    DrumCategory("Open Hi-Hat", (46,)),
    DrumCategory("Crash", (49, 52, 55, 57)),
    DrumCategory("Ride", (51, 53, 59)),
]
