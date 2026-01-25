from dataclasses import dataclass


@dataclass
class DrumCategory:
    label: str
    pitches: tuple[int]


# See https://musescore.org/sites/musescore.org/files/General%20MIDI%20Standard%20Percussion%20Set%20Key%20Map.pdf
DEFAULT_DRUM_CATEGORIES = [
    DrumCategory("Kick", (35, 36)),
    DrumCategory("Snare", (37, 38, 40)),
    DrumCategory("Floor Tom", (41, 43)),
    DrumCategory("Low Tom", (45, 47)),
    DrumCategory("High Tom", (48, 50)),
    DrumCategory("Closed Hi-Hat", (42, 44)),
    DrumCategory("Open Hi-Hat", (46,)),
    DrumCategory("Crash", (49, 52, 55, 57)),
    DrumCategory("Ride", (51, 53, 59)),
]
