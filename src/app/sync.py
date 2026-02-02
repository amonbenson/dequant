import threading
import time
from collections import deque
from dataclasses import dataclass

# True MIDI constants (defined by the MIDI spec, never change)
MIDI_CLOCKS_PER_BEAT = 24

SYNC_TYPES = frozenset(("clock", "start", "stop", "continue", "songpos"))


@dataclass
class MidiSyncConfig:
    beats_per_bar: int = 4
    ticks_per_beat: int = 4  # 4 = 16th notes, 2 = 8th notes, etc.

    @property
    def clocks_per_division(self) -> int:
        return MIDI_CLOCKS_PER_BEAT // self.ticks_per_beat

    @property
    def clocks_per_bar(self) -> int:
        return MIDI_CLOCKS_PER_BEAT * self.beats_per_bar


class MidiSync:
    """Thread-safe MIDI sync state, updated directly from the MIDI callback."""

    def __init__(self, config: MidiSyncConfig | None = None):
        self._config = config or MidiSyncConfig()
        self._lock = threading.Lock()
        self.tick_count = 0
        self.playing = False
        self.bpm = 0.0
        self._tick_times: deque[float] = deque(maxlen=MIDI_CLOCKS_PER_BEAT)
        # Division signaling
        self.division_event = threading.Event()
        self.division_times: deque[float] = deque()

    def handle(self, msg) -> None:
        """Called directly from the MIDI input callback. Must be fast."""
        cpd = self._config.clocks_per_division
        with self._lock:
            match msg.type:
                case "clock":
                    now = time.perf_counter()
                    if not self.playing:
                        self.playing = True
                        self._tick_times.clear()
                    old_div = self.tick_count // cpd
                    self.tick_count += 1
                    new_div = self.tick_count // cpd
                    self._tick_times.append(now)
                    if len(self._tick_times) >= 2:
                        span = self._tick_times[-1] - self._tick_times[0]
                        intervals = len(self._tick_times) - 1
                        self.bpm = 60.0 / ((span / intervals) * MIDI_CLOCKS_PER_BEAT)
                    if new_div != old_div:
                        self.division_times.append(now)
                        self.division_event.set()
                case "start":
                    self.tick_count = 0
                    self.playing = True
                    self._tick_times.clear()
                    self.bpm = 0.0
                case "stop":
                    self.playing = False
                case "continue":
                    self.playing = True
                case "songpos":
                    self.tick_count = msg.pos * 6

    def get_position(self) -> tuple[int, int, float, bool]:
        """Thread-safe read of current (bar, beat, bpm, playing)."""
        cpb = self._config.clocks_per_bar
        with self._lock:
            ticks = self.tick_count
            bpm = self.bpm
            playing = self.playing
        bar = ticks // cpb + 1
        beat = (ticks % cpb) // MIDI_CLOCKS_PER_BEAT + 1
        return bar, beat, bpm, playing
