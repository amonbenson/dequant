import threading
import time
from collections import deque
from dataclasses import dataclass, field

# True MIDI constants (defined by the MIDI spec, never change)
MIDI_CLOCKS_PER_BEAT = 24

SYNC_TYPES = frozenset(("clock", "start", "stop", "continue", "songpos"))


@dataclass
class MidiSyncConfig:
    beats_per_bar: int = 4
    ticks_per_beat: int = 4  # 4 = 16th notes, 2 = 8th notes, etc.

    @property
    def clocks_per_tick(self) -> int:
        return MIDI_CLOCKS_PER_BEAT // self.ticks_per_beat

    @property
    def clocks_per_bar(self) -> int:
        return MIDI_CLOCKS_PER_BEAT * self.beats_per_bar


@dataclass
class Position:
    bar: int = 0
    beat: int = 0
    division: int = 0
    clock: int = 0

    @staticmethod
    def from_clock(total_clocks: int, config: MidiSyncConfig):
        total_tick = total_clocks // config.clocks_per_tick
        total_beat = total_tick // config.ticks_per_beat

        return Position(
            bar=total_beat // config.beats_per_bar,
            beat=total_beat % config.beats_per_bar,
            division=total_tick % config.ticks_per_beat,
            clock=total_clocks % config.clocks_per_tick,
        )


@dataclass
class TransportState:
    bpm: float = 120.0
    playing: bool = False
    position: Position = field(default_factory=Position)


class MidiSync:
    """Thread-safe MIDI sync state, updated directly from the MIDI callback."""

    def __init__(self, config: MidiSyncConfig | None = None):
        self._config = config or MidiSyncConfig()
        self._lock = threading.Lock()
        self.clock = 0
        self.playing = False
        self.bpm = 0.0
        self._tick_times: deque[float] = deque(maxlen=MIDI_CLOCKS_PER_BEAT)
        # Division signaling
        self.division_event = threading.Event()
        self.division_times: deque[float] = deque()

    def handle(self, msg) -> None:
        cpt = self._config.clocks_per_tick
        with self._lock:
            match msg.type:
                case "clock":
                    now = time.perf_counter()
                    if not self.playing:
                        self.playing = True
                        self._tick_times.clear()
                    old_div = self.clock // cpt
                    self.clock += 1
                    new_div = self.clock // cpt
                    self._tick_times.append(now)
                    if len(self._tick_times) >= 2:
                        span = self._tick_times[-1] - self._tick_times[0]
                        intervals = len(self._tick_times) - 1
                        self.bpm = 60.0 / ((span / intervals) * MIDI_CLOCKS_PER_BEAT)
                    if new_div != old_div:
                        self.division_times.append(now)
                        self.division_event.set()
                case "start":
                    self.clock = 0
                    self.playing = True
                    self._tick_times.clear()
                    self.bpm = 0.0
                case "stop":
                    self.playing = False
                case "continue":
                    self.playing = True
                case "songpos":
                    self.clock = msg.pos * 6

    def is_playing(self) -> bool:
        with self._lock:
            return self.playing

    def get_bpm(self) -> float:
        return self.bpm

    def get_clock_position(self) -> int:
        with self._lock:
            return self.clock

    def get_tick_position(self) -> int:
        return self.get_clock_position() // self._config.clocks_per_tick

    def get_position(self) -> Position:
        clock_position = self.get_clock_position()
        return Position.from_clock(clock_position, self._config)

    def get_transport_state(self) -> TransportState:
        return TransportState(
            bpm=self.get_bpm(),
            playing=self.is_playing(),
            position=self.get_position(),
        )
