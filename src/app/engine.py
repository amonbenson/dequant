from dataclasses import dataclass
from enum import Enum
import threading

import mido
import numpy as np

from ..utils.accurate_timer import AccurateTimer
from ..utils.sliding_window_estimator import SlidingWindowEstimator


MIDI_CLOCKS_PER_BEAT = 24


@dataclass
class MidiEngineConfig:
    update_period: float = 0.01
    grace_period: float = 0.005

    beats_per_bar: int = 4
    ticks_per_beat: int = 4

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
    def from_clock(total_clocks: int, config: MidiEngineConfig):
        total_tick = total_clocks // config.clocks_per_tick
        total_beat = total_tick // config.ticks_per_beat

        return Position(
            bar=total_beat // config.beats_per_bar,
            beat=total_beat % config.beats_per_bar,
            division=total_tick % config.ticks_per_beat,
            clock=total_clocks % config.clocks_per_tick,
        )


class MidiEngine:
    class GracePeriodState(Enum):
        WAIT_FIRST_CLOCK = 0
        GRACE_PERIOD = 1
        WAIT_NEXT_CLOCK = 2

    class NoteState(Enum):
        IDLE = 0
        DELAYING = 1
        PLAYING = 2

    def __init__(self, config: MidiEngineConfig = MidiEngineConfig()):
        self.config = config
        self.midi_in = None
        self.midi_out = None

        self.running = False
        self.playing = False
        self.initial_clock_received = False
        self.hotzone = False
        self.clock_position = 0

        self.last_clock_window = np.zeros(16, dtype=np.float64)
        self.last_clock_window_size = 0

        self.clock_duration_estimator = SlidingWindowEstimator(60 / (120 * MIDI_CLOCKS_PER_BEAT))

        self.thread = None
        self.sync_lock = threading.Lock()
        self.clock_event = threading.Event()

        self.note_velocities = np.zeros(128, dtype=np.uint8)
        self.step_callback = None

    def on_step(self, cb):
        self.step_callback = cb

    def get_bpm(self) -> float:
        if self.clock_duration_estimator.accuracy() == 0:
            return 0.0
        else:
            return 60 / (self.clock_duration_estimator.value * MIDI_CLOCKS_PER_BEAT)

    def get_position(self):
        return Position.from_clock(self.clock_position, self.config)

    @staticmethod
    def get_input_ports() -> list[str]:
        return mido.get_input_names()

    @staticmethod
    def get_output_ports() -> list[str]:
        return mido.get_output_names()

    def open_input(self, port_name: str):
        if self.midi_in:
            self.midi_in.close()
        self.midi_in = mido.open_input(port_name, callback=self._on_midi_message)

    def open_output(self, port_name: str):
        if self.midi_out:
            self.midi_out.close()
        self.midi_out = mido.open_output(port_name)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.playing = False
        self.running = False
        if self.midi_in:
            self.midi_in.close()
            self.midi_in = None
        if self.midi_out:
            self.midi_out.close()
            self.midi_out = None
        if self.thread:
            self.thread.join(timeout=1.0)

    def _send(self, msg) -> None:
        if self.midi_out:
            self.midi_out.send(msg)

    def _on_midi_message(self, msg):
        with self.sync_lock:
            # Handle incoming messages
            match msg.type:
                case "clock":
                    if self.initial_clock_received:
                        self.clock_position += 1
                    self.initial_clock_received = True
                    self.playing = True
                    self.clock_duration_estimator.update()
                case "start":
                    self.clock_position = 0
                    self.initial_clock_received = False
                    self.playing = True
                    self.clock_duration_estimator.update(skip_estimate=True)
                case "stop":
                    self.initial_clock_received = False
                    self.playing = False
                case "continue":
                    self.initial_clock_received = False
                    self.playing = True
                    self.clock_duration_estimator.update(skip_estimate=True)
                case "songpos":
                    self.clock_position = msg.pos * 6
                case "note_on":
                    if self.playing:
                        # Store note velocity
                        self.note_velocities[msg.note] = msg.velocity
                    else:
                        # Pass through while not playing
                        self._send(msg)
                case "note_off":
                    if self.playing:
                        # Clear note velocity
                        self.note_velocities[msg.note] = 0
                    else:
                        # Pass through while not playing
                        self._send(msg)
                case _:
                    # Pass through other messages
                    self._send(msg)

    def _loop(self):
        timer = AccurateTimer(self.config.update_period)

        grace_period_state = self.GracePeriodState.WAIT_FIRST_CLOCK
        last_division_time = timer.time
        grace_period_end_time = timer.time

        note_states = np.ones(128, dtype=np.uint8) * self.NoteState.IDLE.value
        note_velocities = np.empty(128, dtype=np.uint8)
        note_offsets = np.empty(128, dtype=np.float32)
        note_duration = np.empty(128, dtype=np.float32)

        while self.running:
            # Wait until we are playing again
            while not self.playing:
                timer.sleep()

            # Get the current subclock position (clock within a division)
            with self.sync_lock:
                subclocks = self.clock_position % self.config.clocks_per_tick

            match grace_period_state:
                case self.GracePeriodState.WAIT_FIRST_CLOCK:
                    # Start the grace period as soon as we cross a division
                    if subclocks == 0:
                        last_division_time = timer.time
                        grace_period_state = self.GracePeriodState.GRACE_PERIOD

                case self.GracePeriodState.GRACE_PERIOD:
                    # If the grace period elapsed, process the notes and wait for the next clock pulse
                    if timer.time >= last_division_time + self.config.grace_period:
                        grace_period_state = self.GracePeriodState.WAIT_NEXT_CLOCK

                        # Store timestamp and velocities
                        grace_period_end_time = timer.time
                        with self.sync_lock:
                            input_velocities = self.note_velocities

                        # Invoke the processor
                        seconds_per_tick = 60 / (self.get_bpm() * self.config.ticks_per_beat)
                        if self.step_callback:
                            tick_offsets, note_velocities = self.step_callback(input_velocities, self.clock_position // self.config.clocks_per_tick)
                            note_offsets = tick_offsets * seconds_per_tick  # Convert delay from ticks to seconds
                        else:
                            note_velocities[:] = input_velocities
                            note_offsets[:] = 0.0
                        note_duration[:] = 0.5 * seconds_per_tick  # Play note for 0.5 ticks

                        # Init note states at each point where we have a velocity > 0
                        note_states = np.where(note_velocities > 0, self.NoteState.DELAYING.value, self.NoteState.IDLE.value)

                case self.GracePeriodState.WAIT_NEXT_CLOCK:
                    # If we've reached the next cycle, start over
                    if subclocks > 0:
                        grace_period_state = self.GracePeriodState.WAIT_FIRST_CLOCK

            # Update note states and fire events on state transitions
            for note in np.flatnonzero(note_states):
                match self.NoteState(note_states[note]):
                    case self.NoteState.IDLE:
                        pass

                    case self.NoteState.DELAYING:
                        if timer.time >= grace_period_end_time + note_offsets[note]:
                            note_states[note] = self.NoteState.PLAYING.value
                            self._send(mido.Message(type="note_on", note=note, velocity=note_velocities[note]))

                    case self.NoteState.PLAYING:
                        if timer.time >= grace_period_end_time + note_offsets[note] + note_duration[note]:
                            note_states[note] = self.NoteState.IDLE.value
                            self._send(mido.Message(type="note_off", note=note, velocity=127))

            timer.sleep()
