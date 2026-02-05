import threading
from dataclasses import dataclass
from enum import Enum

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
        self._midi_in = None
        self._midi_out = None

        self._running = False
        self._playing = False

        self._clock_position = 0
        self._clock_duration_estimator = SlidingWindowEstimator(60 / (120 * MIDI_CLOCKS_PER_BEAT), window_size=100)
        self._initial_clock_received = False

        self._thread = None
        self._lock = threading.Lock()

        self._note_velocities = np.zeros(128, dtype=np.uint8)
        self._step_callback = None

    def on_step(self, cb):
        self._step_callback = cb

    @property
    def running(self) -> bool:
        return self._running

    def get_bpm(self) -> float:
        return 60 / (self._clock_duration_estimator.value * MIDI_CLOCKS_PER_BEAT)

    def get_position(self):
        return Position.from_clock(self._clock_position, self.config)

    @staticmethod
    def get_input_ports() -> list[str]:
        return mido.get_input_names()  # type: ignore[attr-defined]

    @staticmethod
    def get_output_ports() -> list[str]:
        return mido.get_output_names()  # type: ignore[attr-defined]

    def open_input(self, port_name: str):
        if self._midi_in:
            self._midi_in.close()
        self._midi_in = mido.open_input(port_name, callback=self._on_midi_message)  # type: ignore[attr-defined]

    def open_output(self, port_name: str):
        if self._midi_out:
            self._midi_out.close()
        self._midi_out = mido.open_output(port_name)  # type: ignore[attr-defined]

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._playing = False
        self._running = False
        if self._midi_in:
            self._midi_in.close()
            self._midi_in = None
        if self._midi_out:
            self._midi_out.close()
            self._midi_out = None
        if self._thread:
            self._thread.join(timeout=1.0)

    def _send(self, msg) -> None:
        if self._midi_out:
            self._midi_out.send(msg)

    def _on_midi_message(self, msg):
        with self._lock:
            # Handle incoming messages
            match msg.type:
                case "clock":
                    if self._initial_clock_received:
                        self._clock_position += 1
                        self._clock_duration_estimator.update()
                    else:
                        self._clock_duration_estimator.update(skip_estimate=True)
                    self._initial_clock_received = True
                    self._playing = True
                case "start":
                    self._clock_position = 0
                    self._initial_clock_received = False
                    self._playing = True
                    self._clock_duration_estimator.update(skip_estimate=True)
                case "stop":
                    self._initial_clock_received = False
                    self._playing = False
                    self._note_velocities[:] = 0  # Clear all currently playing notes
                case "continue":
                    self._initial_clock_received = False
                    self._playing = True
                    self._clock_duration_estimator.update(skip_estimate=True)
                case "songpos":
                    self._clock_position = msg.pos * 6
                case "note_on":
                    if self._playing:
                        # Store note velocity
                        self._note_velocities[msg.note] = msg.velocity
                    else:
                        # Pass through while not playing
                        self._send(msg)
                case "note_off":
                    if self._playing:
                        # Clear note velocity
                        self._note_velocities[msg.note] = 0
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

        while self._running:
            # Wait until we are playing again
            while not self._playing:
                timer.sleep()

            # Get the current subclock position (clock within a division)
            with self._lock:
                subclocks = self._clock_position % self.config.clocks_per_tick

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

                        # Cutoff all notes that haven't been fired yet (because they were delayed really long)
                        for note in np.flatnonzero(note_states):
                            self._send(mido.Message(type="note_off", note=note, velocity=127))
                        note_states[:] = self.NoteState.IDLE.value

                        # Store timestamp and velocities
                        grace_period_end_time = timer.time
                        with self._lock:
                            input_velocities = self._note_velocities

                        # Invoke the processor
                        seconds_per_tick = 60 / (self.get_bpm() * self.config.ticks_per_beat)
                        if self._step_callback:
                            tick_offsets, note_velocities = self._step_callback(input_velocities, self._clock_position // self.config.clocks_per_tick)
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
