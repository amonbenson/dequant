import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import mido
import numpy as np

from .processor import RealtimeProcessor
from .sync import SYNC_TYPES, MidiSync


@dataclass
class MidiEngineConfig:
    grace_period: float = 0.005  # seconds after division boundary
    poll_interval: float = 0.002  # seconds between division polls


class MidiEngine:
    """Routes MIDI messages and runs the division-quantized processing loop.

    Sync messages are handled directly in the MIDI callback (zero latency).
    Note messages are accumulated per division and batch-processed.
    """

    def __init__(
        self,
        sync: MidiSync,
        processor: RealtimeProcessor,
        config: MidiEngineConfig | None = None,
    ):
        self._config = config or MidiEngineConfig()
        self._sync = sync
        self._processor = processor
        self._inbox: deque = deque()
        self._midi_in = None
        self._midi_out = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._velocities = np.zeros(128, dtype=np.uint8)

    @property
    def running(self) -> bool:
        return self._running

    @staticmethod
    def get_input_ports() -> list[str]:
        return mido.get_input_names()

    @staticmethod
    def get_output_ports() -> list[str]:
        return mido.get_output_names()

    def open_input(self, port_name: str):
        if self._midi_in:
            self._midi_in.close()
        self._midi_in = mido.open_input(port_name, callback=self._on_midi_message)

    def open_output(self, port_name: str):
        if self._midi_out:
            self._midi_out.close()
        self._midi_out = mido.open_output(port_name)

    def _on_midi_message(self, msg):
        """MIDI input callback. Sync is handled inline; rest is queued."""
        if msg.type in SYNC_TYPES:
            self._sync.handle(msg)
        else:
            self._inbox.append(msg)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._midi_in:
            self._midi_in.close()
            self._midi_in = None
        if self._midi_out:
            self._midi_out.close()
            self._midi_out = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _send(self, msg) -> None:
        if self._midi_out:
            self._midi_out.send(msg)

    def _loop(self) -> None:
        sync = self._sync
        cfg = self._config

        note_delays = [0] * 128
        pending_on = [0] * 128
        pending_off = [False] * 128

        while self._running:
            # Wait for division with responsive polling
            got_division = False
            while self._running:
                if sync.division_event.is_set():
                    sync.division_event.clear()
                    got_division = True
                    break
                # Pass through non-synced messages while waiting
                if self._inbox:
                    while self._inbox:
                        self._send(self._inbox.popleft())
                else:
                    time.sleep(cfg.poll_interval)

            if not self._running or not got_division:
                continue

            # Drain division timestamps
            while sync.division_times:
                sync.division_times.popleft()

            # Send pending delay=1 notes from previous division
            for n in range(128):
                if pending_on[n] > 0:
                    self._send(mido.Message("note_on", note=n, velocity=pending_on[n]))
                    pending_on[n] = 0
                if pending_off[n]:
                    self._send(mido.Message("note_off", note=n, velocity=0))
                    pending_off[n] = False

            # Spin-wait grace period
            grace_end = time.perf_counter() + cfg.grace_period
            while time.perf_counter() < grace_end:
                pass

            # Drain inbox into velocity vector
            self._velocities[:] = 0
            while self._inbox:
                msg = self._inbox.popleft()
                if msg.type == "note_on" and msg.velocity > 0:
                    self._velocities[msg.note] = msg.velocity
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if note_delays[msg.note] == 0:
                        self._send(msg)
                    else:
                        pending_off[msg.note] = True
                else:
                    self._send(msg)

            # Process the division
            bar, beat, _, _ = sync.get_position()
            new_velocities, delays = self._processor.process(self._velocities, bar, beat)

            # Send or queue results
            for n in range(128):
                if new_velocities[n] > 0:
                    note_delays[n] = delays[n]
                    if delays[n] == 0:
                        self._send(mido.Message("note_on", note=n, velocity=new_velocities[n]))
                    else:
                        pending_on[n] = new_velocities[n]
