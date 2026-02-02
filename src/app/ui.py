import asyncio
import threading
import time
from collections import deque

import flet as ft
import mido

CLOCKS_PER_BEAT = 24
BEATS_PER_BAR = 4
CLOCKS_PER_BAR = CLOCKS_PER_BEAT * BEATS_PER_BAR

TICKS_PER_BEAT = 4  # 4 = 16th notes, 2 = 8th notes, etc.
CLOCKS_PER_DIVISION = CLOCKS_PER_BEAT // TICKS_PER_BEAT  # 6 clocks per 16th

GRACE_PERIOD = 0.005  # 5ms grace period after division boundary


def process_division(velocities, bar, beat):
    """Dummy processor: takes 128-element velocity vector and position.

    Returns (new_velocities, delays) where delays are 0 (play now)
    or 1 (play next division).
    """
    new_velocities = [0] * 128
    delays = [0] * 128
    for i in range(128):
        if velocities[i] > 0:
            new_velocities[i] = min(127, velocities[i] + 10)
    return new_velocities, delays


class MidiSync:
    """Thread-safe MIDI sync state, updated directly from the MIDI callback."""

    def __init__(self, ticks_per_beat=TICKS_PER_BEAT):
        self._lock = threading.Lock()
        self.tick_count = 0
        self.playing = False
        self.bpm = 0.0
        self._tick_times = deque(maxlen=CLOCKS_PER_BEAT)
        self._clocks_per_division = CLOCKS_PER_BEAT // ticks_per_beat
        # Division signaling
        self.division_event = threading.Event()
        self.division_times = deque()

    def handle(self, msg):
        """Called directly from the MIDI input callback. Must be fast."""
        if msg.type == "clock":
            now = time.perf_counter()
            with self._lock:
                if not self.playing:
                    self.playing = True
                    self._tick_times.clear()
                old_div = self.tick_count // self._clocks_per_division
                self.tick_count += 1
                new_div = self.tick_count // self._clocks_per_division
                self._tick_times.append(now)
                if len(self._tick_times) >= 2:
                    span = self._tick_times[-1] - self._tick_times[0]
                    intervals = len(self._tick_times) - 1
                    self.bpm = 60.0 / ((span / intervals) * CLOCKS_PER_BEAT)
            if new_div != old_div:
                self.division_times.append(now)
                self.division_event.set()
        elif msg.type == "start":
            with self._lock:
                self.tick_count = 0
                self.playing = True
                self._tick_times.clear()
                self.bpm = 0.0
        elif msg.type == "stop":
            with self._lock:
                self.playing = False
        elif msg.type == "continue":
            with self._lock:
                self.playing = True
        elif msg.type == "songpos":
            with self._lock:
                self.tick_count = msg.pos * 6

    def get_position(self):
        """Thread-safe read of current bar/beat/bpm."""
        with self._lock:
            ticks = self.tick_count
            bpm = self.bpm
            playing = self.playing
        bar = ticks // CLOCKS_PER_BAR + 1
        beat = (ticks % CLOCKS_PER_BAR) // CLOCKS_PER_BEAT + 1
        return bar, beat, bpm, playing


_SYNC_TYPES = frozenset(("clock", "start", "stop", "continue", "songpos"))


def main(page: ft.Page):
    page.title = "MIDI App"

    input_ports = mido.get_input_names()
    output_ports = mido.get_output_names()

    midi_in = None
    midi_out = None

    sync = MidiSync()

    # Note inbox (only non-sync messages)
    inbox = deque()
    running = True

    def on_input_select(e: ft.Event[ft.Dropdown]):
        nonlocal midi_in
        if midi_in:
            midi_in.close()
        if e.control.value:
            midi_in = mido.open_input(e.control.value, callback=on_midi_message)
            log.value += f"Opened input: {e.control.value}\n"
            page.update()

    def on_output_select(e: ft.Event[ft.Dropdown]):
        nonlocal midi_out
        if midi_out:
            midi_out.close()
        if e.control.value:
            midi_out = mido.open_output(e.control.value)
            log.value += f"Opened output: {e.control.value}\n"
            page.update()

    def on_midi_message(msg):
        if msg.type in _SYNC_TYPES:
            sync.handle(msg)
        else:
            inbox.append(msg)

    def processing_loop():
        # Per-note delay tracking (for matching note_off to note_on delay)
        note_delays = [0] * 128

        # Pending delay=1 data from previous division
        pending_on = [0] * 128   # velocities to send as note_on
        pending_off = [False] * 128  # True = send note_off

        while running:
            # Wait for division with responsive short-timeout polling
            # (Event.wait wake-up on macOS can lag 5-15ms; polling at
            # 2ms keeps us responsive without burning CPU)
            got_division = False
            while running:
                if sync.division_event.is_set():
                    sync.division_event.clear()
                    got_division = True
                    break
                # Pass through non-synced messages while waiting
                if inbox:
                    while inbox:
                        msg = inbox.popleft()
                        if midi_out:
                            midi_out.send(msg)
                else:
                    time.sleep(0.002)

            if not running or not got_division:
                continue

            # Drain division timestamps (we only care that one happened)
            while sync.division_times:
                sync.division_times.popleft()

            # 1. Send pending delay=1 notes from previous division
            for note_num in range(128):
                if pending_on[note_num] > 0:
                    if midi_out:
                        midi_out.send(mido.Message(
                            "note_on", note=note_num,
                            velocity=pending_on[note_num],
                        ))
                    pending_on[note_num] = 0
                if pending_off[note_num]:
                    if midi_out:
                        midi_out.send(mido.Message(
                            "note_off", note=note_num, velocity=0,
                        ))
                    pending_off[note_num] = False

            # 2. Spin-wait grace period (precise, no OS sleep overshoot)
            grace_end = time.perf_counter() + GRACE_PERIOD
            while time.perf_counter() < grace_end:
                pass

            # 3. Drain inbox into velocity vector
            velocities = [0] * 128
            while inbox:
                msg = inbox.popleft()
                if msg.type == "note_on" and msg.velocity > 0:
                    velocities[msg.note] = msg.velocity
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    # Match delay of the corresponding note_on
                    if note_delays[msg.note] == 0:
                        if midi_out:
                            midi_out.send(msg)
                    else:
                        pending_off[msg.note] = True
                else:
                    # CC, pitch bend, etc. — forward immediately
                    if midi_out:
                        midi_out.send(msg)

            # 4. Process the division
            bar, beat, _, _ = sync.get_position()
            new_velocities, delays = process_division(velocities, bar, beat)

            # 5. Send or queue results
            for note_num in range(128):
                if new_velocities[note_num] > 0:
                    note_delays[note_num] = delays[note_num]
                    if delays[note_num] == 0:
                        if midi_out:
                            midi_out.send(mido.Message(
                                "note_on", note=note_num,
                                velocity=new_velocities[note_num],
                            ))
                    else:
                        pending_on[note_num] = new_velocities[note_num]

    proc_thread = threading.Thread(target=processing_loop, daemon=True)
    proc_thread.start()

    async def ui_updater():
        last_bar = last_beat = -1
        while running:
            bar, beat, bpm, playing = sync.get_position()
            if bar != last_bar or beat != last_beat:
                last_bar, last_beat = bar, beat
                status = "" if playing else "  [stopped]"
                bpm_str = f"  ({bpm:.0f} BPM)" if bpm > 0 else ""
                transport_label.value = f"Bar {bar}  Beat {beat}{bpm_str}{status}"
                page.update()
            await asyncio.sleep(0.05)

    page.run_task(ui_updater)

    def send_note(e):
        if midi_out:
            midi_out.send(mido.Message("note_on", note=60, velocity=64))
            midi_out.send(mido.Message("note_off", note=60, velocity=0))
            log.value += "Sent middle C\n"
            page.update()

    def on_close(e):
        nonlocal running
        running = False

    page.on_close = on_close

    input_dropdown = ft.Dropdown(
        label="MIDI Input",
        options=[ft.dropdown.Option(name) for name in input_ports],
        on_select=on_input_select,
        width=300,
    )

    output_dropdown = ft.Dropdown(
        label="MIDI Output",
        options=[ft.dropdown.Option(name) for name in output_ports],
        on_select=on_output_select,
        width=300,
    )

    transport_label = ft.Text("Bar -  Beat -", size=16, weight=ft.FontWeight.BOLD)

    log = ft.TextField(
        multiline=True,
        read_only=True,
        min_lines=10,
        max_lines=10,
        width=400,
    )

    page.add(
        ft.Column([
            ft.Row([input_dropdown, output_dropdown]),
            transport_label,
            ft.ElevatedButton("Send Note", on_click=send_note),
            ft.Text("Log:"),
            log,
        ])
    )


if __name__ == "__main__":
    ft.app(main)
