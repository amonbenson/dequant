import asyncio

import flet as ft

from .engine import MidiEngine
from .processor import RealtimeProcessor
from .sync import MidiSync


def main(page: ft.Page):
    page.title = "MIDI App"

    sync = MidiSync()
    processor = RealtimeProcessor()
    engine = MidiEngine(sync, processor)
    engine.start()

    def on_input_select(e: ft.Event[ft.Dropdown]):
        if e.control.value:
            engine.open_input(e.control.value)
            log.value += f"Opened input: {e.control.value}\n"
            page.update()

    def on_output_select(e: ft.Event[ft.Dropdown]):
        if e.control.value:
            engine.open_output(e.control.value)
            log.value += f"Opened output: {e.control.value}\n"
            page.update()

    async def ui_updater():
        last_bar = last_beat = -1
        while engine.running:
            bar, beat, bpm, playing = sync.get_position()
            if bar != last_bar or beat != last_beat:
                last_bar, last_beat = bar, beat
                status = "" if playing else "  [stopped]"
                bpm_str = f"  ({bpm:.0f} BPM)" if bpm > 0 else ""
                transport_label.value = f"Bar {bar}  Beat {beat}{bpm_str}{status}"
                page.update()
            await asyncio.sleep(0.05)

    page.run_task(ui_updater)

    def on_close(e):
        engine.stop()

    page.on_close = on_close

    input_dropdown = ft.Dropdown(
        label="MIDI Input",
        options=[ft.dropdown.Option(name) for name in engine.get_input_ports()],
        on_select=on_input_select,
        width=300,
    )

    output_dropdown = ft.Dropdown(
        label="MIDI Output",
        options=[ft.dropdown.Option(name) for name in engine.get_output_ports()],
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
            ft.Text("Log:"),
            log,
        ])
    )


if __name__ == "__main__":
    ft.app(main)
