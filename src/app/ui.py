import asyncio
from pathlib import Path

import flet as ft
import numpy as np
import torch

from .engine import MidiEngine, Position
from ..inference.predictor import Predictor, PredictorConfig
from ..data.drum_category import DrumCategory
from ..config import CONFIG


def main(page: ft.Page):
    category_lookup = DrumCategory.generate_forward_lookup(CONFIG.model.drums.categories)
    category_reverse_lookup = DrumCategory.generate_reverse_lookup(CONFIG.model.drums.categories)

    engine = MidiEngine()
    predictor = Predictor(
        PredictorConfig(
            checkpoint=Path(".data/checkpoints/cp_20260201051208.pt"),
            model=CONFIG.model,
        )
    )

    def handle_step(input_velocities: np.ndarray, step_position: int) -> tuple[np.ndarray, np.ndarray]:
        num_instruments = CONFIG.model.drums.num_instruments

        # Check which categories were triggered
        triggered_categories = category_lookup[input_velocities > 0]
        triggered_categories_unique = np.unique(triggered_categories)

        # Generate the hits array - all triggered categories get set to 1.0, the others remain at 0.0
        hits = np.zeros(num_instruments, dtype=np.float32)
        hits[triggered_categories_unique] = 1.0

        # Run the inference at the current position
        predictor.seek(step_position)
        predictor.process_step(torch.from_numpy(hits))

        # Retrieve the last generated step
        hov = predictor.get_generated_sequence()[-1]
        predicted_offsets = hov[:, 1].numpy()
        predicted_velocities = hov[:, 2].numpy()

        # Check which notes were triggered (based on the categories)
        triggered_notes = category_reverse_lookup[triggered_categories_unique]

        # Set the offsets. Negative values will be clipped to 0.0
        note_offsets = np.zeros(128, dtype=np.float32)
        note_offsets[triggered_notes] = np.clip(predicted_offsets[triggered_categories_unique] + 0.0, 0.0, 1.0)

        # Set the velocities. Increase range to 1..127
        note_velocities = np.zeros(128, dtype=np.uint8)
        note_velocities[triggered_notes] = (np.clip(predicted_velocities[triggered_categories_unique], 0.0, 1.0) * 127.0).astype(np.uint8)

        return note_offsets, note_velocities

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
        prev_pos = Position()

        while engine.running:
            pos = engine.get_position()
            if pos != prev_pos:
                prev_pos = pos
                transport_label.value = f"{pos.bar + 1}.{pos.beat + 1}.{pos.division + 1}+{pos.clock}"
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

    transport_label = ft.Text("-.-.-", size=16, weight=ft.FontWeight.BOLD)

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
    page.title = "Drum Dequantization"

    engine.on_step(handle_step)
    engine.start()


if __name__ == "__main__":
    ft.app(main)
