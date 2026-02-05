import asyncio
from pathlib import Path

import flet as ft
import flet.canvas as cv
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
        predicted_offsets = (hov[:, 1].numpy() + get_offset_bias()) * get_offset_gain()
        predicted_velocities = (hov[:, 2].numpy() + get_velocity_bias()) * get_velocity_gain()

        # Check which notes were triggered (based on the categories)
        triggered_notes = category_reverse_lookup[triggered_categories_unique]

        # Set the offsets. Negative values will be clipped to 0.0
        note_offsets = np.zeros(128, dtype=np.float32)
        note_offsets[triggered_notes] = np.clip(predicted_offsets[triggered_categories_unique] + 0.0, 0.0, 1.0)

        # Set the velocities. Increase range to 1..127
        note_velocities = np.zeros(128, dtype=np.uint8)
        note_velocities[triggered_notes] = (np.clip(predicted_velocities[triggered_categories_unique], 0.0, 1.0) * 127.0).astype(np.uint8)

        update_sequence_canvas()

        return note_offsets, note_velocities

    def on_input_select(e: ft.Event[ft.Dropdown]):
        if e.control.value:
            engine.open_input(e.control.value)
            page.update()

    def on_output_select(e: ft.Event[ft.Dropdown]):
        if e.control.value:
            engine.open_output(e.control.value)
            page.update()

    def labeled_slider(value: float = 0.0, *, min: float = 0.0, max: float = 1.0, label: str = "Value", on_change=None):
        def handle_slider_update(e: ft.Event[ft.Slider]):
            label_text.value = f"{label}: {e.control.value:.2f}"
            if on_change:
                on_change(e)

        label_text = ft.Text(f"{label}: {value:.2f}", width=150)

        slider = ft.Slider(
            value=value,
            min=min,
            max=max,
            expand=True,
            on_change=handle_slider_update,
        )

        def get_current_value() -> float:
            return slider.value or value

        return ft.Row([label_text, slider], expand=True), get_current_value

    input_dropdown = ft.Dropdown(
        label="MIDI Input",
        options=[ft.dropdown.Option(name) for name in engine.get_input_ports()],
        on_select=on_input_select,
        expand=True,
    )

    output_dropdown = ft.Dropdown(
        label="MIDI Output",
        options=[ft.dropdown.Option(name) for name in engine.get_output_ports()],
        on_select=on_output_select,
        expand=True,
    )

    transport_label = ft.Text("-.-.-", size=16, weight=ft.FontWeight.BOLD)

    offset_bias, get_offset_bias = labeled_slider(value=0.05, min=-1.0, max=1.0, label="Offset Bias")
    offset_gain, get_offset_gain = labeled_slider(value=1.0, min=0.1, max=10.0, label="Offset Gain")
    velocity_bias, get_velocity_bias = labeled_slider(value=0.3, min=-1.0, max=1.0, label="Velocity Bias")
    velocity_gain, get_velocity_gain = labeled_slider(value=0.75, min=0.1, max=2.0, label="Velocity Gain")

    def handle_sequence_canvas_resize(e: cv.CanvasResizeEvent):
        nonlocal sequence_canvas_size
        sequence_canvas_size = (int(e.width), int(e.height))

    sequence_canvas = cv.Canvas(
        height=400,
        expand=True,
        on_resize=handle_sequence_canvas_resize,
    )
    sequence_canvas_size = (1, 1)

    def update_sequence_canvas():
        shapes = []
        w, h = sequence_canvas_size

        # Draw the background
        shapes.append(cv.Rect(0, 0, w, h, paint=ft.Paint(color=ft.Colors.SURFACE_CONTAINER)))

        # Draw each step
        full_sequence = predictor.get_cached_sequence().numpy()
        if len(full_sequence) > 0:
            window_size = 32
            display_end = (predictor.get_position() // window_size + 1) * window_size
            sequence = full_sequence[:display_end]
            num_steps, num_instruments, _ = sequence.shape

            step_width = w / num_steps
            step_height = h / num_instruments

            for step_index, step in enumerate(sequence):
                # Draw a column raster
                if step_index % 2 == 0:
                    shapes.append(
                        cv.Rect(
                            x=step_index * step_width,
                            y=0,
                            width=step_width,
                            height=h,
                            paint=ft.Paint(color=ft.Colors.SURFACE),
                        )
                    )

                for instrument_index, (hit, offset, velocity) in enumerate(step):
                    # Skip notes that are not playing
                    if hit < 0.5:
                        continue

                    # Apply bias and gain
                    offset = np.clip((offset + get_offset_bias()) * get_offset_gain(), 0.0, 1.0)
                    velocity = np.clip((velocity + get_velocity_bias()) * get_velocity_gain(), 0.0, 1.0)

                    # Draw the step as a rectangle
                    shapes.append(
                        cv.Rect(
                            x=(step_index + float(offset)) * step_width,
                            y=(num_instruments - instrument_index - 1) * step_height,
                            width=step_width,
                            height=step_height,
                            paint=ft.Paint(color=ft.Colors.with_opacity(float(velocity), ft.Colors.PRIMARY)),
                        )
                    )

            # Draw the playhead
            shapes.append(
                cv.Rect(
                    x=predictor.get_position() * step_width,
                    y=0,
                    width=1,
                    height=h,
                    paint=ft.Paint(color=ft.Colors.ON_SURFACE),
                )
            )

        # Update the canvas
        sequence_canvas.shapes = shapes
        sequence_canvas.update()

    page.add(
        ft.Column(
            [
                ft.Row([input_dropdown, output_dropdown]),
                transport_label,
                offset_bias,
                offset_gain,
                velocity_bias,
                velocity_gain,
                sequence_canvas,
            ],
            expand=True,
        )
    )

    def on_close(e):
        engine.stop()

    async def ui_updater():
        prev_pos = Position()

        while engine.running:
            pos = engine.get_position()
            if pos != prev_pos:
                prev_pos = pos
                transport_label.value = f"{pos.bar + 1}.{pos.beat + 1}.{pos.division + 1}+{pos.clock}"
                page.update()

            await asyncio.sleep(0.05)

    page.title = "Drum Dequantization"
    page.window.width = 700
    page.window.height = 500
    page.window.always_on_top = True
    page.on_close = on_close

    page.theme = ft.Theme(color_scheme=ft.ColorScheme(primary=ft.Colors.AMBER))
    page.dark_theme = page.theme

    page.run_task(ui_updater)

    engine.on_step(handle_step)
    engine.start()


if __name__ == "__main__":
    ft.app(main)
