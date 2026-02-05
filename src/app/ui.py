import asyncio
from pathlib import Path
from threading import Lock
from typing import Callable

import flet as ft
import flet.canvas as cv
import numpy as np
import torch

from .engine import MidiEngine, Position
from ..inference.predictor import Predictor, PredictorConfig
from ..data.drum_category import DrumCategory
from ..config import CONFIG


class LabeledSlider(ft.Row):
    def __init__(
        self,
        value: float = 0.0,
        min: float = 0.0,
        max: float = 1.0,
        label: str = "Value",
        on_change: Callable[[float], None] | None = None,
    ):
        self._label = label
        self._on_change = on_change
        self._default_value = value

        self._label_text = ft.Text(f"{label}: {value:.2f}", width=150)
        self._slider = ft.Slider(
            value=value,
            min=min,
            max=max,
            expand=True,
            on_change=self._handle_slider_change,
        )

        super().__init__(controls=[self._label_text, self._slider], expand=True)

    def _handle_slider_change(self, e: ft.Event[ft.Slider]):
        self._label_text.value = f"{self._label}: {e.control.value:.2f}"
        if self._on_change:
            self._on_change(e.control.value or self._default_value)

    @property
    def current_value(self) -> float:
        return self._slider.value or self._default_value


class SequenceCanvas(cv.Canvas):
    def __init__(
        self,
        predictor: Predictor,
        predictor_lock: Lock,
        get_offset_bias: Callable[[], float],
        get_offset_gain: Callable[[], float],
        get_velocity_bias: Callable[[], float],
        get_velocity_gain: Callable[[], float],
        height: int = 400,
    ):
        super().__init__(height=height, expand=True, on_resize=self._handle_resize)

        self._predictor = predictor
        self._predictor_lock = predictor_lock
        self._get_offset_bias = get_offset_bias
        self._get_offset_gain = get_offset_gain
        self._get_velocity_bias = get_velocity_bias
        self._get_velocity_gain = get_velocity_gain
        self._size = (1, 1)

    def _handle_resize(self, e: cv.CanvasResizeEvent):
        self._size = (int(e.width), int(e.height))

    def _build_sequence_shapes(
        self,
        sequence: np.ndarray,
        position: int,
        w: float,
        h: float,
    ) -> list[cv.Rect]:
        if len(sequence) == 0:
            return []

        shapes: list[cv.Rect] = []
        num_steps, num_instruments, _ = sequence.shape
        step_width = w / num_steps
        step_height = h / num_instruments

        # Draw the grid
        for step_index in range(0, num_steps, 2):
            shapes.append(
                cv.Rect(
                    x=step_index * step_width,
                    y=0,
                    width=step_width,
                    height=h,
                    paint=ft.Paint(color=ft.Colors.SURFACE),
                )
            )

        offset_bias = self._get_offset_bias()
        offset_gain = self._get_offset_gain()
        velocity_bias = self._get_velocity_bias()
        velocity_gain = self._get_velocity_gain()

        # Draw each step
        step_indices, instrument_indices = np.where(sequence[:, :, 0] >= 0.5)
        for step_index, instrument_index in zip(step_indices, instrument_indices):
            _, offset, velocity = sequence[step_index, instrument_index]

            offset = np.clip((offset + offset_bias) * offset_gain, 0.0, 1.0)
            velocity = np.clip((velocity + velocity_bias) * velocity_gain, 0.0, 1.0)

            shapes.append(
                cv.Rect(
                    x=(step_index + float(offset)) * step_width,
                    y=(num_instruments - instrument_index - 1) * step_height,
                    width=step_width,
                    height=step_height,
                    paint=ft.Paint(color=ft.Colors.with_opacity(float(velocity), ft.Colors.PRIMARY)),
                )
            )

        # Playhead position
        # shapes.append(
        #     cv.Rect(
        #         x=position * step_width,
        #         y=0,
        #         width=1,
        #         height=h,
        #         paint=ft.Paint(color=ft.Colors.ON_SURFACE),
        #     )
        # )

        return shapes

    def render(self):
        w, h = self._size
        background = cv.Rect(0, 0, w, h, paint=ft.Paint(color=ft.Colors.SURFACE_CONTAINER))

        with self._predictor_lock:
            sequence = self._predictor.get_context_sequence().numpy(force=True)
            position = self._predictor.get_position()

        # window_size = 32
        # display_end = (position // window_size + 1) * window_size
        # sequence = full_sequence[:display_end]

        sequence_shapes = self._build_sequence_shapes(sequence, position, w, h)

        self.shapes = [background, *sequence_shapes]
        self.update()


class TransportDisplay(ft.Text):
    def __init__(self):
        super().__init__("-.-.-", size=16, weight=ft.FontWeight.BOLD)

    def set_position(self, pos: Position):
        self.value = f"{pos.bar + 1}.{pos.beat + 1}.{pos.division + 1}+{pos.clock}"


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
    predictor_lock = Lock()

    offset_bias_slider = LabeledSlider(value=0.05, min=-1.0, max=1.0, label="Offset Bias")
    offset_gain_slider = LabeledSlider(value=1.0, min=0.1, max=10.0, label="Offset Gain")
    velocity_bias_slider = LabeledSlider(value=0.3, min=-1.0, max=1.0, label="Velocity Bias")
    velocity_gain_slider = LabeledSlider(value=0.75, min=0.1, max=2.0, label="Velocity Gain")

    sequence_canvas = SequenceCanvas(
        predictor=predictor,
        predictor_lock=predictor_lock,
        get_offset_bias=lambda: offset_bias_slider.current_value,
        get_offset_gain=lambda: offset_gain_slider.current_value,
        get_velocity_bias=lambda: velocity_bias_slider.current_value,
        get_velocity_gain=lambda: velocity_gain_slider.current_value,
    )

    transport_display = TransportDisplay()

    def handle_step(input_velocities: np.ndarray, step_position: int) -> tuple[np.ndarray, np.ndarray]:
        num_instruments = CONFIG.model.drums.num_instruments

        triggered_categories = category_lookup[input_velocities > 0]
        triggered_categories_unique = np.unique(triggered_categories)

        hits = np.zeros(num_instruments, dtype=np.float32)
        hits[triggered_categories_unique] = 1.0

        with predictor_lock:
            predictor.seek(step_position)
            predictor.process_step(torch.from_numpy(hits))

            hov = predictor.get_generated_sequence()[-1]
            predicted_offsets = (hov[:, 1].numpy() + offset_bias_slider.current_value) * offset_gain_slider.current_value
            predicted_velocities = (hov[:, 2].numpy() + velocity_bias_slider.current_value) * velocity_gain_slider.current_value

        triggered_notes = category_reverse_lookup[triggered_categories_unique]

        note_offsets = np.zeros(128, dtype=np.float32)
        note_offsets[triggered_notes] = np.clip(predicted_offsets[triggered_categories_unique] + 0.0, 0.0, 1.0)

        note_velocities = np.zeros(128, dtype=np.uint8)
        note_velocities[triggered_notes] = (np.clip(predicted_velocities[triggered_categories_unique], 0.0, 1.0) * 127.0).astype(np.uint8)

        return note_offsets, note_velocities

    def on_input_select(e: ft.Event[ft.Dropdown]):
        if e.control.value:
            engine.open_input(e.control.value)
            page.update()

    def on_output_select(e: ft.Event[ft.Dropdown]):
        if e.control.value:
            engine.open_output(e.control.value)
            page.update()

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

    async def ui_updater():
        prev_pos = Position()

        while engine.running:
            pos = engine.get_position()
            if pos != prev_pos:
                prev_pos = pos
                transport_display.set_position(pos)

            sequence_canvas.render()
            page.update()
            await asyncio.sleep(0.05)

    def on_close(_):
        engine.stop()

    page.title = "Drum Dequantization"
    page.window.width = 700
    page.window.height = 500
    page.window.always_on_top = True
    page.on_close = on_close
    page.theme = ft.Theme(color_scheme=ft.ColorScheme(primary=ft.Colors.AMBER))
    page.dark_theme = page.theme

    page.add(
        ft.Column(
            [
                ft.Row([input_dropdown, output_dropdown]),
                transport_display,
                offset_bias_slider,
                offset_gain_slider,
                velocity_bias_slider,
                velocity_gain_slider,
                sequence_canvas,
            ],
            expand=True,
        )
    )

    page.run_task(ui_updater)

    engine.on_step(handle_step)
    engine.start()


if __name__ == "__main__":
    ft.app(main)
