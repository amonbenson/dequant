import sys
from pathlib import Path

import torch

from ..config import CONFIG
from ..data.drum_category import DrumCategory
from ..inference.predictor import Predictor, PredictorConfig


def run_dequantize_rt(checkpoint: Path):
    predictor = Predictor(
        PredictorConfig(
            checkpoint=checkpoint,
            model=CONFIG.model,
        )
    )

    categories = CONFIG.model.drums.categories
    pitch_to_category_lookup = DrumCategory.generate_forward_lookup(categories)

    # Do a first step for "warmup"
    predictor.process_step(torch.zeros(CONFIG.model.drums.num_instruments))
    predictor.reset()

    # Fire a ready message
    print("ready", flush=True)

    current_pitches = torch.zeros(128, dtype=torch.float32)
    current_hits = torch.zeros(CONFIG.model.drums.num_instruments, dtype=torch.float32)

    try:
        for line in sys.stdin:
            try:
                parts = line.strip().split(" ")
                match parts:
                    case "note", pitch:
                        # Store a note at the specified location
                        current_pitches[int(pitch)] = 1
                    case ["step"]:
                        # Generate the hits tensor
                        current_hits[:] = 0
                        hit_indices = pitch_to_category_lookup[torch.where(current_pitches)[0]]
                        hit_indices = hit_indices[hit_indices != -1]  # filter out any pitches that couldn't be mapped to a distinct category
                        current_hits[hit_indices] = 1

                        # Predict one step
                        predictor.process_step(current_hits)

                        # Output the current position (has already been updated to the next one, so we subtract one)
                        pos = predictor.get_position() - 1
                        print(f"pos {pos}", flush=True)

                        # Output each hit as a separate note. The delay is included as the second parameter
                        predicted_step = predictor.get_generated_sequence()[pos]
                        for pitch, active in enumerate(current_pitches):
                            if active:
                                # Lookup the corresponding hov index
                                instrument_index = pitch_to_category_lookup[pitch].item()
                                if instrument_index != -1:
                                    # Get the predicted offset and velocity for that specific hit
                                    [hit, offset, velocity] = predicted_step[instrument_index]
                                    assert hit == 1, "Internal error with lookup of instrument categories"
                                    print(f"note {pitch} {offset} {velocity}", flush=True)
                                else:
                                    # Pitch has no category associated with it. Return default offset and velocity
                                    print(f"note {pitch} 0.0 0.8", flush=True)

                        # Clear current notes
                        current_pitches[:] = 0
                    case "seek", pos:
                        # Calculate and constraint the specified position
                        pos = int(pos)
                        pos = max(0, min(2**16, pos))

                        # Seek and output only if the position changed
                        if predictor.get_position() != pos:
                            predictor.seek(pos)
                            print(f"pos {pos}", flush=True)
                    case ["exit"]:
                        break
                    case _:
                        print(f"error Unknown command {line}", flush=True, file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"error {e}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
