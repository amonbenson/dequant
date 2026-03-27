import sys
from pathlib import Path

import torch

from ..config import CONFIG
from ..inference.predictor import Predictor, PredictorConfig


def run_dequantize_rt(checkpoint: Path):
    predictor = Predictor(
        PredictorConfig(
            checkpoint=checkpoint,
            model=CONFIG.model,
        )
    )

    # Do a first step for "warmup"
    predictor.process_step(torch.zeros(CONFIG.model.drums.num_instruments))
    predictor.reset()

    current_hits = torch.zeros(CONFIG.model.drums.num_instruments)

    for line in sys.stdin:
        try:
            parts = line.strip().split(" ")
            match parts:
                case "hit", note:
                    # Record a hit at the specified location
                    current_hits[int(note)] = 1
                case ["step"]:
                    # Predict one step
                    predictor.process_step(current_hits)

                    # Clear current hits
                    current_hits[:] = 0

                    # Output the current position (has already been updated to the next one, so we subtract one)
                    pos = predictor.get_position() - 1
                    print(f"pos {pos}")

                    # Output each hit as a separate note. The delay is included as the second parameter
                    predicted_step = predictor.get_generated_sequence()[pos]
                    for i, hov in enumerate(predicted_step):
                        if hov[0]:
                            print(f"hit {i} {hov[1]} {hov[2]}")
                case "seek", pos:
                    # Seek to a specific step position
                    pos = int(pos)
                    pos = max(0, min(2**16, pos))
                    predictor.seek(pos)
                    print(f"pos {pos}")
                case _:
                    print(f"Unknown command: {line}", file=sys.stderr)
        except Exception as e:
            print(e, file=sys.stderr)
