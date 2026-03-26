import sys
import json
import numpy as np
import traceback
import tempfile
from pathlib import Path
import base64

import note_seq
from magenta.models.music_vae import configs as vae_configs
from magenta.models.music_vae.trained_model import TrainedModel as VaeModel


def run_groovae(input_midis, ckpt_dir, eval_seq_len):
    groovae_model = VaeModel(
        vae_configs.CONFIG_MAP["groovae_4bar"],
        batch_size=1,
        checkpoint_dir_or_path=str(Path(ckpt_dir) / "model.ckpt-2721"),
    )

    def _midi_b64_to_note_seq(midi_b64: np.ndarray):
        midi_data = base64.b64decode(midi_b64)
        return note_seq.midi_io.midi_to_note_sequence(midi_data)

    def _note_seq_to_midi_b64(ns):
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            note_seq.midi_io.note_sequence_to_midi_file(ns, str(tmp_path))
            with open(tmp_path, "rb") as f:
                midi_b64 = base64.b64encode(f.read()).decode()
                return midi_b64
        except Exception as e:
            print(f"Warning: MIDI conversion failed ({e})", file=sys.stderr)
            traceback.print_exc()
        finally:
            tmp_path.unlink()

    results = []

    for i, input_midi in enumerate(input_midis):
        if (i + 1) % 50 == 0:
            print(f"GrooVAE {i + 1}/{len(input_midis)}", file=sys.stderr)
        try:
            ns_in = _midi_b64_to_note_seq(input_midi)
            z = groovae_model.encode([ns_in])[0]
            [ns_out] = groovae_model.decode(z, length=eval_seq_len, temperature=1.0)
            output_midi = _note_seq_to_midi_b64(ns_out)
        except Exception as e:
            print(f"Skipping sample {i}: {e}", file=sys.stderr)
            traceback.print_exc()
            output_midi = ""
        results.append(output_midi)

    return results


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())

    preds = run_groovae(
        input_midis=data["input_midis"],
        ckpt_dir=data["ckpt_dir"],
        eval_seq_len=data["eval_seq_len"],
    )

    print(json.dumps(preds))
