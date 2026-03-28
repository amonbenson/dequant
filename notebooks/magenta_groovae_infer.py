import sys
import json
import numpy as np
import traceback
from pathlib import Path
import base64

import note_seq
from magenta.models.music_vae import configs as vae_configs
from magenta.models.music_vae.trained_model import TrainedModel as VaeModel


def run_groovae(input_midis, ckpt_dir, eval_seq_len, tempo_bpm=120, steps_per_beat=4):
    groovae_model = VaeModel(
        vae_configs.CONFIG_MAP["groovae_2bar_humanize"],
        batch_size=1,
        checkpoint_dir_or_path=str(Path(ckpt_dir) / "model.ckpt-3061"),
    )

    def _midi_b64_to_note_seq(midi_b64: np.ndarray):
        midi_data = base64.b64decode(midi_b64)
        return note_seq.midi_io.midi_to_note_sequence(midi_data)

    def _note_seq_to_note_list(ns):
        """Extract notes directly from NoteSequence, bypassing MIDI round-trip.

        Returns a list of dicts with:
          - pitch:    MIDI pitch (int)
          - step:     nearest grid step index (int)
          - offset:   fractional step offset from grid, i.e. step_float - step (float)
          - velocity: normalised velocity in [0, 1] (float)
        """
        steps_ps = tempo_bpm / 60.0 * steps_per_beat
        notes = []
        for note in ns.notes:
            if note.velocity > 0:
                step_float = float(note.start_time) * steps_ps
                step = int(round(step_float))
                offset = step_float - step
                notes.append({
                    "pitch": int(note.pitch),
                    "step": step,
                    "offset": float(offset),
                    "velocity": float(note.velocity) / 127.0,
                })
        return notes

    results = []

    for i, input_midi in enumerate(input_midis):
        if (i + 1) % 50 == 0:
            print(f"GrooVAE {i + 1}/{len(input_midis)}", file=sys.stderr)
        try:
            ns_in = _midi_b64_to_note_seq(input_midi)
            z = groovae_model.encode([ns_in])[0]
            [ns_out] = groovae_model.decode(z, length=eval_seq_len, temperature=0.0)
            note_list = _note_seq_to_note_list(ns_out)
        except Exception as e:
            print(f"Skipping sample {i}: {e}", file=sys.stderr)
            traceback.print_exc()
            note_list = None
        results.append(note_list)

    return results


if __name__ == "__main__":
    data = json.loads(sys.stdin.read())

    preds = run_groovae(
        input_midis=data["input_midis"],
        ckpt_dir=data["ckpt_dir"],
        eval_seq_len=data["eval_seq_len"],
        tempo_bpm=data.get("tempo_bpm", 120),
        steps_per_beat=data.get("steps_per_beat", 4),
    )

    print(json.dumps(preds))
