# Drum Dequantization

[![CI](https://github.com/amonbenson/dequant/actions/workflows/ci.yaml/badge.svg)](https://github.com/amonbenson/dequant/actions/workflows/ci.yaml)

## Environment Setup

You'll need to have Python 3.11 or higher installed. Create a virtual environment and install the required packages:

```shell
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\Activate.ps1`

pip install --upgrade pip
pip install -r requirements-torch-cuda.txt # If you have a CUDA-capable GPU
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Preprocessing

To start downloading and preprocessing the E-GMD-dataset to the default location (`.data/dataset`), run:

```shell
python -m src preprocess
```

## Training

```shell
python -m src train
```

By default, training will resume from the latest checkpoint. To turn that feature off, type:

```shell
python -m src --config.train.no-resume train
```

To load a specific checkpoint, type:

```shell
python -m src --config.train.resume-from "./data/checkpoints/<name-of-checkpoint>.pt" train
```

## Quantization

To evaluate the model performance, it might make sense to manually quantize a drum pattern and then use the model to dequantize it again. The quantization process (removing velocity and timing offset information) can be triggered using the following commands:

```bash
# Copy a sample form our dataset to the .data directory
cp .data/tmp/egmd-midi/e-gmd-v1.0.0/drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.midi .data/original.midi

# Use the HOV algorithm to quantize the midi file
python -m src quantize .data/original.midi .data/quantized.midi
```

## Dequantization

A quantized MIDI file (obtained either from the command above or from another application) can be dequantized using the following command:

```bash
python -m src dequantize .data/quantized.midi .data/dequantized.midi .data/checkpoints/cp_<timestamp>.pt
```

This will load a previously saved checkpoint and use our model to add back velocity and offset information to the midi data.

## Tensorboard

When training, logs will be written to the `/runs/` directory. You can use tensorboard in a separate terminal to monitor the training progress:

```bash
tensorboard --logdir runs
```

## Using the Realtime App

Use the following command to start the realtime dequantization app:

```bash
python -m src app
```

The connects to a DAW via MIDI and receives realtime messages to dequantize them on the fly. When the app opens, click the "Load Checkpoint" button to load a previously saved checkpoint. Then select the MIDI input and output ports that are routed from and to your DAW. On Mac OS, you will need to setup a virtual MIDI port using the "Audio MIDI Setup" application. On Windows, you can use a tool like loopMIDI to create virtual MIDI ports. After the inputs are selected, the app will receive MIDI messages from the input port, dequantize them using the loaded model, and send the dequantized messages to the output port. It is important that you configure your DAW to send synchronization messages (e.g. MIDI clock), otherwise the dequantization will not work properly.

## Configuration

To see a list of available commands and configuration options, type:

```shell
python -m src --help
```
