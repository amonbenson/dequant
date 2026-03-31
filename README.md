# Drum Dequantization

[![CI](https://github.com/amonbenson/dequant/actions/workflows/ci.yaml/badge.svg)](https://github.com/amonbenson/dequant/actions/workflows/ci.yaml)

## Quick Start

### Max4Live Plugin

To infer the model in realtime, we developed a Max4Live plugin: [GitHub Releases](https://github.com/amonbenson/dequant/releases). The plugin folder includes the prebuilt binary for your specific platform as well as the checkpoint for the 1-layer model. To use it, drag the .amxd file onto a MIDI track within Ableton Live. After a few seconds, it should load the model and you will be able to route MIDI messages through it.

On Mac OS, the app will be blocked by default, because it is unsigned. To allow the app, navigate to System Settings > Privacy & Security and scroll down to the bottom. You will see an option to allow the Dequant-App. After that, you will need to remove the Max4Live device from your Live Set and add it again. This time, you should see an option "Open Anyway".

### Model Checkpoints

Our pretrained model checkpoints for the 1-, 3-, and 5-layer model can be downloaded [here](https://github.com/amonbenson/dequant-checkpoints/releases/tag/checkpoints_v1). To infer a model, use either the Max4Live plugin or one of the commands described below under Dequantization.

## Usage

### Environment Setup

You'll need to have Python 3.11 or higher installed. Create a virtual environment and install the required packages:

```shell
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\Activate.ps1`

pip install --upgrade pip
pip install -r requirements-torch-cuda.txt # If you have a CUDA-capable GPU
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Preprocessing

To download and preprocess the E-GMD-dataset to the default location (`.data/dataset`), run:

```shell
python -m src preprocess
```

### Training

```shell
python -m src train
```

Checkpoints will be saved every epoch. To resume from a specific checkpoint, type:

```shell
python -m src --config.train.resume --config.train.resume-from "./data/checkpoints/<name-of-checkpoint>.pt" train
```

### Configuration

To see a list of all available commands and configuration options, type:

```shell
python -m src --help
```

### Quantization

To evaluate the model performance, it might make sense to manually quantize a drum pattern and then use the model to dequantize it again. The quantization process (removing velocity and timing offset information) can be triggered using the following commands:

```bash
# Copy a sample form our dataset to the .data directory
cp .data/tmp/egmd-midi/e-gmd-v1.0.0/drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.midi original.midi

# Use the HOV algorithm to quantize the midi file
python -m src quantize .original.midi quantized.midi
```

### Dequantization (Inference)

A quantized MIDI file (obtained either from the command above or from another application) can be dequantized using the following command:

```bash
python -m src dequantize quantized.midi dequantized.midi dequant_l5.pt
```

This will load a previously saved checkpoint and use our model to add back velocity and offset information to the quantized midi data. When you use one of the fewer-layer models, you will also need to specify the number of layers:

```bash
python -m src --config.model.transformer.n_layers=1 dequantize quantized.midi dequantized.midi dequant_l1.pt
```

### Tensorboard

While training, logs will be written to the `/runs/` directory. You can use tensorboard in a separate terminal to monitor the training progress:

```bash
tensorboard --logdir runs
```

### Using the Realtime App (Deprecated)

The realtime app was used during development and is now largely replaced by the Max4Live plugin, which provides better performance and lower latency and jitter. If you want to use it nonetheless, type the following command:

```bash
python -m src app
```

The connects to a DAW via MIDI and receives realtime messages to dequantize them on the fly. When the app opens, click the "Load Checkpoint" button to load a previously saved checkpoint. Then select the MIDI input and output ports that are routed from and to your DAW. On Mac OS, you will need to setup a virtual MIDI port using the "Audio MIDI Setup" application. On Windows, you can use a tool like loopMIDI to create virtual MIDI ports. After the inputs are selected, the app will receive MIDI messages from the input port, dequantize them using the loaded model, and send the dequantized messages to the output port. It is important that you configure your DAW to send synchronization messages (e.g. MIDI clock), otherwise the dequantization will not work properly.
