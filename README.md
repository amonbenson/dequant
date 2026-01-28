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

## Configuration

To see a list of available commands and configuration options, type:

```shell
python -m src --help
```
