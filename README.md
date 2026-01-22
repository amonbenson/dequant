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

To start downloading and preprocessing the E-GMD-dataset to the default location (`.data/`), run:

```shell
python -m src.preprocess
```

To see a list of available commands, type:

```shell
python -m src.preprocess --help
```
