# UROP - Anomaly Detection in Industrial Machines with Multi-modal Sensors

## Installation

### 1. Prerequisites

Make sure you have Python (3.10+) installed on your system.

```bash
python --version
# or
python3 --version
```

### 2. Cloning the Repository

1.  Clone the repository:
    ```bash
    git clone https://github.com/nMaax/UROP
    ```
2.  Navigate to the project directory:
    ```bash
    cd UROP
    ```

### 3. Setting up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies. You can easily do it using Make

```bash
make install
```

This will setup automatically a venv for you, while installing all dependencies in requirements.txt automatically.

Alternatively, you can set up the virtual environment manually using:

1. Create a new virtual environment:
    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On Unix-like systems (MacOS/Linux):
        ```bash
        source venv/bin/activate
        ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

As a third option, one can use conda as following:

1. Create a new conda environment:
    ```bash
    conda create -n urop python=3.10
    ```

2. Activate the conda environment:
    ```bash
    conda activate urop
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 4. Download dataset

You can automatically download the dataset using

```bash
make data
```

this will create a new folder `data` withinin the project ready-to-use with the Dataset object you can find in `src\` directory, and a folder `datasets`, which contains the uzipped content of `data`.

Alternatively, you can manually download and extract the dataset using:

```bash
python -m utils.download_and_extract
```

## Reproduced Result on Baseline Autoencoder

- Results on BrushlessMotor are available at [this notebook](notebooks/02.02-baseline-autoencoder-brushless-motor-eval.ipynb)
- Results on RoboticArm are available at [this notebook](notebooks/02.12-baseline-autoencoder-robotic-arm-eval.ipynb)

## Transformer-based implementation with RoPe Positional Encoding (my contribution)

- Model architecutre is available [here](models/transformer.py)
- Training of such model on BrushlessMotor is available at [this notebook](notebooks/04.01-naiveTransformer-RoPe-brushless-motor-train.ipynb)
- Training of such model on RoboticArm is available at [this notebook](notebooks/04.11-naiveTransformer-RoPe-robotic-arm-train.ipynb)
- Results on BrushlessMotor available at [this notebook](notebooks/04.02-naiveTransformer-RoPe-brushless-motor-eval.ipynb)
- Results on RoboticArm available at [this notebook](notebooks/04.12-naiveTransformer-RoPe-robotic-arm-eval.ipynb)

> An spreadsheet with a summary of results is available [here](https://docs.google.com/spreadsheets/d/1ulYtD9WzXzzn-zB499xZ_EkRmIqcj1fGJuZP-UhYN2Y/edit?usp=sharing)