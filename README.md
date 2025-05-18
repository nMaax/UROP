# UROP - Anomaly Detection in Industrial Machines with Multi-modal Sensors

## Installation

### Prerequisites

Make sure you have Python installed on your system.

* Python 3.10+
    ```bash
    python --version
    # or
    python3 --version
    ```

### Cloning the Repository

1.  Clone the repository:
    ```bash
    git clone https://github.com/nMaax/UROP
    ```
2.  Navigate to the project directory:
    ```bash
    cd UROP
    ```

### Setting up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies. You can easily do it using Make

```bash
make install
```

This will setup automatically a venv for you, while installing all dependencies in requirements.txt automatically.

### Download dataset

You can automatically download the dataset using

```bash
make data
```

this will create a new folder `data` withinin the project ready-to-use with the Dataset object you can find in `src\` directory.

## Reproduced Result on Baseline Autoencoder

- Results on BrushlessMotor are available at [this notebook](notebooks/02.02-baseline-autoencoder-brushless-motor-eval.ipynb)
- Results on RoboticArm are available at [this notebook](notebooks/02.12-baseline-autoencoder-robotic-arm-eval.ipynb)

## Naive Transformer implementation

- Results on RoboticArm available at [this notebook](notebooks/03.01-naiveTransformer.ipynb)

## RoPe Position Encoding Transformer implementation

- Results on RoboticArm available at [this notebook](notebooks/03.02-naiveTransformer-RoPe.ipynb)
