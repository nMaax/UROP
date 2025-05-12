# UROP - Anomaly Detection in Industrial Machines with Multi-modal Sensors

## Reproduced Result on Baseline Autoencoder

- Results on BrushlessMotor are available at [this notebook](notebooks/02.02-baseline-autoencoder-brushless-motor-eval.ipynb)
- Results on RoboticArm are available at [this notebook](notebooks/02.12-baseline-autoencoder-robotic-arm-eval.ipynb)

## Naive Transformer implementation

Simple Transformer Encoder with Sinusoidal positional encoding. Next steps could be:
- Using different positional encodings
- Reducing the number of transformer layers (at the moment with just 50 batches the model is able to learn) 

- Results on RoboticArm available at [this notebook](notebooks/03.01-naiveTransformer.ipynb)