# UROP - Anomaly Detection in Industrial Machines with Multi-modal Sensors

## Reproduced Result on Baseline Autoencoder

- Results on BrushlessMotor are available at [this notebook](notebooks/02.02-baseline-autoencoder-brushless-motor-eval.ipynb)
- Results on RoboticArm are available at [this notebook](notebooks/02.12-baseline-autoencoder-robotic-arm-eval.ipynb)

## Naive Transformer implementation

- Results on RoboticArm available at [this notebook](notebooks/03.01-naiveTransformer.ipynb)

## RoPe Position Encoding Transformer implementation

- Results on RoboticArm available at [this notebook](notebooks/03.02-naiveTransformer-RoPe.ipynb)

## Possible improvements

- Use He or Xavier init
- Add LayerNorm after positional encoding to stabilize training.
- Use separate projections for mic and motion before merging into the Transformer.
- Increase Transformer depth (e.g., 6–8 layers) for better pattern extraction
- Add L1 regularization to the loss
- Try Mahalanobis distance instead of MSE
- Inject Gaussian noise during training
- Increase dropout
- Use gradient clipping
- Transformer AutoEncoder approach

## Pre-text

- Masking part of the TS
