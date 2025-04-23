# UROP - Anomaly Detection in Industrial Machines with Multi-modal Sensors

### Reproduced Result on Baseline Autoencoder

- Results on BrushlessMotor are available at [this notebook](notebooks/02.02-baseline-autoencoder-brushless-motor-eval.ipynb)
- Results on RoboticArm are available at [this notebook](notebooks/02.12-baseline-autoencoder-robotic-arm-eval.ipynb)

### Notes on IMAD-DS
- Source vs. Target: Controlled in-lab setting vs. actuall industry plan recordings
- Sensors' data is supposed to be taken synchronized: look at the corresponding CSV file and pick the right recordings togheter, then feed synchronized windows to the model
- Pre-text ideas:
    - Contrastive learning techniques: SimCLR, MoCo, or TS-TCC
    - Contrastive learning augmentations: jittering, noise, masking, etc.
    - Contrastive learning on log-mel spectrograms as images (using DINO, Mamba ViT, and other models)
    - ...
- Training ideas:
    - Using distillation if model is too large
    - Using quantization (INT8) and pruning
    - Tiny Transformers + 1D Convolutions
    - Train on source, fine tune on target
    - ...
- Metrics:
    - AUC
    - pAUC
    - Harmonic Mean of AUC-pAUC
    - F1 Score
    - ...