import torch
import torch.nn as nn

class BaselineAutoencoder(nn.Module):
    def __init__(self, input_dim=5620):
        super().__init__()

        # Store configuration for later use (in get_config)
        self.config = {'input_dim': input_dim}

        # Explicit layer dimensions for encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),  
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 16)  # Bottleneck layer (latent space)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)  # Reconstruct original input size
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def get_config(self):
        """Returns the configuration of the model as a dictionary"""
        return self.config

    @classmethod
    def from_config(cls, config):
        """Creates a new model instance from the provided configuration dictionary"""
        return cls(input_dim=config['input_dim'])
