import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=5620):
        super().__init__()

        # Explicit layer dimensions for encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),  # Input dim example: 4704 (e.g., flattened mic+acc+gyro window)
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

# Example usage
if __name__ == "__main__":
    model = Autoencoder()
    dummy_input = torch.randn(32, 4704)  # Example batch size of 32
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)