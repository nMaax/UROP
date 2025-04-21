import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=3, encoder_block=None, decoder_block=None):
        super(Autoencoder, self).__init__()

        hidden_dim = 2048  # Size of the hidden layers

        # Define default encoder and decoder block if none provided
        if encoder_block is None:
            encoder_block = lambda in_dim, out_dim: [nn.Linear(in_dim, out_dim), nn.ReLU()]
        if decoder_block is None:
            decoder_block = lambda in_dim, out_dim: [nn.Linear(in_dim, out_dim), nn.ReLU()]

        # Build encoder: input_dim -> hidden_dim (x num_layers-1) -> latent_dim
        encoder_layers = encoder_block(input_dim, hidden_dim)
        for _ in range(num_layers - 1):
            encoder_layers += encoder_block(hidden_dim, hidden_dim)
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))  # Bottleneck layer
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: latent_dim -> hidden_dim (x num_layers-1) -> input_dim
        decoder_layers = decoder_block(latent_dim, hidden_dim)
        for _ in range(num_layers - 1):
            decoder_layers += decoder_block(hidden_dim, hidden_dim)
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))  # Output layer
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Encode the input to a latent representation
        latent = self.encoder(x)
        # Decode the latent representation back to the input space
        reconstructed = self.decoder(latent)
        return reconstructed

# Example usage:
if __name__ == "__main__":
    batch_size = 1024
    sequence_length = 100  # For instance, 100 time steps
    input_dim = sequence_length
    latent_dim = 16

    # Instantiate the model
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Generate dummy input data
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(dummy_input)
    
    # Print input and output shapes
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)