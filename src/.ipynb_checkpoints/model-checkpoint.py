# src/model.py
import torch
import torch.nn as nn
from src.config import latent_dimension_size

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=latent_dimension_size): # Default latant space dimension is 32
        super(Autoencoder, self).__init__()
        
        # Encoder: Compresses 784 pixels down to latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Expands latent_dim back to 784 pixels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid() # Sigmoid forces output to [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28) # Reshape back to image format