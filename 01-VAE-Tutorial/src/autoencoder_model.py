"""
Standard Autoencoder (AE) Implementation

This module implements a deterministic autoencoder for comparison with VAE.
Unlike VAE, this produces a single deterministic latent code for each input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AE_Encoder(nn.Module):
    """
    Convolutional Encoder Network (Deterministic)
    Maps input images directly to latent codes
    """
    def __init__(self, latent_dim=20, input_channels=1):
        super(AE_Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layer to latent code
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # Convolutional encoding
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        # Flatten and map to latent space
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        
        return z


class AE_Decoder(nn.Module):
    """
    Convolutional Decoder Network
    Maps latent codes back to image space
    """
    def __init__(self, latent_dim=20, output_channels=1):
        super(AE_Decoder, self).__init__()
        
        # Fully connected layer from latent to feature maps
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        # Fully connected
        h = self.fc(z)
        h = h.view(h.size(0), 128, 4, 4)
        
        # Transposed convolutions
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        reconstruction = torch.sigmoid(self.deconv3(h))
        
        # Crop to 28x28
        reconstruction = reconstruction[:, :, :28, :28]
        
        return reconstruction


class Autoencoder(nn.Module):
    """
    Standard (Deterministic) Autoencoder
    """
    def __init__(self, latent_dim=20, input_channels=1):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = AE_Encoder(latent_dim, input_channels)
        self.decoder = AE_Decoder(latent_dim, input_channels)
        
    def forward(self, x):
        """
        Forward pass through autoencoder
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            reconstruction: Reconstructed images
            z: Latent codes
        """
        # Encode
        z = self.encoder(x)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, z
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent codes to images"""
        return self.decoder(z)


def ae_loss(reconstruction, x):
    """
    Autoencoder Loss = Reconstruction Loss (MSE or BCE)
    
    Args:
        reconstruction: Reconstructed images
        x: Original images
        
    Returns:
        loss: Reconstruction loss
    """
    # Binary Cross-Entropy for images in [0, 1]
    loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    
    return loss


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = Autoencoder(latent_dim=20).to(device)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    reconstruction, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent code shape: {z.shape}")
    
    # Test loss
    loss = ae_loss(reconstruction, x)
    print(f"\nReconstruction loss: {loss.item():.4f}")

