"""
Variational Autoencoder (VAE) Implementation

This module implements a VAE with convolutional encoder and decoder
for image data (MNIST, Fashion-MNIST, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional Encoder Network
    Maps input images to latent distribution parameters (mu, log_var)
    """
    def __init__(self, latent_dim=20, input_channels=1):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)              # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)             # 7x7 -> 4x4
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers for mu and log_var
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)
        
    def forward(self, x):
        # Convolutional encoding
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        # Flatten
        h = h.view(h.size(0), -1)
        
        # Compute mu and log_var
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        return mu, log_var


class Decoder(nn.Module):
    """
    Convolutional Decoder Network
    Maps latent codes back to image space
    """
    def __init__(self, latent_dim=20, output_channels=1):
        super(Decoder, self).__init__()
        
        # Fully connected layer from latent to feature maps
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        
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


class VAE(nn.Module):
    """
    Variational Autoencoder
    Combines encoder and decoder with reparameterization trick
    """
    def __init__(self, latent_dim=20, input_channels=1):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(latent_dim, input_channels)
        self.decoder = Decoder(latent_dim, input_channels)
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, I)
        
        Args:
            mu: Mean of latent distribution (batch_size, latent_dim)
            log_var: Log variance of latent distribution (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent code (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log_var)
        eps = torch.randn_like(std)     # epsilon ~ N(0, I)
        z = mu + std * eps              # z = mu + sigma * epsilon
        return z
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            reconstruction: Reconstructed images
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Encode
        mu, log_var = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, log_var
    
    def sample(self, num_samples, device='cpu'):
        """
        Generate new samples by sampling from the prior p(z) = N(0, I)
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated images (num_samples, channels, height, width)
        """
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Decode
            samples = self.decoder(z)
            
        return samples
    
    def encode(self, x):
        """
        Encode input to latent space (returns only mu, not sampling)
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            mu: Mean of latent distribution (used as deterministic encoding)
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """
        Decode latent codes to images
        
        Args:
            z: Latent codes (batch_size, latent_dim)
            
        Returns:
            reconstruction: Decoded images
        """
        return self.decoder(z)


def vae_loss(reconstruction, x, mu, log_var, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + beta * KL Divergence
    
    Args:
        reconstruction: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        beta: Weight for KL divergence (beta=1 for standard VAE)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (Binary Cross-Entropy for images in [0, 1])
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Closed-form KL divergence between N(mu, sigma^2) and N(0, I)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = VAE(latent_dim=20).to(device)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28).to(device)
    reconstruction, mu, log_var = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = vae_loss(reconstruction, x, mu, log_var)
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kl_loss.item():.4f}")
    
    # Test sampling
    samples = model.sample(10, device)
    print(f"\nGenerated samples shape: {samples.shape}")

