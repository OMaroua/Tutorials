"""
Utility functions for VAE tutorial
Includes visualization and helper functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_latent_space(model, data_loader, device, method='pca', is_vae=True, max_samples=10000):
    """
    Visualize the latent space using PCA or t-SNE
    
    Args:
        model: Trained VAE or AE model
        data_loader: DataLoader for the dataset
        device: torch device
        method: 'pca' or 'tsne'
        is_vae: True if model is VAE, False if AE
        max_samples: Maximum number of samples to use
        
    Returns:
        fig: matplotlib figure
    """
    model.eval()
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if len(latent_codes) * data_loader.batch_size >= max_samples:
                break
                
            data = data.to(device)
            
            if is_vae:
                mu, _ = model.encoder(data)
                z = mu  # Use mean for visualization
            else:
                z = model.encoder(data)
            
            latent_codes.append(z.cpu().numpy())
            labels.append(target.numpy())
    
    latent_codes = np.concatenate(latent_codes, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(latent_codes)
        title = f'PCA Projection of Latent Space'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(latent_codes)
        title = f't-SNE Projection of Latent Space'
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', 
                        alpha=0.6, s=20)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Digit')
    plt.grid(alpha=0.3)
    
    return fig


def visualize_reconstructions(model, data_loader, device, is_vae=True, num_images=10):
    """
    Visualize original images and their reconstructions
    
    Args:
        model: Trained VAE or AE model
        data_loader: DataLoader for the dataset
        device: torch device
        is_vae: True if model is VAE, False if AE
        num_images: Number of images to display
        
    Returns:
        fig: matplotlib figure
    """
    model.eval()
    
    # Get a batch of data
    data, _ = next(iter(data_loader))
    data = data[:num_images].to(device)
    
    with torch.no_grad():
        if is_vae:
            reconstruction, _, _ = model(data)
        else:
            reconstruction, _ = model(data)
    
    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    return fig


def interpolate_latent(model, data_loader, device, is_vae=True, num_steps=10):
    """
    Interpolate between two points in latent space
    
    Args:
        model: Trained VAE or AE model
        data_loader: DataLoader for the dataset
        device: torch device
        is_vae: True if model is VAE, False if AE
        num_steps: Number of interpolation steps
        
    Returns:
        fig: matplotlib figure
    """
    model.eval()
    
    # Get two random images
    data, labels = next(iter(data_loader))
    img1, img2 = data[0:1].to(device), data[1:2].to(device)
    label1, label2 = labels[0].item(), labels[1].item()
    
    with torch.no_grad():
        # Encode to latent space
        if is_vae:
            z1, _ = model.encoder(img1)
            z2, _ = model.encoder(img2)
        else:
            z1 = model.encoder(img1)
            z2 = model.encoder(img2)
        
        # Interpolate
        interpolated_images = []
        alphas = np.linspace(0, 1, num_steps)
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img_interp = model.decoder(z_interp)
            interpolated_images.append(img_interp.cpu().squeeze())
    
    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
    fig.suptitle(f'Interpolation from digit {label1} to digit {label2}', fontsize=14)
    
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=8)
    
    plt.tight_layout()
    return fig


def generate_samples(model, device, num_samples=16):
    """
    Generate new samples from the prior (VAE only)
    
    Args:
        model: Trained VAE model
        device: torch device
        num_samples: Number of samples to generate
        
    Returns:
        fig: matplotlib figure
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decoder(z)
    
    # Plot
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    fig.suptitle('Generated Samples from Prior N(0, I)', fontsize=14)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx].cpu().squeeze(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_losses(train_losses, val_losses=None, loss_components=None):
    """
    Plot training (and validation) losses
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        loss_components: Dict with 'recon' and 'kl' lists (for VAE)
        
    Returns:
        fig: matplotlib figure
    """
    if loss_components is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(train_losses, label='Train')
        if val_losses is not None:
            axes[0].plot(val_losses, label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(loss_components['recon'], label='Reconstruction')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].grid(alpha=0.3)
        
        # KL divergence
        axes[2].plot(loss_components['kl'], label='KL Divergence')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence')
        axes[2].grid(alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train')
        if val_losses is not None:
            ax.plot(val_losses, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_latent_2d_manifold(model, device, n=20, digit_size=28):
    """
    Plot 2D manifold of latent space (only works for 2D latent space)
    
    Args:
        model: Trained VAE model with 2D latent space
        device: torch device
        n: Number of points along each axis
        digit_size: Size of output images
        
    Returns:
        fig: matplotlib figure
    """
    if model.latent_dim != 2:
        print(f"Warning: This function is designed for 2D latent space, but model has {model.latent_dim}D")
        return None
    
    model.eval()
    
    # Create a grid in latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    
    figure = np.zeros((digit_size * n, digit_size * n))
    
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.FloatTensor([[xi, yi]]).to(device)
                x_decoded = model.decoder(z)
                digit = x_decoded[0].cpu().squeeze().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.title('2D Latent Space Manifold')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    print("Utility functions for VAE visualization")
    print("Import these functions in your notebook or training script")

