"""
Create visualizations for VAE tutorial
Run this script to generate diagrams and plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
import os
os.makedirs('../assets/images', exist_ok=True)

def create_autoencoder_architecture():
    """Create autoencoder architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Colors
    input_color = '#e8f4f8'
    encoder_color = '#b8dce8'
    latent_color = '#6c5ce7'
    decoder_color = '#b8dce8'
    output_color = '#e8f4f8'
    
    # Input
    input_box = FancyBboxPatch((0.5, 2), 1.5, 2, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 3, 'Input\nx ∈ ℝ^D', ha='center', va='center', fontsize=12, weight='bold')
    
    # Encoder
    encoder_box = FancyBboxPatch((3, 1.5), 2, 3, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=encoder_color, linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(4, 3, 'Encoder\nf_φ(x)', ha='center', va='center', fontsize=12, weight='bold')
    
    # Latent
    latent_box = FancyBboxPatch((6, 2.25), 1.5, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=latent_color, linewidth=2)
    ax.add_patch(latent_box)
    ax.text(6.75, 3, 'z ∈ ℝ^d', ha='center', va='center', fontsize=12, weight='bold', color='white')
    
    # Decoder
    decoder_box = FancyBboxPatch((8.5, 1.5), 2, 3, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=decoder_color, linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(9.5, 3, 'Decoder\ng_θ(z)', ha='center', va='center', fontsize=12, weight='bold')
    
    # Output
    output_box = FancyBboxPatch((11.5, 2), 1.5, 2, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.25, 3, 'Output\nx̂ ∈ ℝ^D', ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    ax.annotate('', xy=(3, 3), xytext=(2, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 3), xytext=(5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 3), xytext=(7.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(11.5, 3), xytext=(10.5, 3), arrowprops=arrow_props)
    
    # Loss arrow (reconstruction)
    ax.annotate('', xy=(1.25, 1.5), xytext=(12.25, 1.5), 
                arrowprops=dict(arrowstyle='<->', lw=2, color='red', linestyle='--'))
    ax.text(6.75, 0.8, 'Reconstruction Loss: ||x - x̂||²', ha='center', fontsize=11, color='red')
    
    plt.title('Autoencoder Architecture', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../assets/images/autoencoder_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created autoencoder_architecture.png")


def create_vae_architecture():
    """Create VAE architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Colors
    input_color = '#e8f4f8'
    encoder_color = '#74b9ff'
    sampling_color = '#6c5ce7'
    decoder_color = '#74b9ff'
    output_color = '#e8f4f8'
    
    # Input
    input_box = FancyBboxPatch((0.5, 2.5), 1.5, 2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 3.5, 'Input\nx ∈ ℝ^D', ha='center', va='center', fontsize=12, weight='bold')
    
    # Encoder
    encoder_box = FancyBboxPatch((3, 2), 2.5, 3, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=encoder_color, linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(4.25, 3.5, 'Encoder\nq_φ(z|x)', ha='center', va='center', fontsize=12, weight='bold')
    
    # Mu and Sigma
    mu_box = FancyBboxPatch((6.5, 4), 1.2, 0.8, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#fdcb6e', linewidth=1.5)
    ax.add_patch(mu_box)
    ax.text(7.1, 4.4, 'μ', ha='center', va='center', fontsize=14, weight='bold')
    
    sigma_box = FancyBboxPatch((6.5, 2.5), 1.2, 0.8, boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor='#fdcb6e', linewidth=1.5)
    ax.add_patch(sigma_box)
    ax.text(7.1, 2.9, 'σ', ha='center', va='center', fontsize=14, weight='bold')
    
    # Sampling (reparameterization)
    sampling_box = FancyBboxPatch((8.5, 2.5), 2, 2, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor=sampling_color, linewidth=2)
    ax.add_patch(sampling_box)
    ax.text(9.5, 4, 'Sampling', ha='center', va='center', fontsize=11, weight='bold', color='white')
    ax.text(9.5, 3.5, 'z = μ + σ⊙ε', ha='center', va='center', fontsize=10, color='white')
    ax.text(9.5, 3, 'ε ~ N(0,I)', ha='center', va='center', fontsize=9, color='white')
    
    # Decoder
    decoder_box = FancyBboxPatch((11.5, 2), 2.5, 3, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=decoder_color, linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(12.75, 3.5, 'Decoder\np_θ(x|z)', ha='center', va='center', fontsize=12, weight='bold')
    
    # Output
    output_box = FancyBboxPatch((15, 2.5), 1.5, 2, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(15.75, 3.5, 'Output\nx̂ ∈ ℝ^D', ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    ax.annotate('', xy=(3, 3.5), xytext=(2, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 4.4), xytext=(5.5, 4.2), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 2.9), xytext=(5.5, 3.2), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 4.2), xytext=(7.7, 4.3), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 3), xytext=(7.7, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(11.5, 3.5), xytext=(10.5, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(15, 3.5), xytext=(14, 3.5), arrowprops=arrow_props)
    
    # Loss annotations
    ax.annotate('', xy=(1.25, 1.5), xytext=(15.75, 1.5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red', linestyle='--'))
    ax.text(8.5, 0.8, 'Reconstruction Loss', ha='center', fontsize=11, color='red', weight='bold')
    
    ax.text(9.5, 6, 'KL Divergence: D_KL(q_φ(z|x) || p(z))', ha='center', fontsize=11, 
            color='green', weight='bold')
    
    plt.title('Variational Autoencoder (VAE) Architecture', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../assets/images/vae_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created vae_architecture.png")


def create_latent_space_comparison():
    """Create latent space comparison between AE and VAE"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate synthetic latent representations
    np.random.seed(42)
    n_samples = 500
    n_classes = 10
    
    # AE: Clustered but with gaps
    ae_latents = []
    for i in range(n_classes):
        angle = 2 * np.pi * i / n_classes
        center_x = 3 * np.cos(angle)
        center_y = 3 * np.sin(angle)
        cluster = np.random.randn(n_samples // n_classes, 2) * 0.3 + [center_x, center_y]
        ae_latents.append(cluster)
    ae_latents = np.vstack(ae_latents)
    ae_labels = np.repeat(range(n_classes), n_samples // n_classes)
    
    # VAE: More spread out and continuous
    vae_latents = []
    for i in range(n_classes):
        angle = 2 * np.pi * i / n_classes
        center_x = 2 * np.cos(angle)
        center_y = 2 * np.sin(angle)
        cluster = np.random.randn(n_samples // n_classes, 2) * 0.6 + [center_x, center_y]
        vae_latents.append(cluster)
    vae_latents = np.vstack(vae_latents)
    vae_labels = np.repeat(range(n_classes), n_samples // n_classes)
    
    # Plot AE
    scatter1 = axes[0].scatter(ae_latents[:, 0], ae_latents[:, 1], 
                               c=ae_labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title('Autoencoder Latent Space\n(Discontinuous, Patchy)', 
                      fontsize=14, weight='bold')
    axes[0].set_xlabel('z₀', fontsize=12)
    axes[0].set_ylabel('z₁', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    
    # Plot VAE
    scatter2 = axes[1].scatter(vae_latents[:, 0], vae_latents[:, 1],
                               c=vae_labels, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title('VAE Latent Space\n(Continuous, Smooth)', 
                      fontsize=14, weight='bold')
    axes[1].set_xlabel('z₀', fontsize=12)
    axes[1].set_ylabel('z₁', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    
    # Add N(0,I) contours for VAE
    from scipy.stats import multivariate_normal
    x, y = np.mgrid[-5:5:.1, -5:5:.1]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    axes[1].contour(x, y, rv.pdf(pos), levels=3, colors='red', alpha=0.3, linewidths=1.5)
    
    plt.tight_layout()
    plt.savefig('../assets/images/latent_space_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created latent_space_comparison.png")


def create_training_curves():
    """Create synthetic training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    epochs = np.arange(1, 51)
    
    # Total loss
    total_loss = 200 * np.exp(-epochs/15) + 150 + np.random.randn(50) * 3
    axes[0].plot(epochs, total_loss, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_title('Total VAE Loss', fontsize=14, weight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Reconstruction loss
    recon_loss = 180 * np.exp(-epochs/12) + 140 + np.random.randn(50) * 2
    axes[1].plot(epochs, recon_loss, 'g-', linewidth=2, label='Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss', fontsize=14, weight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # KL divergence
    kl_loss = 8 * (1 - np.exp(-epochs/8)) + np.random.randn(50) * 0.3
    axes[2].plot(epochs, kl_loss, 'r-', linewidth=2, label='KL Divergence')
    axes[2].set_title('KL Divergence', fontsize=14, weight='bold')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('KL', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('../assets/images/training_curves.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created training_curves.png")


def create_reparameterization_diagram():
    """Create reparameterization trick diagram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Without reparameterization
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Encoder box
    encoder = FancyBboxPatch((1, 2), 2, 2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#74b9ff', linewidth=2)
    ax.add_patch(encoder)
    ax.text(2, 3, 'Encoder\nφ', ha='center', va='center', fontsize=12, weight='bold')
    
    # Sampling (with X)
    sampling = FancyBboxPatch((4.5, 2), 2, 2, boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor='#ff7675', linewidth=2)
    ax.add_patch(sampling)
    ax.text(5.5, 3.2, 'z ~ q(z|x)', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(5.5, 2.6, '❌ Not differentiable', ha='center', va='center', fontsize=9)
    
    # Network
    network = FancyBboxPatch((7.5, 2), 2, 2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#74b9ff', linewidth=2)
    ax.add_patch(network)
    ax.text(8.5, 3, 'Network\nf(z)', ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrows
    ax.annotate('', xy=(4.5, 3), xytext=(3, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7.5, 3), xytext=(6.5, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'))
    
    ax.text(5, 5, 'Without Reparameterization', ha='center', fontsize=14, weight='bold')
    ax.text(5, 0.5, 'Cannot backpropagate through\nrandom sampling', ha='center', 
            fontsize=10, style='italic', color='red')
    
    # With reparameterization
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Encoder box
    encoder = FancyBboxPatch((0.5, 2.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#74b9ff', linewidth=2)
    ax.add_patch(encoder)
    ax.text(1.4, 3.25, 'Encoder\nμ, σ', ha='center', va='center', fontsize=11, weight='bold')
    
    # Epsilon (external randomness)
    epsilon = FancyBboxPatch((0.5, 0.5), 1.8, 1, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#fdcb6e', linewidth=2)
    ax.add_patch(epsilon)
    ax.text(1.4, 1, 'ε ~ N(0,I)', ha='center', va='center', fontsize=11, weight='bold')
    
    # Transform
    transform = FancyBboxPatch((3.5, 1.5), 2.5, 2, boxstyle="round,pad=0.1",
                              edgecolor='green', facecolor='#55efc4', linewidth=2)
    ax.add_patch(transform)
    ax.text(4.75, 2.7, 'Transform', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(4.75, 2.2, 'z = μ + σ⊙ε', ha='center', va='center', fontsize=10)
    
    # Network
    network = FancyBboxPatch((7.5, 2), 2, 2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#74b9ff', linewidth=2)
    ax.add_patch(network)
    ax.text(8.5, 3, 'Network\nf(z)', ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrows
    ax.annotate('', xy=(3.5, 3), xytext=(2.3, 3.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(3.8, 2), xytext=(2.3, 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(7.5, 2.5), xytext=(6, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Gradient flow annotation
    ax.annotate('', xy=(1.4, 5), xytext=(8.5, 5),
                arrowprops=dict(arrowstyle='<-', lw=2.5, color='blue'))
    ax.text(5, 5.5, '✓ Gradient flows through μ and σ', ha='center', 
            fontsize=10, color='blue', weight='bold')
    
    ax.text(5, 5, 'With Reparameterization Trick', ha='center', fontsize=14, weight='bold')
    ax.text(5, 0.3, 'Randomness externalized,\nfully differentiable!', ha='center', 
            fontsize=10, style='italic', color='green')
    
    plt.tight_layout()
    plt.savefig('../assets/images/reparameterization_trick.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created reparameterization_trick.png")


def create_elbo_decomposition():
    """Create ELBO decomposition visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Main equation at top
    ax.text(6, 9, 'log p_θ(x) = ELBO + KL Divergence', ha='center', va='center',
            fontsize=16, weight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # ELBO box
    elbo_box = FancyBboxPatch((1, 5.5), 4.5, 2.5, boxstyle="round,pad=0.15",
                             edgecolor='green', facecolor='#d5f4e6', linewidth=3)
    ax.add_patch(elbo_box)
    ax.text(3.25, 7.3, 'ELBO (Maximize)', ha='center', fontsize=13, weight='bold', color='green')
    ax.text(3.25, 6.5, '𝔼[log p_θ(x|z)]', ha='center', fontsize=11)
    ax.text(3.25, 6, '−', ha='center', fontsize=14)
    ax.text(3.25, 5.8, 'D_KL(q_φ(z|x) || p(z))', ha='center', fontsize=10)
    
    # KL box
    kl_box = FancyBboxPatch((6.5, 5.5), 4.5, 2.5, boxstyle="round,pad=0.15",
                           edgecolor='red', facecolor='#ffe6e6', linewidth=3)
    ax.add_patch(kl_box)
    ax.text(8.75, 7.3, 'KL Divergence (≥ 0)', ha='center', fontsize=13, weight='bold', color='red')
    ax.text(8.75, 6.5, 'D_KL(q_φ(z|x) || p_θ(z|x))', ha='center', fontsize=11)
    ax.text(8.75, 5.8, 'Gap between bound\nand true likelihood', ha='center', fontsize=9, style='italic')
    
    # Breakdown
    recon_box = FancyBboxPatch((0.5, 2.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                              edgecolor='blue', facecolor='#e3f2fd', linewidth=2)
    ax.add_patch(recon_box)
    ax.text(2.25, 3.8, 'Reconstruction Term', ha='center', fontsize=12, weight='bold', color='blue')
    ax.text(2.25, 3.2, '𝔼[log p_θ(x|z)]', ha='center', fontsize=11)
    ax.text(2.25, 2.7, '≈ -||x - x̂||²', ha='center', fontsize=10)
    
    reg_box = FancyBboxPatch((4.5, 2.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                            edgecolor='purple', facecolor='#f3e5f5', linewidth=2)
    ax.add_patch(reg_box)
    ax.text(6.25, 3.8, 'Regularization Term', ha='center', fontsize=12, weight='bold', color='purple')
    ax.text(6.25, 3.2, 'D_KL(q_φ(z|x) || p(z))', ha='center', fontsize=11)
    ax.text(6.25, 2.7, 'Closed form for Gaussians', ha='center', fontsize=9)
    
    tightness_box = FancyBboxPatch((8.5, 2.5), 3, 1.8, boxstyle="round,pad=0.1",
                                  edgecolor='orange', facecolor='#fff3e0', linewidth=2)
    ax.add_patch(tightness_box)
    ax.text(10, 3.8, 'Tightness of Bound', ha='center', fontsize=12, weight='bold', color='orange')
    ax.text(10, 3.2, 'How close q_φ is', ha='center', fontsize=10)
    ax.text(10, 2.8, 'to true posterior', ha='center', fontsize=10)
    
    # Arrows
    ax.annotate('', xy=(3.25, 5.3), xytext=(2.25, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(3.25, 5.3), xytext=(6.25, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    
    # Bottom note
    ax.text(6, 0.8, 'Maximizing ELBO ⟹ Maximizing log p_θ(x) and Minimizing KL',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.title('Evidence Lower Bound (ELBO) Decomposition', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('../assets/images/elbo_decomposition.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created elbo_decomposition.png")


if __name__ == '__main__':
    print("Generating VAE tutorial visualizations...")
    print("-" * 50)
    
    create_autoencoder_architecture()
    create_vae_architecture()
    create_latent_space_comparison()
    create_training_curves()
    create_reparameterization_diagram()
    create_elbo_decomposition()
    
    print("-" * 50)
    print("✓ All visualizations created successfully!")
    print("Images saved to: ../assets/images/")

