"""
Mathematical Visualizations for VAE Tutorial
=============================================

This script generates visualizations explaining VAE mathematical concepts:
- ELBO decomposition
- KL divergence behavior
- Reparameterization trick
- Loss landscape
- Prior vs Posterior distributions

Author: Maroua Oukrid
Date: November 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
np.random.seed(42)


# ============================================
# 1. ELBO Decomposition Visualization
# ============================================

def plot_elbo_decomposition(save_path='../assets/elbo_decomposition.png'):
    """
    Visualize the relationship between log p(x), ELBO, and KL divergence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: ELBO as lower bound
    epochs = np.arange(0, 30)
    log_px = np.full_like(epochs, 170.0, dtype=float)  # True (unknown) log-likelihood
    elbo = 200 - 35 * np.exp(-epochs/5)  # ELBO converging to log p(x)
    kl_gap = log_px - elbo
    
    ax1.plot(epochs, log_px, 'k--', linewidth=2, label=r'$\log p_\theta(x)$ (unknown)')
    ax1.plot(epochs, elbo, 'b-', linewidth=2, label=r'ELBO $\mathcal{L}(\theta,\phi;x)$')
    ax1.fill_between(epochs, elbo, log_px, alpha=0.3, color='red', 
                      label=r'Gap: $D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$')
    
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('ELBO as Lower Bound on Log-Likelihood', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Component breakdown
    reconstruction = 161 - 10 * (1 - np.exp(-epochs/8))
    kl_loss = 39 * np.exp(-epochs/5) + 3.9
    total_loss = reconstruction + kl_loss
    
    ax2.plot(epochs, total_loss, 'k-', linewidth=2, label='Total Loss', marker='o', markersize=4)
    ax2.plot(epochs, reconstruction, 'b-', linewidth=2, label='Reconstruction Loss')
    ax2.plot(epochs, kl_loss, 'r-', linewidth=2, label='KL Divergence')
    
    ax2.set_xlabel('Training Epoch', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.set_title('VAE Loss Components During Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ELBO decomposition to {save_path}")
    plt.close()


# ============================================
# 2. KL Divergence Visualization
# ============================================

def plot_kl_divergence(save_path='../assets/kl_divergence_behavior.png'):
    """
    Visualize KL divergence between Gaussian distributions.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    x = np.linspace(-5, 5, 1000)
    
    # Plot 1: Effect of mean shift
    prior = norm.pdf(x, 0, 1)
    for mu in [0, 0.5, 1.0, 2.0]:
        posterior = norm.pdf(x, mu, 1)
        kl = 0.5 * mu**2
        ax1.plot(x, posterior, label=f'$\mu={mu:.1f}$, KL={kl:.2f}', linewidth=2)
    
    ax1.plot(x, prior, 'k--', linewidth=3, label='Prior $p(z)$', alpha=0.7)
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('KL Divergence: Effect of Mean Shift', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Effect of variance
    for sigma in [0.5, 0.7, 1.0, 1.5]:
        posterior = norm.pdf(x, 0, sigma)
        kl = 0.5 * (sigma**2 - 1 - np.log(sigma**2))
        ax2.plot(x, posterior, label=f'$\sigma={sigma:.1f}$, KL={kl:.2f}', linewidth=2)
    
    ax2.plot(x, prior, 'k--', linewidth=3, label='Prior $p(z)$', alpha=0.7)
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('KL Divergence: Effect of Variance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: KL as function of mu (sigma fixed)
    mu_vals = np.linspace(-3, 3, 100)
    kl_mu = 0.5 * mu_vals**2
    ax3.plot(mu_vals, kl_mu, 'b-', linewidth=2)
    ax3.fill_between(mu_vals, 0, kl_mu, alpha=0.3)
    ax3.set_xlabel(r'$\mu$ (mean)', fontsize=12)
    ax3.set_ylabel(r'$D_{KL}$', fontsize=12)
    ax3.set_title(r'KL Divergence vs Mean (fixed $\sigma=1$)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: KL as function of sigma (mu fixed)
    sigma_vals = np.linspace(0.1, 3, 100)
    kl_sigma = 0.5 * (sigma_vals**2 - 1 - np.log(sigma_vals**2))
    ax4.plot(sigma_vals, kl_sigma, 'r-', linewidth=2)
    ax4.fill_between(sigma_vals, 0, kl_sigma, alpha=0.3)
    ax4.set_xlabel(r'$\sigma$ (std dev)', fontsize=12)
    ax4.set_ylabel(r'$D_{KL}$', fontsize=12)
    ax4.set_title(r'KL Divergence vs Std Dev (fixed $\mu=0$)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=1, color='k', linestyle='--', alpha=0.5, label='$\sigma=1$ (minimum)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved KL divergence visualization to {save_path}")
    plt.close()


# ============================================
# 3. Reparameterization Trick Visualization
# ============================================

def plot_reparameterization_trick(save_path='../assets/reparameterization_trick.png'):
    """
    Visualize the reparameterization trick.
    """
    fig = plt.figure(figsize=(16, 5))
    
    # Parameters
    mu = 2.0
    sigma = 1.5
    n_samples = 1000
    
    # Standard sampling (non-differentiable)
    ax1 = plt.subplot(131)
    epsilon = np.random.randn(n_samples)
    ax1.hist(epsilon, bins=50, alpha=0.7, color='gray', edgecolor='black', density=True)
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=3, label=r'$\mathcal{N}(0,1)$')
    ax1.set_xlabel(r'$\epsilon$', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Step 1: Sample Noise\n'+r'$\epsilon \sim \mathcal{N}(0, 1)$', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Transformation
    ax2 = plt.subplot(132)
    ax2.arrow(0, 0.5, 0.8, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=3)
    ax2.text(0.4, 0.65, r'$z = \mu + \sigma \cdot \epsilon$', fontsize=16, ha='center', fontweight='bold')
    ax2.text(0.4, 0.35, 'Deterministic\nTransformation', fontsize=12, ha='center', style='italic')
    ax2.text(0.4, 0.15, r'Gradients flow through $\mu, \sigma$', fontsize=11, ha='center', color='green')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Reparameterized sampling (differentiable)
    ax3 = plt.subplot(133)
    z = mu + sigma * epsilon
    ax3.hist(z, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    x = np.linspace(-4, 8, 100)
    ax3.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=3, 
             label=r'$\mathcal{N}(\mu=%.1f, \sigma=%.1f)$' % (mu, sigma))
    ax3.set_xlabel('z', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Step 2: Reparameterized Samples\n'+r'$z \sim q_\phi(z|x)$', 
                  fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reparameterization trick to {save_path}")
    plt.close()


# ============================================
# 4. Prior vs Posterior Visualization
# ============================================

def plot_prior_posterior(save_path='../assets/prior_posterior_comparison.png'):
    """
    Visualize prior vs learned posterior distributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Create grid for 2D visualization
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Prior (isotropic Gaussian)
    prior_mean = [0, 0]
    prior_cov = [[1, 0], [0, 1]]
    prior_rv = multivariate_normal(prior_mean, prior_cov)
    
    # Different posterior examples
    posteriors = [
        {'mean': [0, 0], 'cov': [[1, 0], [0, 1]], 'title': 'Well-regularized\n(KL ≈ 0)'},
        {'mean': [1, 1], 'cov': [[1, 0], [0, 1]], 'title': 'Mean shift\n(KL = 1.0)'},
        {'mean': [0, 0], 'cov': [[2, 0], [0, 2]], 'title': 'High variance\n(KL = 1.39)'},
        {'mean': [2, -1], 'cov': [[0.5, 0], [0, 0.5]], 'title': 'Both effects\n(KL = 2.79)'},
        {'mean': [0, 0], 'cov': [[1, 0.5], [0.5, 1]], 'title': 'Correlated\n(KL = 0.14)'},
    ]
    
    # Plot prior
    ax = axes[0, 0]
    Z_prior = prior_rv.pdf(pos)
    contour = ax.contourf(X, Y, Z_prior, levels=15, cmap='Blues')
    ax.contour(X, Y, Z_prior, levels=10, colors='navy', alpha=0.3, linewidths=0.5)
    ax.set_title('Prior $p(z)$\n'+r'$\mathcal{N}(0, I)$', fontsize=12, fontweight='bold')
    ax.set_xlabel('$z_0$', fontsize=11)
    ax.set_ylabel('$z_1$', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    # Plot posteriors
    for idx, post in enumerate(posteriors):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes[row, col]
        
        post_rv = multivariate_normal(post['mean'], post['cov'])
        Z_post = post_rv.pdf(pos)
        
        # Plot both prior (in background) and posterior
        ax.contour(X, Y, Z_prior, levels=10, colors='gray', alpha=0.3, linewidths=0.5, linestyles='--')
        contour = ax.contourf(X, Y, Z_post, levels=15, cmap='Reds')
        ax.contour(X, Y, Z_post, levels=10, colors='darkred', alpha=0.4, linewidths=0.5)
        
        ax.set_title(f'Posterior $q_\\phi(z|x)$\n{post["title"]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('$z_0$', fontsize=11)
        ax.set_ylabel('$z_1$', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved prior vs posterior comparison to {save_path}")
    plt.close()


# ============================================
# 5. Beta-VAE Trade-off Visualization
# ============================================

def plot_beta_vae_tradeoff(save_path='../assets/beta_vae_tradeoff.png'):
    """
    Visualize the reconstruction-KL trade-off with different beta values.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    betas = [0.1, 0.5, 1.0, 2.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))
    
    epochs = np.arange(0, 30)
    
    for beta, color in zip(betas, colors):
        # Simulate different trajectories based on beta
        recon = 200 - 40 * np.exp(-epochs/5) + beta * 2
        kl = (40 - beta * 8) * np.exp(-epochs/6) + 5 - beta * 0.5
        total = recon + beta * kl
        
        ax1.plot(epochs, total, color=color, linewidth=2, label=f'$\\beta={beta}$')
    
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title(r'Effect of $\beta$ on Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Trade-off plot
    beta_range = np.linspace(0.1, 5, 50)
    recon_final = 161 + beta_range * 1.5
    kl_final = 12 - 1.5 * beta_range
    kl_final = np.maximum(kl_final, 1)  # Floor at 1
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(beta_range, recon_final, 'b-', linewidth=3, label='Reconstruction Loss')
    line2 = ax2_twin.plot(beta_range, kl_final, 'r-', linewidth=3, label='KL Divergence')
    
    ax2.set_xlabel(r'$\beta$ (KL weight)', fontsize=12)
    ax2.set_ylabel('Reconstruction Loss', fontsize=12, color='b')
    ax2_twin.set_ylabel('KL Divergence', fontsize=12, color='r')
    ax2.set_title(r'Reconstruction-KL Trade-off', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line at beta=1
    ax2.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.7, label=r'Standard VAE ($\beta=1$)')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved beta-VAE trade-off to {save_path}")
    plt.close()


# ============================================
# 6. Loss Landscape Visualization
# ============================================

def plot_loss_landscape(save_path='../assets/loss_landscape.png'):
    """
    Visualize VAE loss landscape in parameter space.
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Create parameter grid
    mu_range = np.linspace(-3, 3, 100)
    sigma_range = np.linspace(0.1, 3, 100)
    Mu, Sigma = np.meshgrid(mu_range, sigma_range)
    
    # Compute losses
    recon_loss = 160 + 5 * Mu**2  # Quadratic in mu
    kl_loss = 0.5 * (Sigma**2 + Mu**2 - 1 - np.log(Sigma**2))
    total_loss = recon_loss + kl_loss
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(Mu, Sigma, total_loss, cmap='viridis', alpha=0.8)
    ax1.set_xlabel(r'$\mu$', fontsize=12)
    ax1.set_ylabel(r'$\sigma$', fontsize=12)
    ax1.set_zlabel('Total Loss', fontsize=12)
    ax1.set_title('VAE Loss Landscape', fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(Mu, Sigma, total_loss, levels=20, cmap='viridis')
    ax2.contour(Mu, Sigma, total_loss, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    ax2.plot(0, 1, 'r*', markersize=20, label='Optimal point\n'+r'$(\mu=0, \sigma=1)$')
    ax2.set_xlabel(r'$\mu$', fontsize=12)
    ax2.set_ylabel(r'$\sigma$', fontsize=12)
    ax2.set_title('Loss Contours', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    fig.colorbar(contour, ax=ax2, label='Total Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss landscape to {save_path}")
    plt.close()


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Mathematical Visualizations for VAE Tutorial")
    print("=" * 60)
    
    print("\n[1/6] Creating ELBO decomposition...")
    plot_elbo_decomposition()
    
    print("\n[2/6] Creating KL divergence behavior...")
    plot_kl_divergence()
    
    print("\n[3/6] Creating reparameterization trick...")
    plot_reparameterization_trick()
    
    print("\n[4/6] Creating prior vs posterior comparison...")
    plot_prior_posterior()
    
    print("\n[5/6] Creating beta-VAE trade-off...")
    plot_beta_vae_tradeoff()
    
    print("\n[6/6] Creating loss landscape...")
    plot_loss_landscape()
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - elbo_decomposition.png")
    print("  - kl_divergence_behavior.png")
    print("  - reparameterization_trick.png")
    print("  - prior_posterior_comparison.png")
    print("  - beta_vae_tradeoff.png")
    print("  - loss_landscape.png")

