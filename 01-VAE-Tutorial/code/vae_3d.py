"""
3D Variational Autoencoder (VAE) Implementation
================================================

This script extends the 2D VAE to 3 dimensions, providing increased
representational capacity and better disentanglement of latent factors.

Author: Maroua Oukrid
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import base components from 2D VAE
import sys
sys.path.append('.')
from vae_2d import Sampling, VAE, load_mnist_data


# ============================================
# Model Building Functions (3D)
# ============================================

def build_encoder_3d(latent_dim=3, input_shape=(28, 28, 1)):
    """
    Build the encoder network for 3D latent space.
    
    Args:
        latent_dim: Dimensionality of the latent space (3)
        input_shape: Shape of input images
    
    Returns:
        Keras Model for the encoder
    """
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(512, activation='relu', name='encoder_dense_1')(x)
    x = layers.Dense(256, activation='relu', name='encoder_dense_2')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder_3d')
    return encoder


def build_decoder_3d(latent_dim=3):
    """
    Build the decoder network for 3D latent space.
    
    Args:
        latent_dim: Dimensionality of the latent space (3)
    
    Returns:
        Keras Model for the decoder
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(256, activation='relu', name='decoder_dense_1')(latent_inputs)
    x = layers.Dense(512, activation='relu', name='decoder_dense_2')(x)
    x = layers.Dense(28 * 28, activation='sigmoid', name='decoder_dense_3')(x)
    decoder_outputs = layers.Reshape((28, 28, 1))(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder_3d')
    return decoder


# ============================================
# 3D Visualization Functions
# ============================================

def plot_3d_latent_space(encoder, x_test, y_test, save_path='../assets/Clusters3D.png'):
    """
    Plot the 3D latent space colored by digit class.
    
    Args:
        encoder: Trained encoder model
        x_test: Test images
        y_test: Test labels
        save_path: Path to save the figure
    """
    # Encode test data
    z_mean, _, _ = encoder.predict(x_test, verbose=0)
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2],
                        c=y_test, cmap='tab10', 
                        alpha=0.6, s=10, edgecolors='none')
    
    ax.set_xlabel('z[0]', fontsize=12, labelpad=10)
    ax.set_ylabel('z[1]', fontsize=12, labelpad=10)
    ax.set_zlabel('z[2]', fontsize=12, labelpad=10)
    ax.set_title('3D Latent Space Clustering', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Digit Class', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D latent space to {save_path}")
    plt.close()


def plot_2d_projections(encoder, x_test, y_test, save_path='../assets/2D3DClusters.png'):
    """
    Plot pairwise 2D projections of the 3D latent space.
    
    Args:
        encoder: Trained encoder model
        x_test: Test images
        y_test: Test labels
        save_path: Path to save the figure
    """
    # Encode test data
    z_mean, _, _ = encoder.predict(x_test, verbose=0)
    
    # Create subplots for three 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    projections = [
        (0, 1, 'z[0]', 'z[1]'),
        (1, 2, 'z[1]', 'z[2]'),
        (0, 2, 'z[0]', 'z[2]')
    ]
    
    for ax, (dim1, dim2, label1, label2) in zip(axes, projections):
        scatter = ax.scatter(z_mean[:, dim1], z_mean[:, dim2],
                            c=y_test, cmap='tab10',
                            alpha=0.6, s=10, edgecolors='none')
        ax.set_xlabel(label1, fontsize=12)
        ax.set_ylabel(label2, fontsize=12)
        ax.set_title(f'{label1} vs {label2}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Add shared colorbar
    fig.colorbar(scatter, ax=axes, label='Digit Class', pad=0.02)
    
    plt.suptitle('Pairwise 2D Projections of 3D Latent Space', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D projections to {save_path}")
    plt.close()


def plot_3d_slices(decoder, z2_values=[-1.0, 0.0, 1.0], n=15, digit_size=28):
    """
    Generate 2D manifold slices at different z2 values.
    
    Args:
        decoder: Trained decoder model
        z2_values: List of z2 values to slice at
        n: Grid size for each slice
        digit_size: Size of each digit
    """
    for z2 in z2_values:
        # Sample points in z0-z1 plane
        grid_x = np.linspace(-3, 3, n)
        grid_y = np.linspace(-3, 3, n)[::-1]
        
        figure = np.zeros((digit_size * n, digit_size * n))
        
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi, z2]])
                x_decoded = decoder.predict(z_sample, verbose=0)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
        
        plt.figure(figsize=(12, 12))
        plt.imshow(figure, cmap='viridis')
        plt.title(f'2D Manifold Slice at z[2] = {z2:.1f}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('z[0]', fontsize=14)
        plt.ylabel('z[1]', fontsize=14)
        plt.colorbar(label='Pixel Intensity')
        plt.tight_layout()
        
        # Save with appropriate filename
        save_path = f'../assets/3DLatent{int((z2 + 1) * 1.5 + 1)}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved slice at z2={z2} to {save_path}")
        plt.close()


def analyze_latent_dimensions(encoder, x_test, y_test):
    """
    Analyze variance captured by each latent dimension.
    
    Args:
        encoder: Trained encoder model
        x_test: Test images
        y_test: Test labels
    """
    # Encode test data
    z_mean, _, _ = encoder.predict(x_test, verbose=0)
    
    # Calculate variance of each dimension
    variances = np.var(z_mean, axis=0)
    
    print("\n" + "=" * 60)
    print("Latent Dimension Analysis")
    print("=" * 60)
    
    for i, var in enumerate(variances):
        print(f"Dimension z[{i}]: variance = {var:.4f}")
    
    print(f"\nTotal variance: {np.sum(variances):.4f}")
    print(f"Variance explained by each dimension:")
    
    for i, var in enumerate(variances):
        percentage = (var / np.sum(variances)) * 100
        print(f"  z[{i}]: {percentage:.2f}%")
    
    # Plot variance distribution
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(variances)), variances, color='steelblue', alpha=0.7)
    plt.xlabel('Latent Dimension', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.title('Variance Captured by Each Latent Dimension', 
             fontsize=14, fontweight='bold')
    plt.xticks(range(len(variances)), [f'z[{i}]' for i in range(len(variances))])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../assets/latent_variance_3d.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved variance plot to ../assets/latent_variance_3d.png")
    plt.close()


# ============================================
# Training
# ============================================

def train_3d_vae(latent_dim=3, epochs=30, batch_size=128):
    """
    Complete training pipeline for 3D VAE.
    
    Args:
        latent_dim: Dimensionality of latent space (3)
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained VAE model, encoder, decoder, and test data
    """
    print("=" * 60)
    print("Training 3D Variational Autoencoder")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Build models
    print("\n[2/5] Building 3D encoder and decoder...")
    encoder = build_encoder_3d(latent_dim=latent_dim)
    decoder = build_decoder_3d(latent_dim=latent_dim)
    
    print("\nEncoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    
    # Build VAE
    print("\n[3/5] Creating 3D VAE model...")
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    # Train
    print(f"\n[4/5] Training for {epochs} epochs...")
    history = vae.fit(
        x_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    # Generate visualizations
    print("\n[5/5] Generating 3D visualizations...")
    plot_3d_latent_space(encoder, x_test, y_test)
    plot_2d_projections(encoder, x_test, y_test)
    plot_3d_slices(decoder)
    analyze_latent_dimensions(encoder, x_test, y_test)
    
    print("\n" + "=" * 60)
    print("3D VAE Training Complete!")
    print("=" * 60)
    
    return vae, encoder, decoder, (x_test, y_test), history


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the 3D model
    vae, encoder, decoder, (x_test, y_test), history = train_3d_vae(
        latent_dim=3,
        epochs=30,
        batch_size=128
    )
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Total Loss: {history.history['total_loss'][-1]:.2f}")
    print(f"  Reconstruction Loss: {history.history['reconstruction_loss'][-1]:.2f}")
    print(f"  KL Loss: {history.history['kl_loss'][-1]:.2f}")
    
    # Compare with 2D results
    print("\nComparison with 2D VAE:")
    print("  2D Total Loss: ~165")
    print(f"  3D Total Loss: {history.history['total_loss'][-1]:.2f}")
    
    improvement = 165 - history.history['total_loss'][-1]
    print(f"  Improvement: {improvement:.2f}")
    
    # Save models
    print("\nSaving models...")
    encoder.save('models/encoder_3d.h5')
    decoder.save('models/decoder_3d.h5')
    vae.save_weights('models/vae_3d_weights.h5')
    print("3D models saved successfully!")

