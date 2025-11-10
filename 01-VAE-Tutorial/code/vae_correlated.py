"""
VAE with Non-Equal Covariance Prior
====================================

This script implements a VAE with a correlated Gaussian prior instead of
the standard isotropic prior. This allows modeling of dependent latent factors.

Author: Maroua Oukrid
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import base components
import sys
sys.path.append('.')
from vae_2d import load_mnist_data


# ============================================
# Correlated Sampling Layer
# ============================================

class CorrelatedSampling(layers.Layer):
    """
    Implements correlated sampling using Cholesky decomposition:
    z = μ + L * ε, where Σ = L * L^T
    """
    def __init__(self, covariance_matrix, **kwargs):
        super(CorrelatedSampling, self).__init__(**kwargs)
        # Compute Cholesky decomposition
        self.L = tf.constant(np.linalg.cholesky(covariance_matrix), dtype=tf.float32)
        self.covariance_matrix = tf.constant(covariance_matrix, dtype=tf.float32)
        self.covariance_inv = tf.constant(np.linalg.inv(covariance_matrix), dtype=tf.float32)
        self.covariance_det = tf.constant(np.linalg.det(covariance_matrix), dtype=tf.float32)
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample epsilon from standard normal
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Apply Cholesky transformation: z = μ + L * ε
        z = z_mean + tf.matmul(epsilon, self.L, transpose_b=True)
        
        return z


# ============================================
# Correlated VAE Model
# ============================================

class CorrelatedVAE(keras.Model):
    """
    VAE with correlated Gaussian prior.
    """
    def __init__(self, encoder, decoder, covariance_matrix, **kwargs):
        super(CorrelatedVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        # Store covariance information
        self.covariance_matrix = tf.constant(covariance_matrix, dtype=tf.float32)
        self.covariance_inv = tf.constant(np.linalg.inv(covariance_matrix), dtype=tf.float32)
        self.covariance_det = tf.constant(np.linalg.det(covariance_matrix), dtype=tf.float32)
        self.latent_dim = covariance_matrix.shape[0]
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def compute_kl_divergence(self, z_mean, z_log_var):
        """
        Compute KL divergence with non-equal covariance prior:
        KL(q||p) = 0.5 * [tr(Σ^-1 * diag(σ²)) + μ^T Σ^-1 μ - k + log(det(Σ) / Π σᵢ²)]
        """
        # Variance from log variance
        z_var = tf.exp(z_log_var)
        
        # Term 1: tr(Σ^-1 * diag(σ²))
        trace_term = tf.reduce_sum(
            tf.linalg.diag_part(self.covariance_inv) * z_var,
            axis=1
        )
        
        # Term 2: μ^T Σ^-1 μ
        mahalanobis_term = tf.reduce_sum(
            z_mean * tf.matmul(z_mean, self.covariance_inv),
            axis=1
        )
        
        # Term 3: -k (negative latent dimension)
        dim_term = -tf.cast(self.latent_dim, tf.float32)
        
        # Term 4: log(det(Σ) / Π σᵢ²)
        log_det_term = tf.math.log(self.covariance_det) - tf.reduce_sum(z_log_var, axis=1)
        
        # Total KL divergence
        kl_loss = 0.5 * (trace_term + mahalanobis_term + dim_term + log_det_term)
        
        return tf.reduce_mean(kl_loss)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # Correlated KL divergence
            kl_loss = self.compute_kl_divergence(z_mean, z_log_var)
            
            # Total loss
            total_loss = reconstruction_loss + kl_loss
        
        # Update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Track metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'total_loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


# ============================================
# Model Building with Correlated Sampling
# ============================================

def build_correlated_encoder(latent_dim=2, covariance_matrix=None, input_shape=(28, 28, 1)):
    """
    Build encoder with correlated sampling.
    
    Args:
        latent_dim: Dimensionality of latent space
        covariance_matrix: Covariance matrix for the prior
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
    
    # Use correlated sampling
    z = CorrelatedSampling(covariance_matrix)([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder_correlated')
    return encoder


def build_correlated_decoder(latent_dim=2):
    """Build decoder (same as standard VAE)."""
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(256, activation='relu', name='decoder_dense_1')(latent_inputs)
    x = layers.Dense(512, activation='relu', name='decoder_dense_2')(x)
    x = layers.Dense(28 * 28, activation='sigmoid', name='decoder_dense_3')(x)
    decoder_outputs = layers.Reshape((28, 28, 1))(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder_correlated')
    return decoder


# ============================================
# Visualization Functions
# ============================================

def plot_correlated_latent_space(encoder, x_test, y_test, 
                                  covariance_matrix, 
                                  save_path='../assets/Screenshot 2025-11-07 at 14.41.06.png'):
    """
    Plot latent space with covariance ellipse overlay.
    
    Args:
        encoder: Trained encoder
        x_test: Test images
        y_test: Test labels
        covariance_matrix: Prior covariance matrix
        save_path: Path to save figure
    """
    # Encode test data
    z_mean, _, _ = encoder.predict(x_test, verbose=0)
    
    plt.figure(figsize=(10, 8))
    
    # Plot scatter
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                         c=y_test, cmap='tab10', 
                         alpha=0.6, s=10, edgecolors='none')
    
    # Overlay covariance ellipse
    from matplotlib.patches import Ellipse
    
    # Eigendecomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Plot 1, 2, 3 standard deviation ellipses
    for n_std in [1, 2, 3]:
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ell = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                     edgecolor='red', facecolor='none', linewidth=2, 
                     linestyle='--', alpha=0.5, label=f'{n_std}σ' if n_std == 1 else '')
        plt.gca().add_patch(ell)
    
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('z[0]', fontsize=14)
    plt.ylabel('z[1]', fontsize=14)
    plt.title('Correlated Latent Space with Prior Covariance', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlated latent space to {save_path}")
    plt.close()


def plot_correlated_manifold(decoder, covariance_matrix, n=20, digit_size=28,
                            save_path='../assets/KLAdapted.png'):
    """
    Generate manifold grid accounting for covariance structure.
    
    Args:
        decoder: Trained decoder
        covariance_matrix: Prior covariance matrix
        n: Grid size
        digit_size: Size of each digit
        save_path: Path to save figure
    """
    # Generate grid in principal component space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Get transformation matrix (principal components)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Transform grid point according to covariance structure
            z_pc = np.array([xi, yi])
            z_sample = np.dot(eigenvectors, z_pc * np.sqrt(eigenvalues)).reshape(1, -1)
            
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(12, 12))
    plt.imshow(figure, cmap='viridis')
    plt.title('Latent Manifold with Correlated Prior', fontsize=16, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.colorbar(label='Pixel Intensity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlated manifold to {save_path}")
    plt.close()


def visualize_covariance_matrix(covariance_matrix, save_path='../assets/covariance_matrix.png'):
    """
    Visualize the covariance matrix.
    
    Args:
        covariance_matrix: Covariance matrix to visualize
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(covariance_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Covariance')
    
    # Add text annotations
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            plt.text(j, i, f'{covariance_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.title('Prior Covariance Matrix Σ', fontsize=16, fontweight='bold')
    plt.xlabel('Latent Dimension', fontsize=12)
    plt.ylabel('Latent Dimension', fontsize=12)
    plt.xticks([0, 1], ['z[0]', 'z[1]'])
    plt.yticks([0, 1], ['z[0]', 'z[1]'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved covariance matrix visualization to {save_path}")
    plt.close()


# ============================================
# Training
# ============================================

def train_correlated_vae(covariance_matrix, epochs=30, batch_size=128):
    """
    Train VAE with correlated prior.
    
    Args:
        covariance_matrix: Prior covariance matrix
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Trained model components and history
    """
    print("=" * 60)
    print("Training VAE with Correlated Prior")
    print("=" * 60)
    
    latent_dim = covariance_matrix.shape[0]
    
    print(f"\nCovariance Matrix:")
    print(covariance_matrix)
    print(f"\nCorrelation coefficient: {covariance_matrix[0, 1]:.2f}")
    
    # Visualize covariance matrix
    visualize_covariance_matrix(covariance_matrix)
    
    # Load data
    print("\n[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Build models
    print("\n[2/5] Building correlated encoder and decoder...")
    encoder = build_correlated_encoder(latent_dim=latent_dim, 
                                      covariance_matrix=covariance_matrix)
    decoder = build_correlated_decoder(latent_dim=latent_dim)
    
    # Build VAE
    print("\n[3/5] Creating correlated VAE model...")
    vae = CorrelatedVAE(encoder, decoder, covariance_matrix)
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
    print("\n[5/5] Generating visualizations...")
    plot_correlated_latent_space(encoder, x_test, y_test, covariance_matrix)
    plot_correlated_manifold(decoder, covariance_matrix)
    
    print("\n" + "=" * 60)
    print("Correlated VAE Training Complete!")
    print("=" * 60)
    
    return vae, encoder, decoder, (x_test, y_test), history


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define correlated covariance matrix
    covariance_matrix = np.array([
        [1.0, 0.4],
        [0.4, 0.5]
    ])
    
    print("Correlated VAE Experiment")
    print("=" * 60)
    print("Prior covariance matrix:")
    print(covariance_matrix)
    print("\nThis introduces positive correlation between z[0] and z[1]")
    print("=" * 60)
    
    # Train the model
    vae, encoder, decoder, (x_test, y_test), history = train_correlated_vae(
        covariance_matrix=covariance_matrix,
        epochs=30,
        batch_size=128
    )
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Total Loss: {history.history['total_loss'][-1]:.2f}")
    print(f"  Reconstruction Loss: {history.history['reconstruction_loss'][-1]:.2f}")
    print(f"  KL Loss: {history.history['kl_loss'][-1]:.2f}")
    
    # Save models
    print("\nSaving models...")
    encoder.save('models/encoder_correlated.h5')
    decoder.save('models/decoder_correlated.h5')
    vae.save_weights('models/vae_correlated_weights.h5')
    print("Correlated VAE models saved successfully!")

