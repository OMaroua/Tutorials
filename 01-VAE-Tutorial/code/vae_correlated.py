"""
Correlated 2D Variational Autoencoder (VAE) Implementation
===========================================================

This script implements a VAE with a correlated 2D latent space using a
custom covariance matrix. This allows us to explore how correlation in
the prior affects the learned latent representation.

Author: Maroua Oukrid
Date: November 2024
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
import matplotlib.pyplot as plt


# ============================================
# Custom Sampling Layer with Covariance
# ============================================

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z with custom covariance."""
    
    def __init__(self, cov=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.Generator.from_seed(1337)
        
        if cov is not None:
            # Compute Cholesky factor of covariance matrix
            cov = np.array(cov, dtype=np.float32)
            self.L = tf.constant(np.linalg.cholesky(cov), dtype=tf.float32)
        else:
            self.L = None

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        
        if self.L is not None:
            epsilon = tf.matmul(epsilon, self.L)   # apply correlation
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ============================================
# Build Encoder (Convolutional)
# ============================================

def build_encoder(latent_dim=2, cov_mat=None):
    """
    Build convolutional encoder network for correlated 2D latent space.
    
    Args:
        latent_dim: Dimensionality of the latent space (default: 2)
        cov_mat: Covariance matrix for the prior (default: None, uses identity)
    
    Returns:
        Keras Model for the encoder
    """
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(cov=cov_mat)([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


# ============================================
# Build Decoder (Convolutional Transpose)
# ============================================

def build_decoder(latent_dim=2):
    """
    Build convolutional transpose decoder network.
    
    Args:
        latent_dim: Dimensionality of the latent space (default: 2)
    
    Returns:
        Keras Model for the decoder
    """
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


# ============================================
# VAE Model with Custom Covariance KL Divergence
# ============================================

class VAE(keras.Model):
    """
    Variational Autoencoder with custom covariance prior.
    
    The KL divergence is computed for N(μ, diag(σ²)) vs N(0, Σ) where Σ is
    a custom covariance matrix.
    """
    
    def __init__(self, encoder, decoder, cov_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        # Store prior covariance
        if cov_matrix is None:
            cov_matrix = np.eye(2)   # default to isotropic
        
        self.Sigma = tf.constant(cov_matrix, dtype=tf.float32)
        self.Sigma_inv = tf.linalg.inv(self.Sigma)
        self.log_det_Sigma = tf.math.log(tf.linalg.det(self.Sigma) + 1e-8)
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # ----- Reconstruction loss -----
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            
            # ----- KL divergence for N(0, Σ) -----
            # KL(q||p) = 0.5 * [tr(Σ⁻¹·diag(σ²)) + μᵀΣ⁻¹μ - k + log|Σ| - Σlog(σ²)]
            
            sigma_sq = tf.exp(z_log_var)   # posterior variances
            k = tf.cast(tf.shape(z_mean)[1], tf.float32)  # latent dim (2 here)
            
            # trace(Sigma^-1 * diag(sigma^2))
            trace_term = tf.reduce_sum(
                tf.linalg.diag_part(self.Sigma_inv) * sigma_sq, axis=1
            )
            
            # mu^T * Sigma^-1 * mu
            mu_term = tf.einsum('bi,ij,bj->b', z_mean, self.Sigma_inv, z_mean)
            
            # log(det(Sigma)) - sum(log(sigma^2))
            logdet_term = self.log_det_Sigma - tf.reduce_sum(z_log_var, axis=1)
            
            kl_loss = 0.5 * (trace_term + mu_term - k + logdet_term)
            kl_loss = tf.reduce_mean(kl_loss)
            
            # ----- Total loss -----
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# ============================================
# Visualization Functions
# ============================================

def plot_latent_space(vae, n=30, figsize=15):
    """
    Display a n×n 2D manifold of digits.
    
    Args:
        vae: Trained VAE model
        n: Grid size (number of samples per dimension)
        figsize: Figure size
    """
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Correlated 2D Latent Manifold")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('../assets/latent2Dcorrelated.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_label_clusters(vae, data, labels):
    """
    Display a 2D plot of the digit classes in the latent space.
    
    Args:
        vae: Trained VAE model
        data: Input images
        labels: Corresponding labels
    """
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
    plt.colorbar(label='Digit Class')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Correlated 2D Latent Space Clustering")
    plt.grid(True, alpha=0.3)
    plt.savefig('../assets/Clusters2Dcorrelated.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================
# Training Pipeline
# ============================================

def train_correlated_vae(latent_dim=2, cov_matrix=None, epochs=30, batch_size=128):
    """
    Complete training pipeline for correlated 2D VAE.
    
    Args:
        latent_dim: Dimensionality of latent space (default: 2)
        cov_matrix: Covariance matrix for prior (default: [[1.0, 0.4], [0.4, 0.5]])
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained VAE model, encoder, and decoder
    """
    if cov_matrix is None:
        cov_matrix = [[1.0, 0.4],
                      [0.4, 0.5]]
    
    print("=" * 60)
    print("Training Correlated 2D Variational Autoencoder")
    print("=" * 60)
    print(f"\nPrior Covariance Matrix:")
    print(np.array(cov_matrix))
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    
    # Build models
    print("Building encoder and decoder...")
    encoder = build_encoder(latent_dim=latent_dim, cov_mat=cov_matrix)
    decoder = build_decoder(latent_dim=latent_dim)
    
    print("\nEncoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    
    # Create and compile VAE with custom covariance
    vae = VAE(encoder, decoder, cov_matrix=cov_matrix)
    vae.compile(optimizer=keras.optimizers.Adam())
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = vae.fit(mnist_digits, epochs=epochs, batch_size=batch_size)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return vae, encoder, decoder, history


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Define covariance matrix with correlation
    cov_mat = [[1.0, 0.4],
               [0.4, 0.5]]
    
    # Train the correlated 2D VAE
    vae, encoder, decoder, history = train_correlated_vae(
        latent_dim=2,
        cov_matrix=cov_mat,
        epochs=30,
        batch_size=128
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Load labeled data for clustering visualization
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    
    # 1. Latent manifold
    plot_latent_space(vae, n=30)
    
    # 2. Clustering
    plot_label_clusters(vae, x_train, y_train)
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Total Loss: {history.history['loss'][-1]:.2f}")
    print(f"  Reconstruction Loss: {history.history['reconstruction_loss'][-1]:.2f}")
    print(f"  KL Loss: {history.history['kl_loss'][-1]:.2f}")
    
    print("\nAll visualizations saved to ../assets/")
    
    # Compare with standard 2D VAE
    print("\n" + "=" * 60)
    print("Effect of Correlated Prior:")
    print("=" * 60)
    print("The correlated prior (with off-diagonal elements) encourages")
    print("the encoder to learn latent representations that respect the")
    print("correlation structure specified in the covariance matrix.")
    print("This can lead to different clustering patterns compared to")
    print("a standard isotropic Gaussian prior N(0, I).")
