"""
3D Variational Autoencoder (VAE) Implementation
================================================

This script implements a VAE with a 3-dimensional latent space using
convolutional layers for better feature extraction from MNIST images.

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
from mpl_toolkits.mplot3d import Axes3D


# ============================================
# Custom Sampling Layer
# ============================================

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# ============================================
# Build Encoder (Convolutional)
# ============================================

def build_encoder(latent_dim=3):
    """
    Build convolutional encoder network for 3D latent space.
    
    Args:
        latent_dim: Dimensionality of the latent space (default: 3)
    
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
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


# ============================================
# Build Decoder (Convolutional Transpose)
# ============================================

def build_decoder(latent_dim=3):
    """
    Build convolutional transpose decoder network.
    
    Args:
        latent_dim: Dimensionality of the latent space (default: 3)
    
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
# VAE Model with Custom Training Loop
# ============================================

class VAE(keras.Model):
    """Variational Autoencoder with custom training step."""
    
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
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

def plot_latent_space(vae, n=20, z_fixed=0.0, figsize=12):
    """
    Display a n×n 2D manifold of digits at a fixed z[2] value.
    
    Args:
        vae: Trained VAE model
        n: Grid size (number of samples per dimension)
        z_fixed: Fixed value for z[2] dimension
        figsize: Figure size
    """
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    
    # Create grid of latent values
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi, z_fixed]])  # 3D latent vector
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
    plt.title(f"Latent slice at z[2] = {z_fixed}")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(f'../assets/3d_manifold_z2_{z_fixed}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_label_clusters_3d(vae, data, labels):
    """
    Display a 3D scatter plot of the digit classes in the latent space.
    
    Args:
        vae: Trained VAE model
        data: Input images
        labels: Corresponding labels
    """
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], 
                   c=labels, cmap='tab10', s=3)
    fig.colorbar(p)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    ax.set_title("3D Latent Space Clustering")
    plt.savefig('../assets/Clusters3D.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_projections(vae, data, labels):
    """
    Display all 3 pairwise 2D projections of the 3D latent space.
    
    Args:
        vae: Trained VAE model
        data: Input images
        labels: Corresponding labels
    """
    # Encode the training data into latent means
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    
    # Define all 3 pairwise axis combinations
    pairs = [(0, 1), (1, 2), (0, 2)]
    titles = [("z[0]", "z[1]"), ("z[1]", "z[2]"), ("z[0]", "z[2]")]
    
    plt.figure(figsize=(18, 5))
    
    for i, ((a, b), (xlabel, ylabel)) in enumerate(zip(pairs, titles)):
        plt.subplot(1, 3, i + 1)
        sc = plt.scatter(z_mean[:, a], z_mean[:, b], c=labels, cmap="tab10", s=3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Projection: {xlabel} vs {ylabel}")
        plt.colorbar(sc, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("../assets/2D3DClusters.png", dpi=300)
    plt.show()


# ============================================
# Training Pipeline
# ============================================

def train_3d_vae(latent_dim=3, epochs=30, batch_size=128):
    """
    Complete training pipeline for 3D VAE.
    
    Args:
        latent_dim: Dimensionality of latent space (default: 3)
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained VAE model, encoder, and decoder
    """
    print("=" * 60)
    print("Training 3D Variational Autoencoder")
    print("=" * 60)
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    
    # Build models
    print("Building encoder and decoder...")
    encoder = build_encoder(latent_dim=latent_dim)
    decoder = build_decoder(latent_dim=latent_dim)
    
    print("\nEncoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    
    # Create and compile VAE
    vae = VAE(encoder, decoder)
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
    
    # Train the 3D VAE
    vae, encoder, decoder, history = train_3d_vae(
        latent_dim=3,
        epochs=30,
        batch_size=128
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Load labeled data for clustering visualization
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    
    # 1. 3D scatter plot
    plot_label_clusters_3d(vae, x_train, y_train)
    
    # 2. 2D projections
    plot_2d_projections(vae, x_train, y_train)
    
    # 3. Manifold slices at different z[2] values
    for z2 in [-1.0, 0.0, 1.0]:
        plot_latent_space(vae, n=20, z_fixed=z2)
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Total Loss: {history.history['loss'][-1]:.2f}")
    print(f"  Reconstruction Loss: {history.history['reconstruction_loss'][-1]:.2f}")
    print(f"  KL Loss: {history.history['kl_loss'][-1]:.2f}")
    
    print("\nAll visualizations saved to ../assets/")
