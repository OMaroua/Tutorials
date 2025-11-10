"""
2D Variational Autoencoder (VAE) Implementation
================================================

This script implements a basic 2D VAE on the MNIST dataset.
The 2D latent space allows for easy visualization and interpretation.

Author: Maroua Oukrid
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.manifold import TSNE

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================
# Model Architecture
# ============================================

class Sampling(layers.Layer):
    """
    Implements the reparameterization trick:
    z = μ + σ * ε, where ε ~ N(0, I)
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim=2, input_shape=(28, 28, 1)):
    """
    Build the encoder network that compresses images to latent representations.
    
    Args:
        latent_dim: Dimensionality of the latent space
        input_shape: Shape of input images
    
    Returns:
        Keras Model for the encoder
    """
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(512, activation='relu', name='encoder_dense_1')(x)
    x = layers.Dense(256, activation='relu', name='encoder_dense_2')(x)
    
    # Output: mean and log variance of the latent distribution
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sample from the latent distribution
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder


def build_decoder(latent_dim=2):
    """
    Build the decoder network that reconstructs images from latent codes.
    
    Args:
        latent_dim: Dimensionality of the latent space
    
    Returns:
        Keras Model for the decoder
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(256, activation='relu', name='decoder_dense_1')(latent_inputs)
    x = layers.Dense(512, activation='relu', name='decoder_dense_2')(x)
    x = layers.Dense(28 * 28, activation='sigmoid', name='decoder_dense_3')(x)
    decoder_outputs = layers.Reshape((28, 28, 1))(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
    return decoder


# ============================================
# VAE Model
# ============================================

class VAE(keras.Model):
    """
    Variational Autoencoder combining encoder and decoder with VAE loss.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss (binary cross-entropy)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
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
# Data Loading
# ============================================

def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


# ============================================
# Visualization Functions
# ============================================

def plot_latent_space_clusters(encoder, x_test, y_test, save_path='../assets/Clusters2D.png'):
    """
    Plot the 2D latent space colored by digit class.
    
    Args:
        encoder: Trained encoder model
        x_test: Test images
        y_test: Test labels
        save_path: Path to save the figure
    """
    # Encode test data
    z_mean, _, _ = encoder.predict(x_test, verbose=0)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                         c=y_test, cmap='tab10', 
                         alpha=0.6, s=10, edgecolors='none')
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('z[0]', fontsize=14)
    plt.ylabel('z[1]', fontsize=14)
    plt.title('2D Latent Space Clustering', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved latent space clusters to {save_path}")
    plt.close()


def plot_latent_manifold(decoder, n=20, digit_size=28, save_path='../assets/latent2D.png'):
    """
    Generate a grid of digits by sampling the latent space.
    
    Args:
        decoder: Trained decoder model
        n: Grid size (n x n)
        digit_size: Size of each digit
        save_path: Path to save the figure
    """
    # Sample points from the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]  # Reverse for proper orientation
    
    figure = np.zeros((digit_size * n, digit_size * n))
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(12, 12))
    plt.imshow(figure, cmap='viridis')
    plt.title('2D Latent Space Manifold', fontsize=16, fontweight='bold')
    plt.xlabel('z[0]', fontsize=14)
    plt.ylabel('z[1]', fontsize=14)
    plt.colorbar(label='Pixel Intensity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved latent manifold to {save_path}")
    plt.close()


def plot_reconstructions(vae, x_test, n=10, save_path='../assets/reconstructions_2d.png'):
    """
    Plot original images and their reconstructions.
    
    Args:
        vae: Trained VAE model
        x_test: Test images
        n: Number of examples to show
        save_path: Path to save the figure
    """
    # Get random samples
    idx = np.random.choice(len(x_test), n, replace=False)
    x_samples = x_test[idx]
    
    # Get reconstructions
    reconstructions = vae.predict(x_samples, verbose=0)
    
    # Plot
    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3))
    
    for i in range(n):
        # Original
        axes[0, i].imshow(x_samples[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10, fontweight='bold')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10, fontweight='bold')
    
    plt.suptitle('Original vs Reconstructed Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstructions to {save_path}")
    plt.close()


# ============================================
# Training
# ============================================

def train_vae(latent_dim=2, epochs=30, batch_size=128):
    """
    Complete training pipeline for 2D VAE.
    
    Args:
        latent_dim: Dimensionality of latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained VAE model, encoder, decoder, and test data
    """
    print("=" * 60)
    print("Training 2D Variational Autoencoder")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Build models
    print("\n[2/5] Building encoder and decoder...")
    encoder = build_encoder(latent_dim=latent_dim)
    decoder = build_decoder(latent_dim=latent_dim)
    
    print("\nEncoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    
    # Build VAE
    print("\n[3/5] Creating VAE model...")
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
    print("\n[5/5] Generating visualizations...")
    plot_latent_space_clusters(encoder, x_test, y_test)
    plot_latent_manifold(decoder)
    plot_reconstructions(vae, x_test)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return vae, encoder, decoder, (x_test, y_test), history


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Train the model
    vae, encoder, decoder, (x_test, y_test), history = train_vae(
        latent_dim=2,
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
    encoder.save('models/encoder_2d.h5')
    decoder.save('models/decoder_2d.h5')
    vae.save_weights('models/vae_2d_weights.h5')
    print("Models saved successfully!")

