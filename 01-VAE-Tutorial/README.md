# Variational Autoencoder (VAE) Tutorial — Understanding Latent Space

[![View Website](https://img.shields.io/badge/View-Website-blue)](index.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An interactive guide to understanding Variational Autoencoders through hands-on experiments with 2D and 3D latent spaces

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [What is a VAE?](#what-is-a-vae)
3. [Model Architecture](#model-architecture)
4. [2D Latent Space Experiments](#2d-latent-space-experiments)
5. [3D Latent Space Extension](#3d-latent-space-extension)
6. [Non-Equal Covariance Prior](#non-equal-covariance-prior)
7. [Implementation Guide](#implementation-guide)
8. [Results & Visualizations](#results--visualizations)
9. [References](#references)

---

## 🎯 Introduction

A **Variational Autoencoder (VAE)** is a powerful generative model that learns to encode input data into a low-dimensional probabilistic latent space and reconstruct it back to the original form. Unlike standard autoencoders that map each input deterministically to a latent vector, VAEs learn a *distribution* over the latent variables.

**Why is this important?**
- Enables generation of new, realistic samples
- Creates smooth, continuous latent representations
- Allows interpolation between data points
- Provides a probabilistic framework for understanding data

In this tutorial, we'll explore VAEs through experiments on the **MNIST** dataset of handwritten digits, starting with a simple 2D latent space and progressively adding complexity.

---

## 🧠 What is a VAE?

### Key Concepts

A VAE consists of three main components:

1. **Encoder** (Recognition Model)
   - Compresses input data $x$ into latent variables $z$
   - Outputs parameters of a distribution: $q(z|x) = \mathcal{N}(\mu, \sigma^2)$

2. **Latent Space** (Probabilistic Representation)
   - Prior distribution: $p(z) = \mathcal{N}(0, I)$
   - Learned posterior: $q(z|x)$

3. **Decoder** (Generative Model)
   - Reconstructs input from latent samples
   - $p(x|z)$ generates output from latent code

### The Reparameterization Trick

To enable backpropagation through random sampling, VAEs use the **reparameterization trick**:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This separates the stochastic component ($\epsilon$) from the learnable parameters ($\mu$, $\sigma$).

### Loss Function

The VAE optimizes a combination of two objectives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

- **Reconstruction Loss**: Ensures output fidelity (binary cross-entropy or MSE)
- **KL Divergence**: Regularizes latent space to match the prior

$$\mathcal{L}_{\text{KL}} = D_{KL}(q(z|x) \| p(z)) = \frac{1}{2} \sum_{i=1}^{k} \left(1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2\right)$$

---

## 🏗️ Model Architecture

### 2D VAE Architecture

```
Input: 28×28 grayscale image (784 pixels)
    ↓
Encoder:
    Dense(512, relu)
    Dense(256, relu)
    ↓
Latent Parameters:
    z_mean: Dense(2)
    z_log_var: Dense(2)
    ↓
Sampling Layer (reparameterization)
    z = μ + σ ⊙ ε
    ↓
Decoder:
    Dense(256, relu)
    Dense(512, relu)
    Dense(784, sigmoid)
    ↓
Output: 28×28 reconstructed image
```

### Training Hyperparameters

- **Optimizer**: Adam
- **Epochs**: 30
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Latent Dimensions**: 2 (or 3 for extended version)

---

## 🎨 2D Latent Space Experiments

### Objective

Train a VAE with a **2-dimensional latent space** on MNIST to:
- Visualize how digits are organized in latent space
- Generate new digits by sampling from the latent space
- Understand the learned manifold structure

### Results

After 30 epochs:
- **Reconstruction Loss**: ~161
- **KL Divergence**: ~3.9
- **Total Loss**: ~165

### Visualization 1: Latent Manifold

By sampling a grid of points in the 2D latent space and decoding them, we can visualize the learned manifold:

![2D Latent Manifold](assets/latent2D.png)

**Key Observations:**
- Smooth transitions between different digit classes
- Continuous manifold structure
- Neighboring points produce visually similar digits
- Edge regions show interesting interpolations

### Visualization 2: Clustering

Plotting the latent encodings of test samples reveals natural clustering:

![2D Clustering](assets/Clusters2D.png)

**Key Observations:**
- Each digit class forms distinct clusters
- Similar digits (e.g., 3 and 8, 4 and 9) are positioned near each other
- Some overlap indicates visual similarity
- The encoder has learned meaningful semantic relationships

---

## 📦 3D Latent Space Extension

### Motivation

While 2D latent spaces are easy to visualize, they may be too restrictive for complex data. By extending to 3D, we can:
- Increase representational capacity
- Capture more factors of variation
- Reduce reconstruction loss

### Implementation

Simply change the latent dimension:

```python
latent_dim = 3  # Changed from 2
```

All other architecture components remain the same.

### Results

Training for 30 epochs with 3D latent space:
- **Total Loss**: ~164 (slightly lower than 2D)
- Better variance representation
- More disentangled factors

### Visualization 1: 3D Scatter Plot

![3D Clustering](assets/Clusters3D.png)

The 3D latent space shows clear, smooth clustering with enhanced separation between digit classes.

### Visualization 2: Pairwise 2D Projections

To understand how information is distributed across dimensions, we plot pairs $(z_0, z_1)$, $(z_1, z_2)$, and $(z_0, z_2)$:

![2D Projections of 3D Space](assets/2D3DClusters.png)

**Key Observations:**
- Each dimension contributes meaningful variation
- Clusters remain coherent across all projections
- No single dimension dominates

### Visualization 3: 2D Cross-Sections

By fixing $z_2$ and varying $(z_0, z_1)$, we can slice through the 3D manifold:

| $z_2 = -1.0$ | $z_2 = 0.0$ | $z_2 = 1.0$ |
|:------------:|:-----------:|:-----------:|
| ![](assets/3Dlatent1.png) | ![](assets/3DLatent2.png) | ![](assets/3DLatent3.png) |

**Key Observations:**
- Different slices show different digit styles
- $z_2$ primarily modulates stroke thickness and style
- $z_0$ and $z_1$ capture digit identity
- Smooth transitions demonstrate continuity

---

## 🔄 Non-Equal Covariance Prior

### Motivation

The standard VAE assumes an **isotropic prior**:

$$p(z) = \mathcal{N}(0, I)$$

This enforces independence between latent dimensions. However, real-world factors are often correlated. We can model this with a **correlated Gaussian prior**:

$$p(z) = \mathcal{N}(0, \Sigma)$$

where $\Sigma$ is a full covariance matrix.

### Implementation Changes

#### 1. Modified Sampling Layer

Replace standard reparameterization with Cholesky-based sampling:

```python
# Standard: z = μ + σ ⊙ ε
# Correlated: z = μ + L ε, where Σ = L Lᵀ

L = np.linalg.cholesky(Sigma)
z = z_mean + tf.matmul(epsilon, L.T)
```

For this experiment, we use:

$$\Sigma = \begin{bmatrix} 1.0 & 0.4 \\ 0.4 & 0.5 \end{bmatrix}$$

This introduces positive correlation between $z_0$ and $z_1$.

#### 2. Updated KL Divergence

The KL divergence must account for the correlated prior:

$$D_{KL}(q(z|x) \| p(z)) = \frac{1}{2} \left[\text{tr}(\Sigma^{-1}\text{diag}(\sigma^2)) + \mu^T\Sigma^{-1}\mu - k + \log\frac{\det(\Sigma)}{\prod_i \sigma_i^2}\right]$$

### Results

| Generated Manifold | Latent Clusters |
|:------------------:|:---------------:|
| ![](assets/KLAdapted.png) | ![](assets/Screenshot%202025-11-07%20at%2014.41.06.png) |

**Key Observations:**
- Manifold appears **tilted and stretched**
- Latent clusters form elongated, rotated patterns
- Geometry matches the correlation structure in $\Sigma$
- Reconstruction quality remains similar to isotropic case

### Discussion

Introducing correlation adds flexibility by allowing latent dimensions to capture related factors of variation. While reconstruction remains comparable, the latent geometry becomes more structured and interpretable according to the specified correlation pattern.

---

## 💻 Implementation Guide

### Prerequisites

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

### Basic 2D VAE Implementation

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Hyperparameters
latent_dim = 2
input_shape = (28, 28, 1)

# Encoder
encoder_inputs = keras.Input(shape=input_shape)
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)

z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(latent_inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_outputs = layers.Reshape((28, 28, 1))(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        
        return reconstructed

# Compile and train
vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss='binary_crossentropy')

# Load MNIST data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)

# Train
vae.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.1)
```

### Visualization Code

```python
import matplotlib.pyplot as plt

# 1. Plot latent space clusters
z_mean, _, _ = encoder.predict(x_test)
plt.figure(figsize=(10, 8))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='tab10', alpha=0.5)
plt.colorbar()
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.title('2D Latent Space Clustering')
plt.show()

# 2. Generate manifold grid
n = 20
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

figure = np.zeros((28 * n, 28 * n))
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28,
               j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='viridis')
plt.title('2D Latent Manifold')
plt.axis('off')
plt.show()
```

---

## 📊 Results & Visualizations

### Summary Table

| Configuration | Latent Dim | Total Loss | KL Loss | Reconstruction Loss |
|--------------|------------|------------|---------|---------------------|
| Standard 2D  | 2          | ~165       | ~3.9    | ~161                |
| Standard 3D  | 3          | ~164       | ~4.2    | ~160                |
| Correlated 2D| 2          | ~166       | ~4.1    | ~162                |

### Key Findings

1. **2D Latent Space**
   - Easy to visualize and interpret
   - Clear digit clustering
   - Smooth manifold transitions
   - Sufficient for MNIST complexity

2. **3D Latent Space**
   - Slightly better reconstruction
   - Additional dimension captures style variations
   - More disentangled representations
   - Still interpretable through projections

3. **Correlated Prior**
   - Reshapes latent geometry
   - Enables modeling of dependent factors
   - Similar reconstruction quality
   - More structured latent organization

---

## 🎓 Key Takeaways

### What We Learned

1. **VAEs learn continuous latent representations** that enable smooth interpolation and generation
2. **The dimensionality of latent space** affects both reconstruction quality and interpretability
3. **KL divergence regularization** is crucial for learning structured, meaningful latent spaces
4. **Different priors** (isotropic vs. correlated) influence the geometry of learned representations
5. **Visualization techniques** (manifold grids, clustering, projections) help understand what the model learns

### When to Use VAEs

✅ **Good for:**
- Learning compact data representations
- Generating new samples similar to training data
- Interpolating between data points
- Unsupervised feature learning
- Data compression with probabilistic framework

❌ **Limitations:**
- Generated samples may be blurry (compared to GANs)
- Assumes specific distributional form
- KL divergence can be difficult to balance
- May struggle with very high-dimensional latent spaces

---

## 🔗 References

### Papers
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling (2013)
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) - Doersch (2016)

### Code Resources
- [Keras VAE Example](https://keras.io/examples/generative/vae/)
- [TensorFlow CVAE Tutorial](https://www.tensorflow.org/tutorials/generative/cvae)

### Additional Reading
- [Understanding VAEs](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)

---

## 📝 License

MIT License - Feel free to use this tutorial for educational purposes.

---

## 👤 Author

**Maroua Oukrid**

If you found this tutorial helpful, please star the repository!

---

**Last Updated**: November 2024

