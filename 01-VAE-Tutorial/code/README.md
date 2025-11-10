# VAE Implementation Code

This directory contains the Python implementations for all VAE experiments discussed in the tutorial.

## 📁 Files

### Core Implementations

1. **`vae_2d.py`** - Standard 2D VAE
   - Basic VAE with 2-dimensional latent space
   - Easy visualization and interpretation
   - Generates latent space clusters and manifold grids

2. **`vae_3d.py`** - Extended 3D VAE
   - 3-dimensional latent space for increased capacity
   - 3D scatter plots and 2D projections
   - Cross-sectional slices through latent space

3. **`vae_correlated.py`** - VAE with Correlated Prior
   - Non-isotropic Gaussian prior with covariance structure
   - Cholesky-based sampling
   - Modified KL divergence computation

## 🚀 Getting Started

### Installation

1. Create a virtual environment (recommended):
```bash
python -m venv vae_env
source vae_env/bin/activate  # On Windows: vae_env\Scripts\activate
```

2. Install requirements:
```bash
pip install -r ../requirements.txt
```

### Running the Experiments

#### 2D VAE
```bash
python vae_2d.py
```

**Outputs:**
- `../assets/Clusters2D.png` - Latent space clustering
- `../assets/latent2D.png` - 2D manifold grid
- `../assets/reconstructions_2d.png` - Original vs reconstructed images
- `models/encoder_2d.h5` - Trained encoder
- `models/decoder_2d.h5` - Trained decoder

**Expected Results:**
- Total Loss: ~165
- Reconstruction Loss: ~161
- KL Loss: ~3.9
- Training time: ~10-15 minutes (CPU), ~3-5 minutes (GPU)

#### 3D VAE
```bash
python vae_3d.py
```

**Outputs:**
- `../assets/Clusters3D.png` - 3D latent space scatter plot
- `../assets/2D3DClusters.png` - Pairwise 2D projections
- `../assets/3DLatent1.png`, `3DLatent2.png`, `3DLatent3.png` - Manifold slices
- `../assets/latent_variance_3d.png` - Variance per dimension
- `models/encoder_3d.h5` - Trained encoder
- `models/decoder_3d.h5` - Trained decoder

**Expected Results:**
- Total Loss: ~164
- Reconstruction Loss: ~160
- KL Loss: ~4.2
- Training time: ~10-15 minutes (CPU), ~3-5 minutes (GPU)

#### Correlated Prior VAE
```bash
python vae_correlated.py
```

**Outputs:**
- `../assets/Screenshot 2025-11-07 at 14.41.06.png` - Correlated latent space
- `../assets/KLAdapted.png` - Manifold with tilted structure
- `../assets/covariance_matrix.png` - Prior covariance visualization
- `models/encoder_correlated.h5` - Trained encoder
- `models/decoder_correlated.h5` - Trained decoder

**Expected Results:**
- Total Loss: ~166
- Reconstruction Loss: ~162
- KL Loss: ~4.1
- Correlation coefficient: 0.4
- Training time: ~10-15 minutes (CPU), ~3-5 minutes (GPU)

## 🔧 Customization

### Changing Hyperparameters

Edit the main execution block at the bottom of each script:

```python
if __name__ == "__main__":
    vae, encoder, decoder, (x_test, y_test), history = train_vae(
        latent_dim=2,      # Change latent dimensions
        epochs=30,         # Change number of epochs
        batch_size=128     # Change batch size
    )
```

### Modifying Architecture

The encoder and decoder architectures can be modified in the `build_encoder()` and `build_decoder()` functions:

```python
def build_encoder(latent_dim=2, input_shape=(28, 28, 1)):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(encoder_inputs)
    
    # Modify these layers
    x = layers.Dense(512, activation='relu')(x)  # Change units
    x = layers.Dense(256, activation='relu')(x)  # Add/remove layers
    
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    # ...
```

### Custom Covariance Matrix

For the correlated prior VAE, modify the covariance matrix:

```python
# Example: Negative correlation
covariance_matrix = np.array([
    [1.0, -0.5],
    [-0.5, 1.0]
])

# Example: Stronger positive correlation
covariance_matrix = np.array([
    [1.5, 0.8],
    [0.8, 0.8]
])
```

## 📊 Understanding the Output

### Loss Metrics

- **Reconstruction Loss**: Measures how well the VAE reconstructs input images
  - Lower is better
  - Binary cross-entropy for MNIST (pixel values in [0,1])
  
- **KL Divergence**: Regularization term ensuring latent space matches prior
  - Should be positive but not too large
  - Too low → posterior collapse
  - Too high → poor reconstruction

- **Total Loss**: Sum of reconstruction and KL losses
  - Overall training objective
  - Should decrease over epochs

### Visualization Interpretation

1. **Latent Space Clusters**
   - Points are test images encoded to latent space
   - Colors represent digit classes (0-9)
   - Clustering indicates learned semantic structure
   - Overlap shows visual similarity between digits

2. **Latent Manifold Grid**
   - Each cell is a decoded sample from a grid point in latent space
   - Smooth transitions indicate continuous learned manifold
   - Center typically contains most common digit features
   - Edges show interesting interpolations

3. **3D Projections**
   - Show how information is distributed across dimensions
   - Each 2D projection reveals different aspects of clustering
   - Helps identify redundancy or disentanglement

## 🐛 Troubleshooting

### Issue: Models not converging
**Solution:** 
- Reduce learning rate: `Adam(learning_rate=0.0001)`
- Increase number of epochs
- Check data normalization

### Issue: Posterior collapse (KL loss → 0)
**Solution:**
- Add KL annealing: gradually increase KL weight from 0 to 1
- Reduce encoder capacity
- Use free bits technique

### Issue: Blurry reconstructions
**Solution:**
- This is normal for VAEs (vs GANs)
- Try increasing model capacity
- Consider using perceptual loss instead of MSE
- Reduce KL weight (beta-VAE: β < 1)

### Issue: GPU memory errors
**Solution:**
- Reduce batch size: `batch_size=64` or `batch_size=32`
- Use gradient checkpointing
- Train on CPU (slower but works)

## 🔬 Further Experiments

### Beta-VAE
Modify the loss to emphasize disentanglement:
```python
total_loss = reconstruction_loss + beta * kl_loss  # beta > 1
```

### Conditional VAE (CVAE)
Add class labels as additional input to encoder and decoder.

### Different Architectures
- Add convolutional layers for image data
- Use ResNet-style skip connections
- Try different activation functions

### Other Datasets
Replace MNIST with:
- Fashion-MNIST
- CIFAR-10
- CelebA (faces)

## 📚 References

- [Original Keras VAE Example](https://keras.io/examples/generative/vae/)
- [Kingma & Welling (2013) - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Doersch (2016) - Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

## 📧 Questions?

If you have questions or find issues with the code, please:
1. Check the main tutorial README
2. Review the paper references
3. Open an issue on GitHub

---

**Happy Coding! 🎉**

