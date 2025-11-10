# 🧠 Variational Autoencoder (VAE) Tutorial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

> A comprehensive, interactive tutorial on Variational Autoencoders with hands-on experiments exploring 2D/3D latent spaces and correlated priors.

---

## 📖 About

This tutorial provides an in-depth exploration of Variational Autoencoders (VAEs) through three progressive experiments:

1. **2D Latent Space** - Foundational understanding with easy visualization
2. **3D Latent Space** - Extended capacity and factor disentanglement  
3. **Correlated Prior** - Non-isotropic Gaussian priors for dependent factors

Each experiment includes complete implementations, visualizations, and theoretical explanations.

---

## 🚀 Quick Start

### View the Tutorial

**Option 1: Website Version** (Recommended)
```bash
# Clone or navigate to the tutorial
cd 01-VAE-Tutorial

# Open index.html in your browser
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows
```

**Option 2: GitHub README**
```bash
# Read the markdown version
cat README.md
# Or view on GitHub for formatted rendering
```

### Run the Code

1. **Create virtual environment:**
```bash
python -m venv vae_env
source vae_env/bin/activate  # On Windows: vae_env\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run experiments:**
```bash
# 2D VAE
python code/vae_2d.py

# 3D VAE
python code/vae_3d.py

# Correlated Prior VAE
python code/vae_correlated.py
```

---

## 📁 Project Structure

```
01-VAE-Tutorial/
├── README.md                 # Main tutorial (GitHub-friendly)
├── index.html               # Interactive website version
├── styles.css               # Modern styling
├── script.js                # Interactive components
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── TUTORIAL_README.md       # This file
│
├── assets/                  # Visualizations and images
│   ├── README.md           # Image placement guide
│   ├── latent2D.png        # 2D manifold
│   ├── Clusters2D.png      # 2D clustering
│   ├── Clusters3D.png      # 3D clustering
│   ├── 2D3DClusters.png    # 3D projections
│   ├── 3DLatent*.png       # 3D slices
│   └── KLAdapted.png       # Correlated prior results
│
└── code/                    # Python implementations
    ├── README.md           # Code documentation
    ├── vae_2d.py           # 2D VAE implementation
    ├── vae_3d.py           # 3D VAE implementation
    └── vae_correlated.py   # Correlated prior VAE
```

---

## 🎓 What You'll Learn

### Core Concepts

- **VAE Architecture**: Encoder, decoder, and latent space
- **Reparameterization Trick**: Differentiable sampling
- **Loss Function**: Reconstruction loss + KL divergence
- **Latent Space Geometry**: How VAEs organize data

### Advanced Topics

- **Dimensionality Effects**: 2D vs 3D latent spaces
- **Disentanglement**: Separating factors of variation
- **Prior Selection**: Isotropic vs correlated Gaussian priors
- **Visualization Techniques**: Manifolds, clustering, projections

### Practical Skills

- Implementing VAEs in TensorFlow/Keras
- Training and debugging generative models
- Visualizing high-dimensional latent spaces
- Customizing architecture and hyperparameters

---

## 🔬 Experiments Overview

### Experiment 1: 2D Latent Space

**Goal**: Understand basic VAE behavior with easy visualization

**Key Results**:
- Total Loss: ~165
- Clear digit clustering in 2D plane
- Smooth manifold transitions

**Visualizations**:
- Latent space scatter plot colored by digit
- 2D manifold grid (20×20 samples)
- Original vs reconstructed images

**Duration**: ~10-15 minutes (CPU), ~3-5 minutes (GPU)

---

### Experiment 2: 3D Latent Space

**Goal**: Explore increased representational capacity

**Key Results**:
- Total Loss: ~164 (improved)
- Better factor disentanglement
- Third dimension captures style variations

**Visualizations**:
- 3D scatter plot with rotation
- Pairwise 2D projections
- Cross-sectional slices at different z₂ values
- Variance explained per dimension

**Duration**: ~10-15 minutes (CPU), ~3-5 minutes (GPU)

---

### Experiment 3: Correlated Prior

**Goal**: Model dependent latent factors with non-isotropic prior

**Configuration**:
```python
Σ = [[1.0, 0.4],
     [0.4, 0.5]]
```

**Key Results**:
- Total Loss: ~166
- Tilted, elongated latent clusters
- Geometry matches correlation structure

**Visualizations**:
- Latent space with covariance ellipses
- Tilted manifold grid
- Covariance matrix heatmap

**Duration**: ~10-15 minutes (CPU), ~3-5 minutes (GPU)

---

## 🛠️ Technical Details

### Requirements

- Python 3.8 or higher
- TensorFlow 2.10+
- NumPy, Matplotlib, scikit-learn
- ~2GB RAM for training
- GPU recommended but not required

### Model Architecture

```
Encoder:
  Input (28×28×1) → Flatten (784)
  → Dense(512, ReLU) → Dense(256, ReLU)
  → z_mean(latent_dim), z_log_var(latent_dim)
  → Sampling → z(latent_dim)

Decoder:
  z(latent_dim) → Dense(256, ReLU)
  → Dense(512, ReLU) → Dense(784, Sigmoid)
  → Reshape(28×28×1)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 30 |
| Batch Size | 128 |
| Dataset | MNIST (60k train, 10k test) |
| Validation Split | 10% |

---

## 📊 Expected Results

### Loss Curves

All models should show:
- **Total loss**: Decreasing steadily
- **Reconstruction loss**: Majority of total loss (~160)
- **KL loss**: Small but non-zero (~4)

### Visual Quality

- **Good**: Smooth manifolds, clear clustering
- **Issues**: 
  - Blurriness is normal (VAE property)
  - Posterior collapse → adjust KL weight
  - Poor clustering → increase capacity or epochs

---

## 🎨 Customization Guide

### Change Latent Dimensions

```python
# In vae_2d.py, vae_3d.py
vae, encoder, decoder, data, history = train_vae(
    latent_dim=5,  # Change from 2 or 3
    epochs=30,
    batch_size=128
)
```

### Modify Architecture

```python
def build_encoder(latent_dim=2):
    # Add more layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    # ...
```

### Beta-VAE (Emphasis on Disentanglement)

```python
# In VAE.train_step()
total_loss = reconstruction_loss + beta * kl_loss  # beta > 1
```

### Custom Covariance

```python
# In vae_correlated.py
covariance_matrix = np.array([
    [2.0, -0.7],  # Negative correlation
    [-0.7, 1.5]
])
```

---

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Make sure virtual environment is activated and requirements installed
```bash
source vae_env/bin/activate
pip install -r requirements.txt
```

### Issue: GPU not detected
**Solution**: Install TensorFlow GPU version
```bash
pip install tensorflow-gpu==2.10.0
```

### Issue: Out of memory
**Solution**: Reduce batch size
```python
batch_size=64  # or even 32
```

### Issue: Posterior collapse (KL → 0)
**Solution**: 
- Add KL annealing (gradually increase weight)
- Reduce encoder capacity
- Increase decoder capacity

### Issue: Poor reconstruction
**Solution**:
- Train longer (more epochs)
- Increase model capacity
- Check data normalization
- Reduce KL weight (beta < 1)

---

## 📚 Further Reading

### Papers

- [Kingma & Welling (2013) - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Doersch (2016) - Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [Higgins et al. (2017) - β-VAE](https://openreview.net/forum?id=Sy2fzU9gl)

### Tutorials

- [Keras VAE Example](https://keras.io/examples/generative/vae/)
- [TensorFlow Probability VAE](https://www.tensorflow.org/probability/examples/Variational_Autoencoders)
- [Understanding VAEs (Blog)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

### Extensions

- **Conditional VAE (CVAE)**: Add class labels
- **Adversarial VAE**: Combine with GAN discriminator
- **Hierarchical VAE**: Multiple latent layers
- **VQ-VAE**: Discrete latent codes

---

## 🤝 Contributing

Found an issue or want to improve the tutorial?

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit (`git commit -am 'Add improvement'`)
5. Push (`git push origin feature/improvement`)
6. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Maroua Oukrid**

- A20594067
- November 2024

---

## 🌟 Acknowledgments

- Based on original Keras VAE example
- MNIST dataset from Yann LeCun et al.
- Inspired by Kingma & Welling's seminal paper

---

## 📬 Questions & Feedback

If you have questions, suggestions, or found this tutorial helpful:

- ⭐ Star the repository
- 🐛 Open an issue for bugs
- 💡 Suggest improvements via pull requests
- 📧 Contact via GitHub

---

**Happy Learning! 🎉**

*Last Updated: November 2024*

