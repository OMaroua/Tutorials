# 🎨 Variational Autoencoders (VAEs) Tutorial

## 📖 **Overview**

This tutorial provides a comprehensive introduction to **Variational Autoencoders (VAEs)**, one of the most important generative models in deep learning. You'll learn the theory behind VAEs and implement one from scratch using PyTorch.

---

## 🎯 **Learning Objectives**

By the end of this tutorial, you will be able to:

- ✅ Understand the mathematical foundation of VAEs
- ✅ Implement the encoder and decoder networks
- ✅ Apply the reparameterization trick
- ✅ Compute and optimize the VAE loss (reconstruction + KL divergence)
- ✅ Generate new samples from the latent space
- ✅ Visualize and explore the latent space representation
- ✅ Apply VAEs to real datasets (MNIST, Fashion-MNIST, etc.)

---

## 📚 **What You'll Learn**

### 1. **VAE Theory**
- Traditional autoencoders vs. variational autoencoders
- Probabilistic interpretation
- The Evidence Lower Bound (ELBO)
- Latent space distributions

### 2. **Implementation**
- Building the encoder (recognition model)
- Building the decoder (generative model)
- Reparameterization trick for backpropagation
- Loss function implementation

### 3. **Training & Evaluation**
- Training loop with PyTorch
- Monitoring reconstruction quality
- KL divergence balancing
- Generating new samples

### 4. **Advanced Topics**
- Conditional VAEs (CVAE)
- β-VAE for disentangled representations
- Applications in computer vision and healthcare

---

## 🛠️ **Prerequisites**

**Required Knowledge:**
- Python programming
- Basic neural networks
- PyTorch fundamentals
- Probability and statistics (Gaussian distributions, KL divergence)

**Required Software:**
- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook

---

## 📦 **Installation**

```bash
# Navigate to this tutorial folder
cd 01-VAE-Tutorial

# Install requirements
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook VAE_Tutorial.ipynb
```

---

## 📁 **Folder Structure**

```
01-VAE-Tutorial/
│
├── README.md                    # This file
├── VAE_Tutorial.ipynb          # Main tutorial notebook
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── vae_model.py           # VAE model implementation
│   ├── train.py               # Training script
│   └── utils.py               # Helper functions
│
├── assets/
│   ├── vae_architecture.png   # Architecture diagram
│   ├── latent_space.png       # Latent space visualization
│   └── generated_samples.png  # Generated images
│
└── data/
    └── README.md              # Data information
```

---

## 🚀 **Quick Start**

### Option 1: Interactive Notebook (Recommended)
```bash
jupyter notebook VAE_Tutorial.ipynb
```

### Option 2: Run Training Script
```bash
python src/train.py --dataset mnist --epochs 50 --latent-dim 20
```

---

## 📊 **Datasets Used**

- **MNIST**: Handwritten digits (28x28 grayscale)
- **Fashion-MNIST**: Clothing items (28x28 grayscale)
- **CelebA**: Celebrity faces (optional, for advanced section)

All datasets are automatically downloaded via PyTorch's `torchvision`.

---

## 🔬 **Key Concepts Covered**

### VAE Loss Function
```python
loss = reconstruction_loss + β * kl_divergence
```

Where:
- **Reconstruction Loss**: Measures how well the decoder reconstructs the input
- **KL Divergence**: Regularizes the latent space to follow a standard normal distribution
- **β**: Balancing parameter (β-VAE)

### Reparameterization Trick
```python
z = μ + σ * ε, where ε ~ N(0, 1)
```

This allows gradients to flow through the stochastic sampling operation.

---

## 📈 **Expected Results**

After training, you should be able to:
- Generate realistic new samples
- Interpolate smoothly in latent space
- Reconstruct input images accurately
- Achieve ~100 ELBO on MNIST

---

## 🎓 **Further Reading**

- [Original VAE Paper](https://arxiv.org/abs/1312.6114) - Kingma & Welling (2013)
- [β-VAE Paper](https://openreview.net/forum?id=Sy2fzU9gl) - Higgins et al. (2017)
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) - Doersch (2016)

---

## 💡 **Tips & Tricks**

1. **Start with small latent dimensions** (2-20) to visualize the latent space
2. **Balance the loss terms** - if reconstruction is poor, decrease β
3. **Use batch normalization** for more stable training
4. **Monitor both loss terms** separately during training

---

## 🐛 **Common Issues**

| Issue | Solution |
|-------|----------|
| Blurry reconstructions | Try using a different loss (e.g., perceptual loss) |
| Posterior collapse | Reduce KL weight (β) or use KL annealing |
| Mode collapse | Increase model capacity or latent dimensions |

---

## 🤝 **Contributing**

Found a bug or have a suggestion? Feel free to:
- Open an issue
- Submit a pull request
- Reach out via email

---

## 👩‍💻 **Author**

**Maroua Oukrid**  
Computer Vision & Healthcare AI Researcher

📧 marouaoukrid56@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/Maroua-Oukrid)  
🐱 [GitHub](https://github.com/OMaroua)

---

## 📄 **License**

MIT License - Feel free to use for learning and teaching!

---

⭐ **If you find this tutorial helpful, please star the repository!**

**Next Tutorial:** [Diffusion Models →](../02-Diffusion-Models/)

