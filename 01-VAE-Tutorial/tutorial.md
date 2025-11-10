---
layout: tutorial
title: "Understanding Latents: Variational Autoencoders (VAEs)"
author: Maroua Oukrid
date: 2025-11-09
description: A comprehensive tutorial on Variational Autoencoders covering theory, mathematics, and implementation
topics: [Latent Representations, VAE, Generative Models, Deep Learning Theory]
---

# Understanding Latents: Variational Autoencoders (VAEs)

---

## Table of Contents

1. [Introduction to Latent Representations](#introduction)
2. [Autoencoders (AE)](#autoencoders)
3. [Variational Autoencoders (VAE)](#variational-autoencoders)
4. [Evidence Lower Bound (ELBO) - Derivation 1](#elbo-jensen)
5. [Evidence Lower Bound (ELBO) - Derivation 2](#elbo-alternative)
6. [VAE Optimization](#vae-optimization)
7. [The Reparameterization Trick](#reparameterization)
8. [Training Procedure](#training)
9. [Implementation Details](#implementation)
10. [Experiments on MNIST](#experiments)
11. [Comparison: VAE vs AE](#comparison)
12. [Conclusion](#conclusion)
13. [References](#references)

---

## Introduction to Latent Representations {#introduction}

Modern machine learning models rely fundamentally on **latent representations**—compact encodings that capture the underlying structure of high-dimensional data. Given a data point $\mathbf{x} \in \mathbb{R}^D$, we seek a lower-dimensional representation $\mathbf{z} \in \mathbb{R}^d$ where $d \ll D$.

The **manifold hypothesis** posits that high-dimensional data often lies on or near a low-dimensional manifold embedded in the ambient space. Learning this manifold enables dimensionality reduction, denoising, generation, and interpolation. Well-structured latent spaces exhibit key properties: similar data points cluster together, and semantic operations in data space correspond to arithmetic operations in latent space.

<div style="max-width: 800px; margin: 2rem auto;">
<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="300" fill="#f8fafc"/>
  
  <!-- High-dimensional space -->
  <g>
    <text x="100" y="30" font-size="16" font-weight="bold" fill="#1e293b">High-Dimensional Space (D=784)</text>
    <circle cx="100" cy="150" r="4" fill="#ef4444"/>
    <circle cx="120" cy="130" r="4" fill="#ef4444"/>
    <circle cx="85" cy="170" r="4" fill="#3b82f6"/>
    <circle cx="140" cy="160" r="4" fill="#3b82f6"/>
    <circle cx="110" cy="190" r="4" fill="#10b981"/>
    <circle cx="130" cy="180" r="4" fill="#10b981"/>
    <!-- More scattered points -->
    <circle cx="70" cy="140" r="3" fill="#ef4444" opacity="0.5"/>
    <circle cx="150" cy="145" r="3" fill="#3b82f6" opacity="0.5"/>
    <circle cx="95" cy="200" r="3" fill="#10b981" opacity="0.5"/>
  </g>
  
  <!-- Arrow -->
  <g>
    <path d="M 200 150 L 350 150" stroke="#6366f1" stroke-width="3" fill="none" marker-end="url(#arrowhead)"/>
    <text x="240" y="140" font-size="14" fill="#6366f1">Encoder</text>
    <text x="240" y="170" font-size="12" fill="#64748b">f_φ(x)</text>
  </g>
  
  <!-- Latent space -->
  <g>
    <text x="550" y="30" font-size="16" font-weight="bold" fill="#1e293b">Latent Space (d=2)</text>
    <!-- Organized clusters -->
    <ellipse cx="570" cy="120" rx="40" ry="30" fill="#ef4444" opacity="0.2"/>
    <ellipse cx="650" cy="150" rx="40" ry="30" fill="#3b82f6" opacity="0.2"/>
    <ellipse cx="600" cy="190" rx="40" ry="30" fill="#10b981" opacity="0.2"/>
    <!-- Points -->
    <circle cx="570" cy="120" r="5" fill="#ef4444"/>
    <circle cx="560" cy="125" r="5" fill="#ef4444"/>
    <circle cx="650" cy="150" r="5" fill="#3b82f6"/>
    <circle cx="655" cy="155" r="5" fill="#3b82f6"/>
    <circle cx="600" cy="190" r="5" fill="#10b981"/>
    <circle cx="605" cy="185" r="5" fill="#10b981"/>
  </g>
  
  <!-- Arrow marker definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  
  <!-- Legend -->
  <g transform="translate(300, 250)">
    <circle cx="0" cy="0" r="4" fill="#ef4444"/>
    <text x="10" y="5" font-size="12" fill="#64748b">Class 1</text>
    <circle cx="70" cy="0" r="4" fill="#3b82f6"/>
    <text x="80" y="5" font-size="12" fill="#64748b">Class 2</text>
    <circle cx="140" cy="0" r="4" fill="#10b981"/>
    <text x="150" y="5" font-size="12" fill="#64748b">Class 3</text>
  </g>
</svg>
<p style="text-align: center; color: #64748b; font-style: italic; margin-top: 0.5rem;">Figure 1: Dimensionality reduction from high-dimensional data to structured latent space</p>
</div>

---

## Autoencoders (AE) {#autoencoders}

An autoencoder consists of an encoder $f_\phi: \mathbb{R}^D \rightarrow \mathbb{R}^d$ and decoder $g_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^D$ trained to minimize reconstruction error:

$$
\mathcal{L}_{AE} = \mathbb{E}_{\mathbf{x} \sim p_{data}} [\| \mathbf{x} - g_\theta(f_\phi(\mathbf{x})) \|^2]
$$

The bottleneck dimension $d \ll D$ forces the network to learn a compressed representation $\mathbf{z} = f_\phi(\mathbf{x})$ that retains sufficient information for accurate reconstruction.

<div style="max-width: 900px; margin: 2rem auto;">
<svg viewBox="0 0 900 400" xmlns="http://www.w3.org/2000/svg">
  <rect width="900" height="400" fill="#ffffff"/>
  
  <!-- Input layer -->
  <g>
    <text x="50" y="30" font-size="14" font-weight="bold" fill="#1e293b">Input x</text>
    <text x="40" y="50" font-size="11" fill="#64748b">D=784</text>
    <rect x="30" y="70" width="80" height="250" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="4"/>
    <line x1="70" y1="90" x2="70" y2="290" stroke="#6366f1" stroke-width="1" stroke-dasharray="3,3"/>
  </g>
  
  <!-- Encoder arrow -->
  <path d="M 115 195 L 180 195" stroke="#6366f1" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="130" y="190" font-size="12" fill="#6366f1">Encode</text>
  
  <!-- Hidden layers (encoder) -->
  <rect x="185" y="120" width="60" height="150" fill="#ddd6fe" stroke="#8b5cf6" stroke-width="2" rx="4"/>
  <rect x="250" y="140" width="60" height="110" fill="#ddd6fe" stroke="#8b5cf6" stroke-width="2" rx="4"/>
  
  <!-- Latent space (bottleneck) -->
  <g>
    <text x="340" y="30" font-size="14" font-weight="bold" fill="#1e293b">Latent z</text>
    <text x="345" y="50" font-size="11" fill="#64748b">d=20</text>
    <rect x="315" y="160" width="50" height="70" fill="#fef3c7" stroke="#f59e0b" stroke-width="3" rx="4"/>
    <text x="328" y="200" font-size="16" fill="#78350f">z</text>
  </g>
  
  <!-- Hidden layers (decoder) -->
  <rect x="370" y="140" width="60" height="110" fill="#ddd6fe" stroke="#8b5cf6" stroke-width="2" rx="4"/>
  <rect x="435" y="120" width="60" height="150" fill="#ddd6fe" stroke="#8b5cf6" stroke-width="2" rx="4"/>
  
  <!-- Decoder arrow -->
  <path d="M 500 195 L 565 195" stroke="#8b5cf6" stroke-width="2" marker-end="url(#arrow2)"/>
  <text x="510" y="190" font-size="12" fill="#8b5cf6">Decode</text>
  
  <!-- Output layer -->
  <g>
    <text x="590" y="30" font-size="14" font-weight="bold" fill="#1e293b">Output x̂</text>
    <text x="585" y="50" font-size="11" fill="#64748b">D=784</text>
    <rect x="570" y="70" width="80" height="250" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" rx="4"/>
    <line x1="610" y1="90" x2="610" y2="290" stroke="#6366f1" stroke-width="1" stroke-dasharray="3,3"/>
  </g>
  
  <!-- Loss arrow -->
  <path d="M 325 340 Q 325 370, 410 370 Q 495 370, 495 340" stroke="#ef4444" stroke-width="2" fill="none"/>
  <text x="380" y="390" font-size="13" fill="#ef4444" font-weight="bold">Reconstruction Loss: ||x - x̂||²</text>
  
  <!-- Arrows -->
  <defs>
    <marker id="arrow1" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
    <marker id="arrow2" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#8b5cf6"/>
    </marker>
  </defs>
  
  <!-- Labels -->
  <text x="195" y="300" font-size="11" fill="#64748b">Encoder f_φ</text>
  <text x="380" y="300" font-size="11" fill="#64748b">Decoder g_θ</text>
</svg>
<p style="text-align: center; color: #64748b; font-style: italic; margin-top: 0.5rem;">Figure 2: Autoencoder architecture with encoder-decoder structure and latent bottleneck</p>
</div>

### Limitations for Generation

Standard autoencoders suffer from a critical limitation: the latent space is discontinuous and irregular. Each input maps deterministically to a single point $\mathbf{z}$, and the decoder is only trained on these specific points. Randomly sampling from the latent space produces poor results, as most regions remain unexplored during training. The model lacks a probabilistic framework to impose structure or quantify uncertainty.

<div style="max-width: 850px; margin: 2rem auto;">
<svg viewBox="0 0 850 350" xmlns="http://www.w3.org/2000/svg">
  <rect width="850" height="350" fill="#ffffff"/>
  
  <!-- Autoencoder latent space -->
  <g>
    <text x="150" y="30" font-size="15" font-weight="bold" fill="#1e293b">Standard Autoencoder</text>
    <text x="120" y="50" font-size="12" fill="#64748b">Disconnected, sparse latent space</text>
    <rect x="50" y="70" width="300" height="250" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="4"/>
    
    <!-- Scattered points (islands) -->
    <circle cx="100" cy="120" r="6" fill="#ef4444"/>
    <circle cx="110" cy="125" r="5" fill="#ef4444"/>
    <circle cx="200" cy="150" r="6" fill="#3b82f6"/>
    <circle cx="210" cy="155" r="5" fill="#3b82f6"/>
    <circle cx="280" cy="200" r="6" fill="#10b981"/>
    <circle cx="270" cy="205" r="5" fill="#10b981"/>
    
    <!-- Empty regions marked with X -->
    <text x="140" y="180" font-size="20" fill="#94a3b8">×</text>
    <text x="250" y="120" font-size="20" fill="#94a3b8">×</text>
    <text x="180" y="250" font-size="20" fill="#94a3b8">×</text>
    
    <text x="130" y="335" font-size="11" fill="#64748b">❌ Cannot sample random z</text>
  </g>
  
  <!-- VAE latent space -->
  <g>
    <text x="550" y="30" font-size="15" font-weight="bold" fill="#1e293b">Variational Autoencoder</text>
    <text x="520" y="50" font-size="12" fill="#64748b">Continuous, structured latent space</text>
    <rect x="450" y="70" width="300" height="250" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" rx="4"/>
    
    <!-- Continuous distributions -->
    <ellipse cx="510" cy="130" rx="35" ry="25" fill="#ef4444" opacity="0.3"/>
    <ellipse cx="600" cy="170" rx="40" ry="30" fill="#3b82f6" opacity="0.3"/>
    <ellipse cx="680" cy="220" rx="35" ry="28" fill="#10b981" opacity="0.3"/>
    
    <!-- Gradient overlays showing smooth transitions -->
    <ellipse cx="555" cy="150" rx="25" ry="20" fill="#fbbf24" opacity="0.2"/>
    <ellipse cx="640" cy="195" rx="30" ry="22" fill="#06b6d4" opacity="0.2"/>
    
    <!-- Sample points from distributions -->
    <circle cx="510" cy="130" r="4" fill="#ef4444"/>
    <circle cx="505" cy="135" r="3" fill="#ef4444" opacity="0.7"/>
    <circle cx="520" cy="128" r="3" fill="#ef4444" opacity="0.7"/>
    <circle cx="600" cy="170" r="4" fill="#3b82f6"/>
    <circle cx="595" cy="175" r="3" fill="#3b82f6" opacity="0.7"/>
    <circle cx="610" cy="168" r="3" fill="#3b82f6" opacity="0.7"/>
    <circle cx="680" cy="220" r="4" fill="#10b981"/>
    <circle cx="675" cy="225" r="3" fill="#10b981" opacity="0.7"/>
    <circle cx="688" cy="218" r="3" fill="#10b981" opacity="0.7"/>
    
    <text x="530" y="335" font-size="11" fill="#64748b">✓ Can sample z ~ N(0,I)</text>
  </g>
  
  <!-- Comparison arrow -->
  <path d="M 360 200 L 440 200" stroke="#6366f1" stroke-width="3" marker-end="url(#arrow3)"/>
  <defs>
    <marker id="arrow3" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
</svg>
<p style="text-align: center; color: #64748b; font-style: italic; margin-top: 0.5rem;">Figure 3: Comparison of latent space structure in Autoencoders vs VAEs</p>
</div>

**Variational Autoencoders** address these limitations by introducing a probabilistic framework that ensures continuous, well-structured latent spaces suitable for generation.

---

## Variational Autoencoders (VAE) {#variational-autoencoders}

A **Variational Autoencoder (VAE)** is a generative model that learns a probabilistic mapping between data and latent spaces. Unlike standard autoencoders, VAEs:
- Encode inputs as probability distributions rather than single points
- Impose a prior distribution on the latent space
- Use variational inference to learn the posterior distribution
- Generate new samples by sampling from the learned prior

### Probabilistic Framework

VAEs treat both the encoder and decoder as probabilistic models:

#### Generative Process (Decoder)

The VAE assumes the following generative story for how data is created:

1. Sample a latent code from a prior distribution: $$\mathbf{z} \sim p(\mathbf{z})$$
2. Generate data from a conditional distribution: $$\mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z})$$

The prior is typically chosen to be a standard Gaussian:

$
p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})
$

The decoder network parameterizes $p_\theta(\mathbf{x}|\mathbf{z})$ , often as:

$
p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_\theta(\mathbf{z}), \boldsymbol{\sigma}_\theta^2(\mathbf{z})\mathbf{I})
$

![VAE Architecture](https://via.placeholder.com/900x400/6c5ce7/ffffff?text=VAE+Architecture+with+Reparameterization)  
*Figure 2: Variational Autoencoder architecture showing the probabilistic encoder and decoder*

#### Inference Model (Encoder)

The true posterior $$p_\theta(\mathbf{z}|\mathbf{x})$$ is intractable to compute directly. Instead, we introduce an **approximate posterior** (also called the **recognition model** or **inference network**):

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x})\mathbf{I})$$

where the encoder network outputs the parameters $$\boldsymbol{\mu}_\phi(\mathbf{x})$$ and $$\boldsymbol{\sigma}_\phi^2(\mathbf{x})$$ of a Gaussian distribution.

### Goal: Maximum Likelihood Estimation

Our ultimate goal is to maximize the likelihood of the observed data:

$$\max_\theta \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})} [\log p_\theta(\mathbf{x})]$$

or equivalently:

$$\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)})$$

### The Marginal Likelihood

The marginal likelihood (or evidence) is obtained by marginalizing over all possible latent codes:

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

Taking the logarithm:

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

### The Intractability Problem

This integral is **intractable** for several reasons:

1. **High Dimensionality**: If $$\mathbf{z} \in \mathbb{R}^d$$, we need to integrate over a $$d$$-dimensional space
2. **Complex Decoder**: When $$p_\theta(\mathbf{x}|\mathbf{z})$$ is parameterized by a neural network, the integral has no closed form
3. **Posterior Computation**: Computing $$p_\theta(\mathbf{z}|\mathbf{x})$$ requires knowing $$p_\theta(\mathbf{x})$$, creating a circular dependency

The solution is to derive a lower bound on the log likelihood that is tractable to optimize. This is called the **Evidence Lower Bound (ELBO)**.

---

## Evidence Lower Bound - Derivation 1 (Jensen's Inequality) {#elbo-jensen}

The first derivation uses Jensen's inequality to establish that the ELBO is indeed a lower bound on the log likelihood.

### Step 1: Introduce the Approximate Posterior

Starting with the log marginal likelihood:

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

We multiply and divide by the approximate posterior $$q_\phi(\mathbf{z}|\mathbf{x})$$:

$$\log p_\theta(\mathbf{x}) = \log \int \frac{q_\phi(\mathbf{z}|\mathbf{x})}{q_\phi(\mathbf{z}|\mathbf{x})} p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

### Step 2: Express as an Expectation

Rearranging:

$$\log p_\theta(\mathbf{x}) = \log \int q_\phi(\mathbf{z}|\mathbf{x}) \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} d\mathbf{z}$$

This is equivalent to:

$$\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})} \left[ \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

### Step 3: Apply Jensen's Inequality

Jensen's inequality states that for a concave function $$f$$ (and $$\log$$ is concave):

$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$

For the logarithm:

$$\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$$

Applying this:

$$\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})} \left[ \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] \geq \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

### Step 4: Define the ELBO

The right-hand side is the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

Therefore:

$$\log p_\theta(\mathbf{x}) \geq \mathcal{L}_{\theta,\phi}(\mathbf{x})$$

### Key Insight

The ELBO is a **lower bound** on the log likelihood. Maximizing the ELBO pushes up the log likelihood. However, this derivation doesn't reveal:
- How tight the bound is
- What happens when we maximize the ELBO
- The relationship between $q_\phi$ and the true posterior $p_\theta(\mathbf{z}|\mathbf{x})$

For these insights, we need an alternative derivation.

---

## Evidence Lower Bound - Derivation 2 (Decomposition) {#elbo-alternative}

This derivation provides more intuition about what the ELBO represents and why maximizing it makes sense.

### Step 1: Start with the Log Likelihood

We begin with:

$$\log p_\theta(\mathbf{x}) = \log p_\theta(\mathbf{x}) \int q_\phi(\mathbf{z}|\mathbf{x}) d\mathbf{z}$$

This is valid because $$\int q_\phi(\mathbf{z}|\mathbf{x}) d\mathbf{z} = 1$$ (it's a probability distribution).

### Step 2: Move Inside the Integral

$$\log p_\theta(\mathbf{x}) = \int q_\phi(\mathbf{z}|\mathbf{x}) \log p_\theta(\mathbf{x}) d\mathbf{z} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x})]$$

### Step 3: Apply Bayes' Rule

Using the definition of conditional probability:

$$p_\theta(\mathbf{x}) = \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})} = \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})}$$

Therefore:

$$\log p_\theta(\mathbf{x}) = \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})} = \log p_\theta(\mathbf{x}, \mathbf{z}) - \log p_\theta(\mathbf{z}|\mathbf{x})$$

### Step 4: Substitute into the Expectation

$$\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}, \mathbf{z}) - \log p_\theta(\mathbf{z}|\mathbf{x})]$$

### Step 5: Introduce the Approximate Posterior

Multiply and divide by $$q_\phi(\mathbf{z}|\mathbf{x})$$:

$$\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] + \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

### Step 6: Recognize the Components

The first term is the **ELBO**:

$$\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

The second term is the **KL divergence** between $$q_\phi(\mathbf{z}|\mathbf{x})$$ and $$p_\theta(\mathbf{z}|\mathbf{x})$$:

$$D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

### Step 7: Final Decomposition

$$\boxed{\log p_\theta(\mathbf{x}) = \mathcal{L}_{\theta,\phi}(\mathbf{x}) + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))}$$

### Key Insights from This Derivation

1. **The ELBO is a Lower Bound**: Since $$D_{KL} \geq 0$$ always:
   $$\mathcal{L}_{\theta,\phi}(\mathbf{x}) \leq \log p_\theta(\mathbf{x})$$

2. **Tightness of the Bound**: The gap between the ELBO and the log likelihood is exactly the KL divergence. The better $$q_\phi(\mathbf{z}|\mathbf{x})$$ approximates the true posterior $$p_\theta(\mathbf{z}|\mathbf{x})$$, the tighter the bound.

3. **Joint Optimization**: Maximizing the ELBO w.r.t. $$\phi$$ minimizes the KL divergence, making $$q_\phi$$ a better approximation to the true posterior. Maximizing w.r.t. $$\theta$$ improves the model's ability to explain the data.

4. **Why This Works**: We cannot compute $$p_\theta(\mathbf{z}|\mathbf{x})$$ directly, but we don't need to! By maximizing the ELBO, we simultaneously:
   - Push up the log likelihood $$\log p_\theta(\mathbf{x})$$
   - Make $$q_\phi(\mathbf{z}|\mathbf{x})$$ closer to $$p_\theta(\mathbf{z}|\mathbf{x})$$

---

## ELBO as Reconstruction + Regularization

We can rewrite the ELBO in an alternative form that provides intuition for what the VAE is doing:

$$\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

Using $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})$:

$$\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] + \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})} \right]$$

$$= \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction Term}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization Term}}$$

Therefore:

$$\boxed{\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}$$

### Interpretation

**Reconstruction Term**: $$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})]$$
- Measures how well the decoder reconstructs the input
- Sample $$\mathbf{z}$$ from the encoder's distribution $$q_\phi(\mathbf{z}|\mathbf{x})$$
- Evaluate how likely the decoder thinks the original $$\mathbf{x}$$ is given $$\mathbf{z}$$
- Similar to the reconstruction loss in standard autoencoders
- Encourages the latent code to contain information about the input

**Regularization Term**: $$D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$
- Measures how different the approximate posterior is from the prior
- Regularizes the latent space to follow the prior distribution $$p(\mathbf{z})$$
- Prevents the encoder from cheating by encoding each input to an arbitrary region
- Ensures that random samples from $$p(\mathbf{z})$$ can be decoded to realistic data
- Encourages the latent space to be smooth and continuous

### The Trade-off

Maximizing the ELBO requires balancing these two objectives:
- High reconstruction: Encode all information, but latent space may become irregular
- Low KL divergence: Follow the prior closely, but may lose information

The VAE finds the optimal trade-off that maximizes $\mathcal{L}_{\theta,\phi}(\mathbf{x})$.

---

## VAE Optimization {#vae-optimization}

Now that we have established the ELBO as our training objective, let's examine how to optimize it.

### Training Objective

The VAE training objective is to maximize the expected ELBO over the data distribution:

$$\max_{\theta, \phi} \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})} [\mathcal{L}_{\theta,\phi}(\mathbf{x})]$$

In practice, we work with a finite dataset $$\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}\}$$ and use mini-batch gradient descent:

$$\max_{\theta, \phi} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\theta,\phi}(\mathbf{x}^{(i)})$$

### Computing the Reconstruction Term

The reconstruction term is:

$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})]$$

To estimate this expectation, we use Monte Carlo sampling:

1. Sample $$\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$$
2. Compute $$\log p_\theta(\mathbf{x}|\mathbf{z})$$
3. Average over multiple samples (often just 1 sample suffices)

$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] \approx \frac{1}{L} \sum_{l=1}^{L} \log p_\theta(\mathbf{x}|\mathbf{z}^{(l)})$$

where $$\mathbf{z}^{(l)} \sim q_\phi(\mathbf{z}|\mathbf{x})$$.

#### Gaussian Decoder

If we model the decoder as:

$$p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 \mathbf{I})$$

Then:

$$\log p_\theta(\mathbf{x}|\mathbf{z}) = -\frac{D}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2$$

If we assume constant variance $$\sigma^2$$ or set it to 1, this reduces to minimizing the **mean squared error (MSE)**:

$$-\log p_\theta(\mathbf{x}|\mathbf{z}) \propto \|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2$$

#### Bernoulli Decoder

For binary data (e.g., black and white images), we model each dimension independently as a Bernoulli distribution:

$$p_\theta(\mathbf{x}|\mathbf{z}) = \prod_{i=1}^{D} \text{Bernoulli}(x_i; p_i)$$

where $$p_i = \boldsymbol{\mu}_\theta(\mathbf{z})_i$$ (the decoder output after sigmoid activation).

The log likelihood is:

$$\log p_\theta(\mathbf{x}|\mathbf{z}) = \sum_{i=1}^{D} [x_i \log p_i + (1-x_i) \log(1-p_i)]$$

This is the **binary cross-entropy (BCE)** loss.

### Computing the KL Divergence

The regularization term is:

$$D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

With the choices:
- Prior: $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Approximate posterior: $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x})\mathbf{I})$

The KL divergence between two multivariate Gaussians has a **closed-form solution**:

For $$q = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$ and $$p = \mathcal{N}(\mathbf{0}, \mathbf{I})$$, where $$\boldsymbol{\Sigma}$$ is diagonal:

$$D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{d} \left( \mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1 \right)$$

#### Derivation of KL Divergence (Gaussian Case)

The KL divergence between two distributions is:

$$D_{KL}(q \| p) = \mathbb{E}_{q} \left[ \log \frac{q(\mathbf{z})}{p(\mathbf{z})} \right] = \mathbb{E}_{q} [\log q(\mathbf{z})] - \mathbb{E}_{q} [\log p(\mathbf{z})]$$

For $$q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$$:

$$\log q(\mathbf{z}) = -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^{d} \log \sigma_j^2 - \frac{1}{2} \sum_{j=1}^{d} \frac{(z_j - \mu_j)^2}{\sigma_j^2}$$

For $$p = \mathcal{N}(\mathbf{0}, \mathbf{I})$$:

$$\log p(\mathbf{z}) = -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^{d} z_j^2$$

Taking expectations and simplifying (the full derivation is tedious but straightforward):

$$\mathbb{E}_{q} [\log q(\mathbf{z})] = -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^{d} \log \sigma_j^2 - \frac{d}{2}$$

$$\mathbb{E}_{q} [\log p(\mathbf{z})] = -\frac{d}{2} \log(2\pi) - \frac{1}{2} \sum_{j=1}^{d} (\mu_j^2 + \sigma_j^2)$$

Subtracting:

$$D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{d} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

This is computationally efficient and **requires no sampling**!

### Complete Loss Function

Putting it all together, the VAE loss for a single data point is:

$$\mathcal{L}_{VAE}(\mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

Note the sign change: we typically minimize loss rather than maximize the ELBO.

In practice, for a mini-batch of size $$M$$:

$$\mathcal{L}_{VAE} = \frac{1}{M} \sum_{i=1}^{M} \left[ \underbrace{\|\mathbf{x}^{(i)} - \text{Decoder}(\mathbf{z}^{(i)})\|^2}_{\text{Reconstruction Loss}} + \underbrace{\frac{1}{2} \sum_{j=1}^{d} (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)}_{\text{KL Divergence}} \right]$$

where $$\mathbf{z}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}^{(i)}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}^{(i)})\mathbf{I})$$.

---

## The Reparameterization Trick {#reparameterization}

There's a critical problem with the loss function as stated: **how do we backpropagate through a random sampling operation**?

### The Problem

The encoder outputs $$\boldsymbol{\mu}_\phi(\mathbf{x})$$ and $$\boldsymbol{\sigma}_\phi(\mathbf{x})$$, and we need to sample:

$$\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x})\mathbf{I})$$

Then we pass $$\mathbf{z}$$ to the decoder. The problem is: **sampling is not differentiable**. We cannot compute gradients through a random operation.

### Naive Approach (Doesn't Work)

If we try to compute:

$$\frac{\partial \mathcal{L}}{\partial \phi}$$

we encounter:

$$\frac{\partial \mathcal{L}}{\partial \phi} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \frac{\partial \mathbf{z}}{\partial \phi}$$

But $$\mathbf{z}$$ is sampled randomly, so $$\frac{\partial \mathbf{z}}{\partial \phi}$$ is undefined!

### The Reparameterization Trick

The key insight is to **separate the randomness from the parameters**.

Instead of sampling:
$$\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2\mathbf{I})$$

We rewrite it as:
1. Sample $$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$  (pure randomness, independent of parameters)
2. Transform: $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$  (deterministic transformation)

where $$\odot$$ denotes element-wise multiplication.

### Why This Works

**Mathematically**: The reparameterization is valid because if $$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$, then $$\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2\mathbf{I})$$. This is a property of Gaussian distributions:

- $$\mathbb{E}[\mathbf{z}] = \mathbb{E}[\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}] = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \mathbb{E}[\boldsymbol{\epsilon}] = \boldsymbol{\mu}$$
- $$\text{Var}[\mathbf{z}] = \text{Var}[\boldsymbol{\sigma} \odot \boldsymbol{\epsilon}] = \boldsymbol{\sigma}^2 \odot \text{Var}[\boldsymbol{\epsilon}] = \boldsymbol{\sigma}^2$$

**Computationally**: Now the randomness comes from $$\boldsymbol{\epsilon}$$, which is independent of $$\phi$$ and $$\theta$$. The transformation $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$ is fully differentiable:

$$\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = \mathbf{I}, \quad \frac{\partial \mathbf{z}}{\partial \boldsymbol{\sigma}} = \text{diag}(\boldsymbol{\epsilon})$$

Gradients can now flow through $$\boldsymbol{\mu}$$ and $$\boldsymbol{\sigma}$$ to the encoder parameters $$\phi$$.

### Implementation

In PyTorch:

```python
def reparameterize(mu, log_var):
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1)
    
    Args:
        mu: Mean of the latent Gaussian [batch_size x latent_dim]
        log_var: Log variance of the latent Gaussian [batch_size x latent_dim]
    
    Returns:
        z: Sampled latent vector [batch_size x latent_dim]
    """
    std = torch.exp(0.5 * log_var)  # std = sqrt(var) = exp(0.5 * log_var)
    eps = torch.randn_like(std)     # Sample epsilon from N(0, 1)
    z = mu + eps * std              # z = mu + std * epsilon
    return z
```

**Note**: We typically work with `log_var` instead of `var` or `std` for numerical stability. The exponential ensures positivity of variance.

### Computational Graph

```
Input x
    |
Encoder
    |
    +---> mu ------+
    |              |
    +---> log_var -+---> std = exp(0.5 * log_var)
                   |
         eps ~ N(0,1)
                   |
                   +---> z = mu + std * eps
                              |
                           Decoder
                              |
                         Reconstruction x_hat
```

The path from encoder parameters to the loss goes through **mu** and **std**, which are differentiable operations, even though **eps** is random.

---

## Training Procedure {#training}

### Algorithm: Training a VAE

**Input:** Dataset $\mathcal{D} = \{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}\}$, batch size $M$, learning rate $\alpha$

**Initialize:** Encoder parameters $$\phi$$, decoder parameters $$\theta$$

**Repeat until convergence:**

1. Sample a mini-batch $$\{\mathbf{x}^{(i)}\}_{i=1}^{M}$$ from $$\mathcal{D}$$

2. **Forward pass:**
   - For each $\mathbf{x}^{(i)}$ in the mini-batch:
     - Encode: $\boldsymbol{\mu}^{(i)}, \log \boldsymbol{\sigma}^{2(i)} = \text{Encoder}_\phi(\mathbf{x}^{(i)})$
     - Sample: $\boldsymbol{\epsilon}^{(i)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
     - Reparameterize: $\mathbf{z}^{(i)} = \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\epsilon}^{(i)}$
     - Decode: $\hat{\mathbf{x}}^{(i)} = \text{Decoder}_\theta(\mathbf{z}^{(i)})$

3. **Compute loss:**
   $$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \left[ \text{ReconstructionLoss}(\mathbf{x}^{(i)}, \hat{\mathbf{x}}^{(i)}) + \text{KL}(\boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}) \right]$$
   
   where:
   - Reconstruction: $\|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|^2$ or BCE
   - KL: $\frac{1}{2} \sum_{j=1}^{d} \left( (\mu_j^{(i)})^2 + (\sigma_j^{(i)})^2 - \log(\sigma_j^{(i)})^2 - 1 \right)$

4. **Backward pass:**
   - Compute gradients: $\nabla_\theta \mathcal{L}$, $\nabla_\phi \mathcal{L}$
   - Update parameters: 
     - $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$
     - $\phi \leftarrow \phi - \alpha \nabla_\phi \mathcal{L}$

### Hyperparameters and Training Tips

1. **Learning Rate**: Typical values: $10^{-4}$ to $10^{-3}$ with Adam optimizer

2. **Latent Dimensionality**: Start with 2D for visualization, use 10-100 for complex datasets

3. **Architecture**: 
   - For images: Use convolutional encoders/decoders
   - For tabular data: Use fully connected layers

4. **Beta-VAE**: Add a weight $\beta$ to the KL term:
   $$\mathcal{L}_{\beta-VAE} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$
   
   - $$\beta < 1$$: Prioritize reconstruction
   - $$\beta > 1$$: Encourage disentangled representations
   - $$\beta = 1$$: Standard VAE

5. **KL Annealing**: Gradually increase the weight of the KL term from 0 to 1 during training to prevent "posterior collapse"

6. **Posterior Collapse**: The encoder ignores the input and always outputs the prior. This happens when the decoder is too powerful. Solutions:
   - KL annealing
   - Free bits: Allow a minimum amount of KL divergence per dimension
   - Architectural choices: Don't make the decoder too powerful

---

## Implementation Details {#implementation}

### Architecture for MNIST

For the MNIST dataset (28×28 grayscale images), we use a convolutional architecture.

#### Encoder Network

```
Input: [batch_size, 1, 28, 28]
    ↓
Conv2d(1, 32, kernel=3, stride=2, padding=1)  → [batch_size, 32, 14, 14]
ReLU
    ↓
Conv2d(32, 64, kernel=3, stride=2, padding=1) → [batch_size, 64, 7, 7]
ReLU
    ↓
Flatten → [batch_size, 64*7*7 = 3136]
    ↓
Linear(3136, 256)
ReLU
    ↓
    +---> Linear(256, latent_dim) → mu
    |
    +---> Linear(256, latent_dim) → log_var
```

#### Decoder Network

```
Input z: [batch_size, latent_dim]
    ↓
Linear(latent_dim, 256)
ReLU
    ↓
Linear(256, 64*7*7)
ReLU
    ↓
Reshape → [batch_size, 64, 7, 7]
    ↓
ConvTranspose2d(64, 32, kernel=3, stride=2, padding=1, output_padding=1) → [batch_size, 32, 14, 14]
ReLU
    ↓
ConvTranspose2d(32, 1, kernel=3, stride=2, padding=1, output_padding=1) → [batch_size, 1, 28, 28]
Sigmoid (to get values in [0, 1])
```

### Loss Computation in PyTorch

```python
def vae_loss(x, x_recon, mu, log_var):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        x: Original input [batch_size, *]
        x_recon: Reconstructed input [batch_size, *]
        mu: Latent mean [batch_size, latent_dim]
        log_var: Latent log variance [batch_size, latent_dim]
    
    Returns:
        loss: Total VAE loss (scalar)
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (binary cross-entropy for MNIST)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    loss = recon_loss + kl_loss
    
    return loss, recon_loss, kl_loss
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        
        # Forward pass
        x_recon, mu, log_var = model(data)
        
        # Compute loss
        loss, recon, kl = vae_loss(data, x_recon, mu, log_var)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()
    
    # Average over dataset
    num_samples = len(dataloader.dataset)
    return train_loss / num_samples, train_recon / num_samples, train_kl / num_samples
```

---

## Experiments on MNIST {#experiments}

### Dataset

**MNIST** (Modified National Institute of Standards and Technology database):
- 60,000 training images
- 10,000 test images
- 28×28 grayscale images of handwritten digits (0-9)
- Pixel values normalized to [0, 1]

### Experimental Setup

- **Model**: Convolutional VAE
- **Latent Dimension**: 2 (for visualization) and 20 (for generation quality)
- **Optimizer**: Adam with learning rate 1e-3
- **Batch Size**: 128
- **Training Epochs**: 20
- **Hardware**: GPU (CUDA if available)

### Results

#### 1. Reconstruction Quality

After training, the VAE can accurately reconstruct input images. The reconstruction quality depends on:
- Latent dimension size (higher = better reconstruction)
- Training duration
- Architecture capacity

**Expected reconstruction error** (per pixel, MSE): ~0.01-0.03

#### 2. Latent Space Visualization (2D)

With a 2D latent space, we can visualize the learned manifold:

**Scatter Plot of Encoded Digits:**
- Each point represents a digit encoded to $$\mathbf{z} \in \mathbb{R}^2$$
- Different digits form distinct clusters
- Similar digits (like 4 and 9) are closer together
- The distribution approximately follows $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$ due to the KL regularization

**Latent Space Traversal:**
- Sample a grid of points from the 2D latent space
- Decode each point to generate an image
- Visualize as a 2D grid showing smooth transitions between digit types

#### 3. Generation

Sample $$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$ and decode:

$$\hat{\mathbf{x}} = \text{Decoder}_\theta(\mathbf{z})$$

The generated images are:
- Realistic looking digits
- Smooth and well-formed (not noisy)
- Diverse (capturing all digit types)

#### 4. Interpolation

Given two images $$\mathbf{x}_1$$ and $$\mathbf{x}_2$$:

1. Encode to latent: $$\mathbf{z}_1 = \boldsymbol{\mu}_\phi(\mathbf{x}_1)$$, $$\mathbf{z}_2 = \boldsymbol{\mu}_\phi(\mathbf{x}_2)$$
2. Interpolate: $$\mathbf{z}_t = (1-t) \mathbf{z}_1 + t \mathbf{z}_2$$ for $$t \in [0, 1]$$
3. Decode: $$\hat{\mathbf{x}}_t = \text{Decoder}_\theta(\mathbf{z}_t)$$

Result: A smooth morphing from one digit to another, demonstrating the continuity of the latent space.

---

## Comparison: VAE vs Standard Autoencoder {#comparison}

Let's highlight the key differences experimentally.

### Standard Autoencoder

**Architecture:** Same encoder/decoder structure, but:
- Encoder outputs a single vector $$\mathbf{z}$$ (not $$\boldsymbol{\mu}$$ and $$\boldsymbol{\sigma}$$)
- No reparameterization
- No KL divergence term

**Loss:** Only reconstruction loss
$$\mathcal{L}_{AE} = \|\mathbf{x} - \text{Decoder}(\text{Encoder}(\mathbf{x}))\|^2$$

### Key Differences

| Aspect | Autoencoder (AE) | Variational Autoencoder (VAE) |
|--------|------------------|-------------------------------|
| **Latent Representation** | Deterministic point $\mathbf{z}$ | Probability distribution $q_\phi(\mathbf{z}\|\mathbf{x})$ |
| **Training Objective** | Reconstruction only | Reconstruction + KL regularization |
| **Latent Space** | Irregular, discontinuous | Smooth, continuous, structured |
| **Generation** | Poor (undefined regions) | Good (sample from $p(\mathbf{z})$) |
| **Interpolation** | May produce artifacts | Smooth transitions |
| **Reconstruction** | Slightly better | Slightly worse (due to regularization) |
| **Probabilistic** | No | Yes |

### Experimental Comparison

#### Latent Space Structure

**AE**: Encoded points cluster tightly, but with gaps. Random points in latent space may decode to nonsense.

**VAE**: Encoded points spread out to fill the space. The KL term encourages coverage of the unit Gaussian. Random samples decode to realistic images.

#### Generation Quality

**AE**: Sampling random $$\mathbf{z}$$ often produces blurry or unrealistic images, because most regions of latent space are "unmapped."

**VAE**: Sampling $$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$ consistently produces recognizable digits.

#### Reconstruction Fidelity

**AE**: Slightly sharper reconstructions (no regularization pressure).

**VAE**: Slightly blurrier (trades off reconstruction for smooth latent space).

### When to Use Which?

**Use Standard Autoencoder when:**
- You only care about compression/reconstruction
- You don't need to generate new samples
- You want maximum reconstruction quality
- Dimensionality reduction is the goal (like PCA++)

**Use VAE when:**
- You want to generate new samples
- You need interpolation or smooth latent space
- You want probabilistic representations
- You need a principled generative model

---

## Conclusion {#conclusion}

### Summary

We've covered the complete theory and practice of Variational Autoencoders:

1. **Latent Representations**: The foundation of modern deep learning, capturing abstract structure of data

2. **Autoencoders**: Deterministic encoder-decoder networks that learn compressed representations but lack generative capabilities

3. **VAEs**: Probabilistic generative models that learn smooth latent spaces by:
   - Modeling data generation as $p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})$
   - Learning an approximate posterior $q_\phi(\mathbf{z}|\mathbf{x})$
   - Maximizing the Evidence Lower Bound (ELBO)

4. **ELBO**: A tractable lower bound on the log likelihood:
   $$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

5. **Reparameterization Trick**: Makes VAEs trainable via backpropagation by separating randomness from parameters

6. **Implementation**: Convolutional architecture for images, careful loss computation, standard gradient-based optimization

### Key Insights

- **Probabilistic Thinking**: VAEs show how probabilistic modeling leads to better generative models
- **Variational Inference**: Approximate intractable posteriors with learned distributions
- **Trade-offs**: Balance reconstruction quality with latent space regularity
- **Continuous Latent Spaces**: Enable interpolation, smooth generation, and meaningful arithmetic

### Limitations of VAEs

Despite their elegance, VAEs have some limitations:

1. **Blurry Reconstructions**: The Gaussian assumption and KL term can lead to over-smoothed outputs
2. **Mode Collapse**: May not capture all modes of a multi-modal distribution
3. **Difficult to Train**: Balancing reconstruction and KL can be tricky
4. **Amortization Gap**: The encoder may not perfectly invert the decoder

### Extensions and Advanced Topics

Modern research has developed many VAE variants:

- **$$\beta$$-VAE**: Disentangled representations via $$\beta > 1$$
- **VQ-VAE**: Discrete latent spaces using vector quantization
- **Hierarchical VAEs**: Multiple layers of latent variables
- **Conditional VAEs**: Condition on labels or other information
- **Importance Weighted VAEs (IWAE)**: Tighter bounds via multiple samples
- **Normalizing Flows**: More expressive approximate posteriors

### Beyond VAEs

VAEs paved the way for modern generative models:

- **GANs (Generative Adversarial Networks)**: Adversarial training for sharp images
- **Diffusion Models**: Current state-of-the-art for image generation (DALL-E 2, Stable Diffusion)
- **Transformers**: Autoregressive models for text and images (GPT, DALL-E)

However, VAEs remain important for their:
- Theoretical elegance
- Stable training
- Explicit likelihood
- Latent space interpretability

---

## References {#references}

### Key Papers

1. **Kingma & Welling (2013)**: "Auto-Encoding Variational Bayes"
   - Original VAE paper introducing the reparameterization trick
   - [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

2. **Rezende, Mohamed & Wierstra (2014)**: "Stochastic Backpropagation and Approximate Inference in Deep Generative Models"
   - Independent discovery of VAEs
   - [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)

3. **Higgins et al. (2017)**: "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
   - Disentangled representations
   - [ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl)

4. **Bowman et al. (2016)**: "Generating Sentences from a Continuous Space"
   - VAEs for text generation
   - [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)

5. **Van Den Oord et al. (2017)**: "Neural Discrete Representation Learning"
   - VQ-VAE for discrete latents
   - [NeurIPS 2017](https://arxiv.org/abs/1711.00937)

### Textbooks and Tutorials

1. **Deep Learning (Goodfellow, Bengio & Courville)** - Chapter 20: Deep Generative Models
2. **Pattern Recognition and Machine Learning (Bishop)** - Variational Inference
3. **Probabilistic Machine Learning: Advanced Topics (Murphy)** - VAEs and Deep Generative Models

### Online Resources

1. [Tutorial on Variational Autoencoders - Carl Doersch](https://arxiv.org/abs/1606.05908)
2. [Understanding VAEs - Jaan Altosaar](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
3. [From Autoencoder to Beta-VAE - Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)

---

**End of Tutorial**

For hands-on implementation and experiments, see the accompanying Jupyter notebook `VAE_Tutorial.ipynb` and source code in the `src/` directory.

Questions or feedback? Contact: [Your Email]

**License:** MIT

