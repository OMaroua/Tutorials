# Assets Directory

This directory contains all the images and visualizations generated from the VAE experiments.

## Required Images

Please add the following images from your experiments to this directory:

### 2D Latent Space Experiments
- **latent2D.png** - 2D manifold grid showing generated digits across latent space
- **Clusters2D.png** - Scatter plot of 2D latent encodings colored by digit class

### 3D Latent Space Experiments
- **Clusters3D.png** - 3D scatter plot of latent encodings
- **2D3DClusters.png** - Pairwise 2D projections of the 3D latent space
- **3Dlatent1.png** - 2D manifold slice at z₂ = -1.0
- **3DLatent2.png** - 2D manifold slice at z₂ = 0.0
- **3DLatent3.png** - 2D manifold slice at z₂ = 1.0

### Correlated Prior Experiments
- **KLAdapted.png** - Manifold with correlated prior showing tilted structure
- **Screenshot 2025-11-07 at 14.41.06.png** - Latent clusters under correlated prior

## Image Guidelines

For best results:
- Use PNG format for crisp visualizations
- Recommended resolution: 1200x1200 pixels for manifold grids
- Use high DPI (300) for publication-quality images
- Ensure good contrast for readability on dark backgrounds

## Generating Images

Refer to the Python scripts in the `code/` directory to regenerate these visualizations from your trained models.

