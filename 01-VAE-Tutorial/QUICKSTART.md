# ⚡ Quick Start Guide

Get up and running with the VAE tutorial in 5 minutes!

## 📖 View the Tutorial

### Option 1: Interactive Website (Recommended)

1. Open `index.html` in your browser
2. Navigate through sections using the top menu
3. Enjoy interactive features and beautiful visualizations

```bash
# macOS
open index.html

# Linux
xdg-open index.html

# Windows
start index.html
```

### Option 2: GitHub README

Read `README.md` - fully formatted with math equations on GitHub.

---

## 💻 Run the Code

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv vae_env

# Activate it
source vae_env/bin/activate  # macOS/Linux
# OR
vae_env\Scripts\activate     # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Experiments

```bash
# Navigate to code directory
cd code

# Run 2D VAE (basic)
python vae_2d.py

# Run 3D VAE (extended)
python vae_3d.py

# Run Correlated Prior VAE (advanced)
python vae_correlated.py
```

---

## 📊 What to Expect

Each script will:
1. ✅ Download MNIST dataset (automatic, first time only)
2. ✅ Build and train the model (~10-15 minutes on CPU)
3. ✅ Generate visualizations in `assets/` directory
4. ✅ Save trained models in `code/models/`
5. ✅ Print training metrics

### Output Files

After running `vae_2d.py`:
```
assets/
  ├── Clusters2D.png           # Latent space clustering
  ├── latent2D.png            # 2D manifold grid
  └── reconstructions_2d.png  # Input vs output

code/models/
  ├── encoder_2d.h5
  ├── decoder_2d.h5
  └── vae_2d_weights.h5
```

---

## 🎯 Expected Results

### Training Metrics (2D VAE)
```
Epoch 30/30
469/469 [======] - 15s - loss: 165.23 - reconstruction_loss: 161.11 - kl_loss: 3.91
```

### Visualizations
- **Clusters**: Clear separation of digit classes in 2D space
- **Manifold**: Smooth grid showing generated digits
- **Reconstructions**: Slight blurriness is normal for VAEs

---

## 🎨 View Your Results

```bash
# Open assets folder
cd assets
open .  # macOS
nautilus .  # Linux
explorer .  # Windows
```

Your generated images will be here! Compare them with examples in the tutorial.

---

## ❓ Having Issues?

### "Module not found"
→ Make sure virtual environment is activated and requirements installed

### "GPU not found" (Optional - not required)
→ Install TensorFlow GPU: `pip install tensorflow-gpu`

### Out of memory
→ Reduce batch size in the script (line ~200): `batch_size=64`

### Poor results
→ Train longer: change `epochs=30` to `epochs=50`

---

## 📚 Learn More

- Read the full tutorial in `README.md`
- Check `TUTORIAL_README.md` for detailed documentation
- Explore `code/README.md` for implementation details

---

## 🎉 You're All Set!

Now you can:
- 🔬 Experiment with different architectures
- 🎨 Generate custom visualizations
- 📊 Analyze latent space properties
- 🚀 Extend to other datasets

**Enjoy learning about VAEs!** ⭐

