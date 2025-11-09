# 📊 Data Folder

This folder will contain datasets used in the VAE tutorial.

## 📥 **Datasets**

The tutorial uses the following datasets, which are automatically downloaded:

### **MNIST**
- **Description**: Handwritten digits (0-9)
- **Size**: 60,000 training images, 10,000 test images
- **Format**: 28x28 grayscale images
- **Download**: Automatic via `torchvision.datasets.MNIST`

### **Fashion-MNIST**
- **Description**: Fashion items (t-shirts, trousers, bags, etc.)
- **Size**: 60,000 training images, 10,000 test images
- **Format**: 28x28 grayscale images
- **Download**: Automatic via `torchvision.datasets.FashionMNIST`

## 📁 **Data Structure**

After running the tutorial, this folder will contain:

```
data/
├── MNIST/
│   ├── raw/
│   └── processed/
└── FashionMNIST/
    ├── raw/
    └── processed/
```

## 🚫 **Note**

Data files are excluded from git tracking (see `.gitignore`). They will be automatically downloaded when you first run the tutorial.

---

**Need help?** Check the main tutorial [README](../README.md)

