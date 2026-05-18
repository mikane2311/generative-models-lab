# Generative Models Lab

Deep Generative Models in PyTorch: Autoencoders, Variational Autoencoders, and GANs.

## Overview

This project implements and compares three fundamental deep generative architectures on the MNIST dataset:

| Model | Architecture | Key Feature | Best For |
|-------|-------------|-------------|----------|
| **Autoencoder (AE)** | Encoder-Decoder with bottleneck | Lossy compression | Dimensionality reduction |
| **Variational Autoencoder (VAE)** | Probabilistic latent space | Smooth, generatable manifold | Image generation |
| **Generative Adversarial Network (GAN)** | Generator + Discriminator game | Sharp, realistic outputs | High-quality synthesis |

## Project Structure
generative-models-lab/

├── notebooks/

│   └── generative_models_lab.ipynb    # Main Colab notebook

├── src/

│   ├── config.py                      # Hyperparameters & device setup

│   ├── ae.py                          # Autoencoder architecture

│   ├── vae.py                         # VAE with ELBO loss

│   ├── gan.py                         # Generator & Discriminator

│   ├── train.py                       # Training loops

│   └── utils.py                       # Visualization helpers

├── results/                             # Generated images & loss curves

├── models/                              # Saved model checkpoints

├── report/                              # Written analysis

└── docs/                                # Additional documentation


## Setup

### Local Development


# Clone repository
git clone https://github.com/mikane2311/generative-models-lab.git
cd generative-models-lab

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
### Google Colab (GPU Training)

1. Upload `notebooks/generative_models_lab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: `Runtime` → `Change runtime type` → `GPU`
3. Mount Google Drive for persistent storage

## Key Results

### Quantitative Comparison

| Model | Test MSE | Notes |
|-------|----------|-------|
| AE | 6.35 | Best reconstruction, unstructured latent space |
| VAE | 9.11 | Slightly worse reconstruction, smooth generatable manifold |

### Training Observations

- **AE**: Fast convergence, clear reconstructions, clustered but gappy latent space
- **VAE**: Successfully generates digits from `N(0,I)`, β-VAE shows recon/KL trade-off
- **GAN**: Reaches near-Nash equilibrium (D loss ≈ 0.65), mode collapse demonstrated with high LR

## Technologies

- PyTorch 2.11
- torchvision 0.26
- matplotlib 3.10
- NumPy 2.2
- Jupyter Notebook

## Author

- Data Engineering Student


## License

MIT
