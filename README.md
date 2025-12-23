# Autoencoders Collection

A comprehensive implementation of various autoencoder architectures using TensorFlow/Keras, featuring multiple variants including convolutional, variational, sparse, and denoising autoencoders.

## 🚀 Features

This repository contains implementations of the following autoencoder types:

### Stacked Autoencoders
- **Stacked Autoencoder**: Basic multi-layer autoencoder with dense layers
- **Tied Stacked Autoencoder**: Weight-sharing between encoder and decoder layers
- **Sparse Stacked Autoencoder**: Regularized with KL divergence for sparsity

### Convolutional Autoencoders
- **Convolutional Autoencoder**: CNN-based encoder-decoder architecture
- **Tied Convolutional Autoencoder**: Weight-sharing in convolutional layers
- **Sparse Convolutional Autoencoder**: Sparsity regularization on latent representations
- **Denoising Convolutional Autoencoder**: Trained to remove noise from input images
- **Denoise Sparse Convolutional Autoencoder**: Combines denoising and sparsity
- **Denoise Sparse Tied Convolutional Autoencoder**: All three techniques combined

### Variational Autoencoders
- **Convolutional Variational Autoencoder**: Probabilistic generative model with convolutional architecture

## 📊 Datasets

The models are trained and evaluated on:
- **Fashion MNIST**: 28×28 grayscale fashion images (10 classes)
- **CIFAR-10**: 32×32 color images (10 classes)

## 🏗️ Architecture Overview

Each autoencoder consists of:
- **Encoder**: Compresses input data into a lower-dimensional latent space
- **Decoder**: Reconstructs the original input from the latent representation

### Key Techniques Implemented
- **Weight Tying**: Sharing weights between encoder and decoder for parameter efficiency
- **Sparsity Regularization**: KL divergence penalty to encourage sparse latent activations
- **Denoising**: Adding noise to inputs during training for robust feature learning
- **Variational Inference**: Probabilistic latent space with reparameterization trick

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/samzmn/autoencoders.git
cd autoencoders
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Training a Model

```python
import tensorflow as tf
from src import train, datasets

# Load dataset
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = datasets.load_fashion_mnist()

# Train a convolutional autoencoder
train.train_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20)
```

### Available Training Functions

- `train_stacked_autoencoder()`
- `train_tied_stacked_autoencoder()`
- `train_sparse_stacked_autoencoder()`
- `train_convolutional_autoencoder()`
- `train_tied_convolutional_autoencoder()`
- `train_denoise_convolutional_autoencoder()`
- `train_sparse_convolutional_autoencoder()`
- `train_denoise_sparse_convolutional_autoencoder()`
- `train_denoise_sparse_tied_convolutional_autoencoder()`
- `train_convolutional_variational_autoencoder()`

### Visualization

```python
from src import visualize

# Visualize reconstructions
visualize.visulize_conv_ae()
```

## 📁 Project Structure

```
autoencoders/
├── src/
│   ├── main.py              # Main execution script
│   ├── train.py             # Training functions for all models
│   ├── models.py            # Autoencoder model definitions
│   ├── datasets.py          # Data loading utilities
│   ├── visualize.py         # Visualization and plotting functions
│   ├── utils.py             # Utility classes and functions
│   └── experiment.ipynb     # Jupyter notebook for experimentation
├── ckpt/                    # Model checkpoints
├── saved_models/            # Saved trained models
├── runs/                    # TensorBoard logs
├── logs/                    # Training metrics (CSV)
├── images/                  # Generated visualizations
└── requirements.txt         # Python dependencies
```

## 🔧 Key Components

### Custom Layers and Regularizers
- `DenseTranspose`: Tied dense layer for weight sharing
- `Conv2DTransposeTied`: Tied convolutional transpose layer
- `KLDivergenceRegularizer`: Sparsity regularization
- `Sampling`: Reparameterization for variational autoencoders

### Training Features
- Early stopping with patience
- Model checkpointing (best validation loss)
- TensorBoard logging
- CSV logging of metrics
- Learning rate scheduling
- GPU acceleration support

## 📈 Results and Visualizations

The repository includes visualization functions to:
- Compare original vs reconstructed images
- Visualize denoising capabilities
- Generate latent space representations
- Plot training metrics and losses

Sample reconstruction results are saved in the `images/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Sam Zamani**  
[sam.zmn99@gmail.com](mailto:your.email@example.com)  
[https://linkedin.com/in/sam-zmn](https://linkedin.com/in/sam-zmn)  
[https://github.com/samzmn](https://github.com/samzmn)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Variational Autoencoder (Kingma & Welling)](https://arxiv.org/abs/1312.6114)
- [Contractive Autoencoders (Rifai et al.)](https://www.sciencedirect.com/science/article/pii/S0925231212000931)

## 🏷️ Tags

`autoencoders`, `deep-learning`, `tensorflow`, `keras`, `computer-vision`, `unsupervised-learning`, `generative-models`, `dimensionality-reduction`