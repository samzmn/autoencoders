"""
Visualizing the Reconstructions
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras._tf_keras.keras as keras

from datasets import load_fashion_mnist
import models

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_reconstructions(model, images, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")

def visulaize_stacked_ae():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model("./saved_models/stacked_autoencoder.keras", custom_objects={"StackedAutoencoder": models.StackedAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("stacked_autoencoder_reconstruction_plot")

def visualize_tied_stacked_ae():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model("./saved_models/tied_stacked_autoencoder.keras", custom_objects={"TiedStackedAutoencoder": models.TiedStackedAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("tied_stacked_autoencoder_reconstruction_plot")

def visualize_sparce_stacked_ae(base_path="."):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/sparse_stacked_autoencoder.keras", custom_objects={"SparseStackedAutoencoder": models.SparseStackedAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("tied_stacked_autoencoder_reconstruction_plot")

def visulize_conv_ae():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model("./saved_models/convolutional_autoencoder.keras", custom_objects={"ConvolutionalAutoencoder": models.ConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("convolutional_autoencoder_reconstruction_plot")

def visulaize_tied_conv_ae():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model("./saved_models/tied_convolutional_autoencoder.keras", custom_objects={"tiedConvolutionalAutoencoder": models.TiedConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("tied_convolutional_autoencoder_reconstruction_plot")

def visulaize_denoise_conv_ae():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model("./saved_models/denoise_convolutional_autoencoder.keras", custom_objects={"DenoiseConvolutionalAutoencoder": models.DenoiseConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("denoise_convolutional_autoencoder_reconstruction_plot")

if __name__=="__main__":
    # visulaize_stacked_ae()
    # visualize_tied_stacked_ae()
    visulize_conv_ae()

