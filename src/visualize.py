"""
Visualizing the Reconstructions
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras._tf_keras.keras as keras

from datasets import load_fashion_mnist, load_cifar10
import models

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

def save_fig(fig_id, base_path=Path() / "images", tight_layout=True, fig_extension="png", resolution=300):
    base_path.mkdir(parents=True, exist_ok=True)
    path = base_path / f"{fig_id}.{fig_extension}"
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

def plot_rgb_reconstructions(model, images, n_images=5, noise=0.1):
    n_images = 5
    new_images = images[:n_images]
    new_images_noisy = new_images + np.random.randn(n_images, 32, 32, 3) * noise
    new_images_denoised = model.predict(new_images_noisy)
    new_images_preicted = model.predict(new_images)

    plt.figure(figsize=(8, n_images * 2))
    for index in range(n_images):
        plt.subplot(n_images, 4, index * 4 + 1)
        plt.imshow(new_images[index])
        plt.axis('off')
        if index == 0:
            plt.title("Original")
        plt.subplot(n_images, 4, index * 4 + 2)
        plt.imshow(new_images_noisy[index].clip(0., 1.))
        plt.axis('off')
        if index == 0:
            plt.title("Noisy")
        plt.subplot(n_images, 4, index * 4 + 3)
        plt.imshow(new_images_denoised[index])
        plt.axis('off')
        if index == 0:
            plt.title("Denoised")
        plt.subplot(n_images, 4, index * 4 + 4)
        plt.imshow(new_images_preicted[index])
        plt.axis('off')
        if index == 0:
            plt.title("Predicted")

def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = images.squeeze(axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

def generate_new_imgs(model, codings_size, n_images=21):
    codings = tf.random.normal(shape=[n_images, codings_size])
    images = model.decoder(codings).numpy()
    plot_multiple_images(images, 7)

def semantic_interpolation(model, codings_size,):
    codings = np.zeros([7, codings_size])
    codings[:, 3] = np.linspace(-0.8, 0.8, 7)  # axis 3 looks best in this case
    images = model.decoder(codings).numpy()
    plot_multiple_images(images)

def visualize_stacked_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/stacked_autoencoder_{tag}.keras", 
                                    custom_objects={"StackedAutoencoder": models.StackedAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("stacked_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_tied_stacked_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/tied_stacked_autoencoder_{tag}.keras", 
                                    custom_objects={"TiedStackedAutoencoder": models.TiedStackedAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("tied_stacked_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_sparse_stacked_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/sparse_stacked_autoencoder_{tag}.keras", 
                                    custom_objects={"SparseStackedAutoencoder": models.SparseStackedAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("sparse_stacked_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visulize_conv_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/convolutional_autoencoder_{tag}.keras", 
                                    custom_objects={"ConvolutionalAutoencoder": models.ConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("convolutional_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_tied_conv_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/tied_convolutional_autoencoder_{tag}.keras", 
                                    custom_objects={"TiedConvolutionalAutoencoder": models.TiedConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("tied_convolutional_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_denoise_conv_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/denoise_convolutional_autoencoder_{tag}.keras", 
                                    custom_objects={"DenoiseConvolutionalAutoencoder": models.DenoiseConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("denoise_convolutional_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_sparse_conv_ae(base_path=".", tag="fashion"):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    model = keras.models.load_model(f"{base_path}/saved_models/sparse_convolutional_autoencoder_{tag}.keras", 
                                    custom_objects={"SparseConvolutionalAutoencoder": models.SparseConvolutionalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig("sparse_convolutional_autoencoder_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_denoise_sparse_conv_ae(base_path=".", tag=""):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_cifar10()
    model = keras.models.load_model(f"{base_path}/saved_models/denoise_sparse_convolutional_autoencoder_{tag}.keras", 
                                    custom_objects={"DenoiseSparseConvolutionalAutoencoder": models.DenoiseSparseConvolutionalAutoencoder})
    plot_rgb_reconstructions(model, X_test, noise=0.2)
    save_fig(f"denoise_sparse_convolutional_autoencoder_{tag}_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_denoise_sparse_tied_conv_ae(base_path=".", tag=""):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_cifar10()
    model = keras.models.load_model(f"{base_path}/saved_models/denoise_sparse_tied_convolutional_autoencoder_{tag}.keras", 
                                    custom_objects={"DenoiseSparseTiedConvolutionalAutoencoder": models.DenoiseSparseTiedConvolutionalAutoencoder})
    plot_rgb_reconstructions(model, X_test, noise=0.2)
    save_fig(f"denoise_sparse_tied_convolutional_autoencoder_{tag}_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_conv_var_ae(base_path=".", tag=""):
    if tag=="fashion":
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    else: # "cifar"
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_cifar10()
    model = keras.models.load_model(f"{base_path}/saved_models/convolutional_variational_autoencoder_{tag}.keras", 
                                    custom_objects={"ConvolutionalVariationalAutoencoder": models.ConvolutionalVariationalAutoencoder})
    plot_reconstructions(model, X_test)
    save_fig(f"convolutional_variational_autoencoder_{tag}_reconstruction_plot", base_path=Path(base_path) / "images")

def visualize_new_generated_conv_var_ae(base_path=".", tag=""):
    model = keras.models.load_model(f"{base_path}/saved_models/convolutional_variational_autoencoder_{tag}.keras", 
                                    custom_objects={"ConvolutionalVariationalAutoencoder": models.ConvolutionalVariationalAutoencoder})
    generate_new_imgs(model, codings_size=10)
    save_fig(f"convolutional_variational_autoencoder_{tag}_generated", tight_layout=False, base_path=Path(base_path) / "images")
    semantic_interpolation(model, codings_size=10)
    save_fig(f"convolutional_variational_autoencoder_{tag}_semantic_interpolation", tight_layout=False, base_path=Path(base_path) / "images")


if __name__=="__main__":
    pass
