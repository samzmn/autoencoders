import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras

import importlib

import train
import datasets
import visualize

def r():
    importlib.reload(train)

def main():
    tf.random.set_seed(42)  # ensures reproducibility on CPU

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = datasets.load_fashion_mnist()

    # train.train_stacked_autoencoder(X_train, X_valid, X_test)
    # train.train_tied_stacked_autoencoder(X_train, X_valid, X_test)
    # train.train_convolutional_autoencoder(X_train, X_valid, X_test)
    # train.train_denoise_convolutional_autoencoder(X_train, X_valid, X_test)
    # visualize.visulaize_denoise_conv_ae()
    train.train_tied_convolutional_autoencoder(X_train, X_valid, X_test)
    visualize.visulaize_tied_conv_ae()

if __name__ == "__main__":
    main()