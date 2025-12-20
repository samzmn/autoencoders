import os
import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras

# os.environ["KERAS_HOME"] = "/mnt/c/Users/samzm/keras_cache"

def load_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist.load_data() # It is cached under ~/.keras/datasets/fashion-mnist/
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
    X_train_full = X_train_full.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
