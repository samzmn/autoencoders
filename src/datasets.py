import os
from typing import Tuple
import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras

# Default cache path -> ~/.keras/datasets/
# os.environ["KERAS_HOME"] = "/mnt/c/Users/samzm/keras_cache"

def load_fashion_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. This dataset can be used as a drop-in replacement for MNIST.

    The classes are:
    Label	Description
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot
    """
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    X_train_full = np.reshape(X_train_full, shape=[-1, 28, 28, 1])
    X_test = np.reshape(X_test, shape=[-1, 28, 28, 1])
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. See more info at the CIFAR homepage.

    The classes are:
    Label	Description
    0	airplane
    1	automobile
    2	bird
    3	cat
    4	deer
    5	dog
    6	frog
    7	horse
    8	ship
    9	truck
    """
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train_full = X_train_full.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_cifar100() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    This dataset is just like the CIFAR-10 (50,000 32x32 color training images and 10,000 test images), except it has 100 classes containing 600 images each. 
    There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. 
    Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
    """
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
    X_train_full = X_train_full.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

if __name__=="__main__":
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_fashion_mnist()
    print(f"fashion SHAPE: x_train -> {X_train.shape}, x_valid -> {X_valid.shape}, x_test -> {X_test.shape}")
    print(f"fashion SHAPE: y_train -> {y_train.shape}, y_valid -> {y_valid.shape}, y_test -> {y_test.shape}")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_cifar10()
    print(f"fashion SHAPE: x_train -> {X_train.shape}, x_valid -> {X_valid.shape}, x_test -> {X_test.shape}")
    print(f"fashion SHAPE: y_train -> {y_train.shape}, y_valid -> {y_valid.shape}, y_test -> {y_test.shape}")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_cifar100()
    print(f"fashion SHAPE: x_train -> {X_train.shape}, x_valid -> {X_valid.shape}, x_test -> {X_test.shape}")
    print(f"fashion SHAPE: y_train -> {y_train.shape}, y_valid -> {y_valid.shape}, y_test -> {y_test.shape}")
