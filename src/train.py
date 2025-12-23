import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras

import datasets
import models
import visualize
from utils import get_default_callbacks


def train_stacked_autoencoder(X_train, X_valid, X_test, epochs=20, base_path: str=".", tag: str=""):
    stacked_ae = models.StackedAutoencoder()
    stacked_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="stacked_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    stacked_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    stacked_ae.evaluate(X_test, X_test)
    keras.models.save_model(stacked_ae, f"{base_path}/saved_models/stacked_autoencoder_{tag}.keras")

def train_tied_stacked_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    tied_stacked_ae = models.TiedStackedAutoencoder()
    tied_stacked_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="tied_stacked_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    tied_stacked_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    tied_stacked_ae.evaluate(X_test, X_test)
    keras.models.save_model(tied_stacked_ae, f"{base_path}/saved_models/tied_stacked_autoencoder_{tag}.keras")

def train_saprse_stacked_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    sparse_stacked_ae = models.SparseStackedAutoencoder(input_shape=(28, 28), latent_dim=30, sparsity_loss_weight=5e-3, sparsity_target=0.1)
    sparse_stacked_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="sparse_stacked_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    sparse_stacked_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    sparse_stacked_ae.evaluate(X_test, X_test)
    keras.models.save_model(sparse_stacked_ae, f"{base_path}/saved_models/sparse_stacked_autoencoder_{tag}.keras")

def train_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    conv_ae = models.ConvolutionalAutoencoder()
    conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="conv_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(conv_ae, f"{base_path}/saved_models/convolutional_autoencoder_{tag}.keras")

def train_tied_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    tied_conv_ae = models.TiedConvolutionalAutoencoder()
    tied_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="tied_conv_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    tied_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    tied_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(tied_conv_ae, f"{base_path}/saved_models/tied_convolutional_autoencoder_{tag}.keras")

def train_denoise_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    denoise_conv_ae = models.DenoiseConvolutionalAutoencoder()
    denoise_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="denoise_conv_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    denoise_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    denoise_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(denoise_conv_ae, f"{base_path}/saved_models/denoise_convolutional_autoencoder_{tag}.keras")

def train_saprse_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    sparse_conv_ae = models.SparseConvolutionalAutoencoder(input_shape=(28, 28), latent_channels=10, sparsity_loss_weight=5e-3, sparsity_target=0.1)
    sparse_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="sparse_conv_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    sparse_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    sparse_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(sparse_conv_ae, f"{base_path}/saved_models/sparse_convolutional_autoencoder_{tag}.keras")

def train_denoise_saprse_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    denoise_sparse_conv_ae = models.DenoiseSparseConvolutionalAutoencoder(input_shape=(32, 32, 3), latent_channels=10,
                                                                          gaussian_noise=0.2, sparsity_loss_weight=5e-3, sparsity_target=0.1)
    denoise_sparse_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="denoise_sparse_conv_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    denoise_sparse_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    denoise_sparse_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(denoise_sparse_conv_ae, f"{base_path}/saved_models/denoise_sparse_convolutional_autoencoder_{tag}.keras")

def train_denoise_saprse_tied_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    denoise_sparse_tied_conv_ae = models.DenoiseSparseTiedConvolutionalAutoencoder(input_shape=(32, 32, 3), latent_channels=10,
                                                                          gaussian_noise=0.2, sparsity_loss_weight=5e-3, sparsity_target=0.1)
    denoise_sparse_tied_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="denoise_sparse_tied_conv_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    denoise_sparse_tied_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    denoise_sparse_tied_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(denoise_sparse_tied_conv_ae, f"{base_path}/saved_models/denoise_sparse_tied_convolutional_autoencoder_{tag}.keras")

def train_convolutional_variational_autoencoder(X_train, X_valid, X_test, epochs=20, base_path:str=".", tag: str=""):
    convolutional_variational_ae = models.ConvolutionalVariationalAutoencoder(input_shape=(32, 32, 3), codings_size=10)
    convolutional_variational_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.huber)
    callbacks = get_default_callbacks(model_name="convolutional_variational_ae", tag=tag, ckpt_path=f"{base_path}/ckpt", tensorboard_path=f"{base_path}/runs", csv_log_dir=f"{base_path}/logs")

    convolutional_variational_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    convolutional_variational_ae.evaluate(X_test, X_test)
    keras.models.save_model(convolutional_variational_ae, f"{base_path}/saved_models/convolutional_variational_autoencoder_{tag}.keras")

if __name__=="__main__":
    tf.random.set_seed(42)
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = datasets.load_fashion_mnist()
    # train_stacked_autoencoder(X_train, X_valid, X_test)
    # train_tied_stacked_autoencoder(X_train, X_valid, X_test)
    
