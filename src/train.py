import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras

import datasets
import models
import visualize

def get_default_callbacks(model_name: str) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=f'./ckpt/{model_name}/checkpoint.weights.h5', monitor="val_loss", save_best_only=True, save_weights_only=True),
        keras.callbacks.TensorBoard(log_dir=f'./runs/{model_name}_fashion_mnist', histogram_freq=5, write_steps_per_second=True),
        keras.callbacks.CSVLogger(filename=f'./logs/{model_name}.csv', append=False),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_delta=1e-4, cooldown=2, min_lr=1e-6),
    ]

def train_stacked_autoencoder(X_train, X_valid, X_test, epochs=20, save_plots=False):
    stacked_ae = models.StackedAutoencoder()
    stacked_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="stacked_ae")

    stacked_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    stacked_ae.evaluate(X_test, X_test)
    keras.models.save_model(stacked_ae, "./saved_models/stacked_autoencoder.keras")
    if save_plots:
        visualize.plot_reconstructions(stacked_ae, X_test)
        visualize.save_fig("stacked_autoencoder_reconstruction_plot")
    return stacked_ae

def train_tied_stacked_autoencoder(X_train, X_valid, X_test, epochs=20, save_plots=False):
    tied_stacked_ae = models.TiedStackedAutoencoder()
    tied_stacked_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="tied_stacked_ae")

    tied_stacked_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    tied_stacked_ae.evaluate(X_test, X_test)
    keras.models.save_model(tied_stacked_ae, "./saved_models/tied_stacked_autoencoder.keras")
    if save_plots:
        visualize.plot_reconstructions(tied_stacked_ae, X_test)
        visualize.save_fig("tied_stacked_autoencoder_reconstruction_plot")
    return tied_stacked_ae

def train_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, save_plots=False):
    conv_ae = models.ConvolutionalAutoencoder()
    conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="conv_ae")

    conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(conv_ae, "./saved_models/convolutional_autoencoder.keras")
    if save_plots:
        visualize.plot_reconstructions(conv_ae, X_test)
        visualize.save_fig("convolutional_autoencoder_reconstruction_plot")
    return conv_ae

def train_tied_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, save_plots=False):
    tied_conv_ae = models.TiedConvolutionalAutoencoder()
    tied_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="tied_conv_ae")

    tied_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    tied_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(tied_conv_ae, "./saved_models/tied_convolutional_autoencoder.keras")
    if save_plots:
        visualize.plot_reconstructions(tied_conv_ae, X_test)
        visualize.save_fig("tied_convolutional_autoencoder_reconstruction_plot")
    return tied_conv_ae

def train_denoise_convolutional_autoencoder(X_train, X_valid, X_test, epochs=20, save_plots=False):
    denoise_conv_ae = models.DenoiseConvolutionalAutoencoder()
    denoise_conv_ae.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.Huber())
    callbacks = get_default_callbacks(model_name="denoise_conv_ae")

    denoise_conv_ae.fit(X_train, X_train, epochs=epochs,
                    validation_data=(X_valid, X_valid),
                    callbacks=callbacks)
    denoise_conv_ae.evaluate(X_test, X_test)
    keras.models.save_model(denoise_conv_ae, "./saved_models/denoise_convolutional_autoencoder.keras")
    if save_plots:
        visualize.plot_reconstructions(denoise_conv_ae, X_test)
        visualize.save_fig("denoise_convolutional_autoencoder_reconstruction_plot")
    return denoise_conv_ae

if __name__=="__main__":
    tf.random.set_seed(42)
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = datasets.load_fashion_mnist()
    # train_stacked_autoencoder(X_train, X_valid, X_test)
    train_tied_stacked_autoencoder(X_train, X_valid, X_test)
    
