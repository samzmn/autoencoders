import tensorflow as tf
import keras._tf_keras.keras as keras

from utils import DenseTranspose, Conv2DTransposeTied, KLDivergenceRegularizer
#--------------------------------------------------------------------------------
# --------------- Stacked Autoencoders -----------------------------------------
#--------------------------------------------------------------------------------
class StackedAutoencoder(keras.models.Model):
    def __init__(self, input_shape=[28, 28], latent_dim=30, **kwargs):
        super(StackedAutoencoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
            keras.layers.Dense(latent_dim, activation="relu", kernel_initializer="he_normal"),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Input(shape=[latent_dim]),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(28 * 28, activation="sigmoid"),
            keras.layers.Reshape(input_shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class TiedStackedAutoencoder(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.original_input_shape = input_shape

        self.dense_1 = tf.keras.layers.Dense(
            100, activation="relu", kernel_initializer="he_normal"
        )
        self.dense_2 = tf.keras.layers.Dense(
            latent_dim, activation="relu", kernel_initializer="he_normal"
        )

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            self.dense_1,
            self.dense_2,
        ])

        self.encoder.build((None, *input_shape))

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            DenseTranspose(self.dense_2, activation="relu"),
            DenseTranspose(self.dense_1),
            tf.keras.layers.Reshape(input_shape),
        ])

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class DenoiseStackedAutoencoder(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal")
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
            tf.keras.layers.Reshape(input_shape)
        ])

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class SparseStackedAutoencoder(keras.models.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)
        self.encoder

#--------------------------------------------------------------------------------
# --------------- Recurrent Autoencoders ---------------------------------------
#--------------------------------------------------------------------------------
class RecurrentAutoencoder(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.LSTM(latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.RepeatVector(28),
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.Dense(28)
        ])
    
    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
#--------------------------------------------------------------------------------
# --------------- Convolutional Autoencoders -----------------------------------
#--------------------------------------------------------------------------------

class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Reshape([28, 28, 1]),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.MaxPool2D(pool_size=2),  # output: 14 × 14 x 16
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.MaxPool2D(pool_size=2),  # output: 7 × 7 x 32
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.MaxPool2D(pool_size=2),  # output: 3 × 3 x 64
            tf.keras.layers.Conv2D(latent_dim, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.GlobalAvgPool2D()  # output: 30
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(3 * 3 * 16),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same"),
            tf.keras.layers.Reshape([28, 28])
        ])

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class TiedConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, latent_channels=16):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = tf.keras.layers.Conv2D(
            32, 3, strides=2, padding="same",
            activation="relu", kernel_initializer="he_normal"
        )
        self.enc2 = tf.keras.layers.Conv2D(
            64, 3, strides=2, padding="same",
            activation="relu", kernel_initializer="he_normal"
        )
        self.enc3 = tf.keras.layers.Conv2D(
            latent_channels, 3, strides=2, padding="same",
            activation="relu", kernel_initializer="he_normal"
        )

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            self.enc1,   # 28 → 14
            self.enc2,   # 14 → 7
            self.enc3,   # 7  → 4
        ])

        # -------- Decoder (tied) --------
        self.decoder = tf.keras.Sequential([
            Conv2DTransposeTied(self.enc3, activation="relu"),   # 4 → 7
            Conv2DTransposeTied(self.enc2, activation="relu"),   # 7 → 14
            Conv2DTransposeTied(self.enc1, activation="sigmoid") # 14 → 28
        ])

    def call(self, x):
        # Accept (28, 28) or (28, 28, 1)
        if x.shape.rank == 3:
            x = tf.expand_dims(x, -1)

        z = self.encoder(x)
        y = self.decoder(z)

        # Return (28, 28)
        return tf.squeeze(y, axis=-1)


class DenoiseConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.GaussianNoise(0.2),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same', strides=2, kernel_initializer="he_normal"), # output: 14 × 14 x 16
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer="he_normal"), # output: 7 × 7 x 32
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer="he_normal"), # output: 3 × 3 x 64
            tf.keras.layers.Conv2D(latent_dim, 3, activation="relu", padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.GlobalAvgPool2D()  # output: 30
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[latent_dim]),
            tf.keras.layers.Dense(3 * 3 * 16),
            tf.keras.layers.Reshape((3, 3, 16)),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', kernel_initializer="he_normal"),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same', kernel_initializer="he_normal"),
            tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', padding='same'),
            tf.keras.layers.Reshape(input_shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    pass