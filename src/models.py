import tensorflow as tf
import keras._tf_keras.keras as keras

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

class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        if not self.dense.built:
            raise ValueError(
                "The tied Dense layer must be built before DenseTranspose."
            )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.dense.kernel.shape[0],),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.kernel, transpose_b=True)
        if self.activation is not None:
            z = self.activation(z + self.bias)
        else:
            z = z + self.bias
        return z
    
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
    
class DenoiseConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, input_shape=(28, 28), latent_dim=30, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Reshape((28, 28, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer="he_normal"),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer="he_normal")
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    pass