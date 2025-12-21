import tensorflow as tf
import keras._tf_keras.keras as keras

def get_default_callbacks(model_name: str, ckpt_path: str="./ckpt", tensorboard_path: str="./runs", csv_log_dir: str=".logs") -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=f'{ckpt_path}/{model_name}/checkpoint.weights.h5', monitor="val_loss", save_best_only=True, save_weights_only=True),
        keras.callbacks.TensorBoard(log_dir=f'{tensorboard_path}/{model_name}_fashion_mnist', histogram_freq=5, write_steps_per_second=True),
        keras.callbacks.CSVLogger(filename=f'{csv_log_dir}/{model_name}.csv', append=False),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_delta=1e-4, cooldown=2, min_lr=1e-6),
    ]

class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = tf.reduce_mean(inputs, axis=0)
        return self.weight * (
            tf.keras.losses.kullback_leibler_divergence(self.target, mean_activities) + 
            tf.keras.losses.kullback_leibler_divergence(1. - self.taget, 1. - mean_activities)
        )

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
    
class Conv2DTransposeTied(tf.keras.layers.Layer):
    def __init__(self, conv_layer, strides=2, padding="same", activation=None):
        super().__init__()
        self.conv_layer = conv_layer
        self.strides = strides
        self.padding = padding.upper()
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        if not self.conv_layer.built:
            raise ValueError("Encoder Conv2D layer must be built first.")

        # Encoder kernel: (kh, kw, Cin, Cout)
        kh, kw, cin, cout = self.conv_layer.kernel.shape

        # Decoder output channels = encoder input channels
        self.bias = self.add_weight(
            name="bias",
            shape=(cin,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x):
        # Flip spatial dims + swap channels
        kernel = tf.reverse(self.conv_layer.kernel, axis=[0, 1])
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 2])

        batch = tf.shape(x)[0]
        h = tf.shape(x)[1] * self.strides
        w = tf.shape(x)[2] * self.strides
        out_ch = kernel.shape[-1]

        output_shape = tf.stack([batch, h, w, out_ch])

        x = tf.nn.conv2d_transpose(
            x,
            kernel,
            output_shape=output_shape,
            strides=[1, self.strides, self.strides, 1],
            padding=self.padding,
        )

        x = tf.nn.bias_add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x