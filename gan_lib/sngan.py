'''
    Spectral Normalization for Generative Adversarial Networks
    Ref:
        - https://arxiv.org/abs/1802.05957
        - https://github.com/pfnet-research/sngan_projection/tree/master/source
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

def build_generator(input_shape):
    x = Input(input_shape)

    y = Conv2DTranspose(512, (3, 3), strides=(2,2), padding="same")(x)
    y = LeakyReLU(0.2)(y)

    y = Conv2DTranspose(256, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv2DTranspose(128, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv2DTranspose(64, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv2D(3, (3, 3), padding="same", activation="tanh")(y)
    return Model(x, y)

def build_discriminator(input_shape):
    x = Input(input_shape)

    y = SNConv2D(64, (3, 3), strides=(2,2), padding="same")(x)
    y = LeakyReLU(0.2)(y)

    y = SNConv2D(128, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = SNConv2D(256, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = SNConv2D(512, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = GlobalAveragePooling2D()(y)
    y = SNDense(1)(y)
    return Model(x, y)

def build_train_step(generator, discriminator):
    d_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)
    g_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_step(real_image, noise):

        # set for training updates of SNConv2D and SNDense
        tf.keras.backend.set_learning_phase(True)

        fake_image = generator(noise)
        pred_real, pred_fake = tf.split(discriminator(tf.concat([real_image, fake_image], axis=0)), num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(tf.maximum(0., 1 - pred_real)) + tf.reduce_mean(tf.maximum(0., 1 + pred_fake))
        g_loss = -tf.reduce_mean(pred_fake)

        d_gradients = tf.gradients(d_loss, discriminator.trainable_variables)
        g_gradients = tf.gradients(g_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        return d_loss, g_loss

    return train_step

class SNConv2D(Conv2D):
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format=None,
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # kernel_constraint=None,
            bias_constraint=None,
            power_iterations = 1,
            **kwargs):

        super(SNConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=self.spectrally_norm,
            bias_constraint=bias_constraint,
            **kwargs)

        assert power_iterations>=1, "The number of power iterations should be positive integer"

        self.Ip = power_iterations
        self.u = self.add_weight(
            name='W_u',
            shape=(1, filters),
            initializer='random_uniform',
            trainable=False
        )

    def spectrally_norm(self, w):
        W_mat = tf.transpose(w, (3, 2, 0, 1))  # (h, w, i, o) => (o, i, h, w)
        W_mat = tf.reshape(W_mat, [tf.shape(W_mat)[0], -1])  # (o, i * h * w)

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = self.l2_norm(tf.matmul(_u, W_mat))
            _u = self.l2_norm(tf.matmul(_v, W_mat, transpose_b=True))

        sigma = tf.reduce_sum(tf.matmul(_u, W_mat) * _v)

        self.u.assign(tf.keras.backend.in_train_phase(_u, self.u))
        return w / (sigma + 1e-8)

    def l2_norm(self, x):
        return x / tf.sqrt(tf.reduce_sum(tf.square(x)) + 1e-8)

class SNDense(Dense):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # kernel_constraint=None,
            bias_constraint=None,
            power_iterations = 1,
            **kwargs):

        super(SNDense, self).__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=self.spectrally_norm,
            bias_constraint=bias_constraint,
            **kwargs)

        assert power_iterations>=1, "The number of power iterations should be positive integer"

        self.Ip = power_iterations
        self.u = self.add_weight(
            name='W_u',
            shape=(1, units),
            initializer='random_uniform',
            trainable=False
        )

    def spectrally_norm(self, w):
        W_mat = tf.transpose(w, (1, 0))  # (i, o) => (o, i)

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = self.l2_norm(tf.matmul(_u, W_mat))
            _u = self.l2_norm(tf.matmul(_v, W_mat, transpose_b=True))

        sigma = tf.reduce_sum(tf.matmul(_u, W_mat) * _v)

        self.u.assign(tf.keras.backend.in_train_phase(_u, self.u))
        return w / (sigma + 1e-8)

    def l2_norm(self, x):
        return x / tf.sqrt(tf.reduce_sum(tf.square(x)) + 1e-8)