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
from tensorflow.keras.constraints import Constraint

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

    y = Conv2D(64, (3, 3), kernel_constraint=SpectralNorm2D(64), strides=(2, 2), padding="same")(x)
    y = LeakyReLU(0.2)(y)

    y = Conv2D(128, (3, 3), kernel_constraint=SpectralNorm2D(128), strides=(2, 2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv2D(256, (3, 3), kernel_constraint=SpectralNorm2D(256), strides=(2, 2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = Conv2D(512, (3, 3), kernel_constraint=SpectralNorm2D(512), strides=(2, 2), padding="same")(y)
    y = LeakyReLU(0.2)(y)

    y = GlobalAveragePooling2D()(y)
    y = Dense(1, kernel_constraint=SpectralNorm1D(1))(y)
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

class SpectralNorm1D(Constraint):
    def __init__(self, output_neurons, power_iterations=1):

        assert power_iterations>=1, "The number of power iterations should be positive integer"
        self.Ip = power_iterations
        u_init = tf.random_uniform_initializer()
        self.u = tf.Variable(initial_value = u_init(shape=(1, output_neurons), dtype='float32'),
                             trainable = False)

    def __call__(self, w):

        W_mat = tf.transpose(w, (1, 0))  # (i, o) => (o, i)

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = l2_norm(tf.matmul(_u, W_mat))
            _u = l2_norm(tf.matmul(_v, W_mat, transpose_b=True))

        sigma = tf.reduce_sum(tf.matmul(_u, W_mat) * _v)
        sigma = tf.cond(sigma == 0, lambda: 1e-8, lambda: sigma)

        self.u.assign(tf.keras.backend.in_train_phase(_u, self.u))
        return w / sigma

class SpectralNorm2D(Constraint):
    def __init__(self, output_neurons, power_iterations=1):

        assert power_iterations>=1, "The number of power iterations should be positive integer"
        self.Ip = power_iterations
        u_init = tf.random_uniform_initializer()
        self.u = tf.Variable(initial_value = u_init(shape=(1, output_neurons), dtype='float32'),
                             trainable = False)

    def __call__(self, w):

        W_mat = tf.transpose(w, (3, 2, 0, 1))  # (h, w, i, o) => (o, i, h, w)
        W_mat = tf.reshape(W_mat, [tf.shape(W_mat)[0], -1])  # (o, i * h * w)

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = l2_norm(tf.matmul(_u, W_mat))
            _u = l2_norm(tf.matmul(_v, W_mat, transpose_b=True))

        sigma = tf.reduce_sum(tf.matmul(_u, W_mat) * _v)
        sigma = tf.cond(sigma == 0, lambda: 1e-8, lambda: sigma)

        self.u.assign(tf.keras.backend.in_train_phase(_u, self.u))
        return w / sigma

def l2_norm(x):
    return x / tf.sqrt(tf.reduce_sum(tf.square(x)) + 1e-8)