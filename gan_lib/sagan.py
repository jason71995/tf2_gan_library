'''
    Self-Attention Generative Adversarial Networks
    Ref:
        - https://arxiv.org/abs/1805.08318
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, GlobalAveragePooling2D, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

def build_generator(input_shape):
    x = Input(input_shape)

    y = Conv2DTranspose(512, (3, 3), strides=(2,2), padding="same")(x)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(512//8)(y)

    y = Conv2DTranspose(256, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(256//8)(y)

    y = Conv2DTranspose(128, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(128//8)(y)

    y = Conv2DTranspose(64, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(64//8)(y)

    y = Conv2D(3, (3, 3), padding="same", activation="tanh")(y)
    return Model(x, y)

def build_discriminator(input_shape):
    x = Input(input_shape)

    y = Conv2D(64, (3, 3), strides=(2,2), padding="same")(x)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(64//8)(y)

    y = Conv2D(128, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(128//8)(y)

    y = Conv2D(256, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(256//8)(y)

    y = Conv2D(512, (3, 3), strides=(2,2), padding="same")(y)
    y = LeakyReLU(0.2)(y)
    y = SelfAttention2D(512//8)(y)

    y = GlobalAveragePooling2D()(y)
    y = Dense(1, activation="sigmoid")(y)
    return Model(x, y)


def build_train_step(generator, discriminator):
    d_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)
    g_optimizer = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9)

    @tf.function
    def train_step(real_image, noise):

        fake_image = generator(noise)
        pred_real, pred_fake = tf.split(discriminator(tf.concat([real_image, fake_image], axis=0)), num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(-tf.math.log(pred_real + 1e-8)) + tf.reduce_mean(-tf.math.log(1 - pred_fake + 1e-8))
        g_loss = tf.reduce_mean(-tf.math.log(pred_fake + 1e-8))

        d_gradients = tf.gradients(d_loss, discriminator.trainable_variables)
        g_gradients = tf.gradients(g_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        return d_loss, g_loss

    return train_step


class SelfAttention2D(Layer):

    def __init__(self, reduced_filters, **kwargs):
        self.reduced_filters = reduced_filters
        super(SelfAttention2D, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_f = self.add_weight(name='w_f',
                                   shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   initializer='glorot_uniform',
                                   trainable=True)

        self.w_g = self.add_weight(name='w_g',
                                   shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   initializer='glorot_uniform',
                                   trainable=True)

        self.w_h = self.add_weight(name='w_h',
                                   shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   initializer='glorot_uniform',
                                   trainable=True)

        self.w_v = self.add_weight(name='w_v',
                                   shape=(1, 1, self.reduced_filters, input_shape[-1]),
                                   initializer='glorot_uniform',
                                   trainable=True)

        self.gamma = self.add_weight(name='gamma',
                                     shape=(1, ),
                                     initializer='zero',
                                     trainable=True)

        super(SelfAttention2D, self).build(input_shape)

    def call(self, x):

        x_shape = tf.shape(x)
        x_f = tf.keras.backend.conv2d(x,self.w_f,padding="same")
        x_g = tf.keras.backend.conv2d(x,self.w_g,padding="same")
        x_h = tf.keras.backend.conv2d(x,self.w_h,padding="same")

        x_f = tf.reshape(x_f, (x_shape[0], x_shape[1] * x_shape[2], -1))
        x_g = tf.reshape(x_g, (x_shape[0], x_shape[1] * x_shape[2], -1))
        x_h = tf.reshape(x_h, (x_shape[0], x_shape[1] * x_shape[2], -1))

        y = tf.matmul(x_f, x_g, transpose_b=True) # attention map
        y = tf.nn.softmax(y, axis = 1)
        y = tf.matmul(y, x_h)
        y = tf.reshape(y, (x_shape[0], x_shape[1], x_shape[2], -1))
        y = tf.keras.backend.conv2d(y,self.w_v,padding="same")
        y = self.gamma * y + x
        return y

    def compute_output_shape(self, input_shape):
        return input_shape