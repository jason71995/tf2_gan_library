import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from PIL import Image

from gan_lib.dcgan import build_generator, build_discriminator, build_train_step
# from gan_lib.lsgan import build_generator, build_discriminator, build_train_step
# from gan_lib.wgan_gp import build_generator, build_discriminator, build_train_step
# from gan_lib.sngan import build_generator, build_discriminator, build_train_step
# from gan_lib.sagan import build_generator, build_discriminator, build_train_step

epoch = 50
steps = 1000
image_size = (32, 32, 3)
noise_size = (2, 2, 32)
batch_size = 16

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_of_data = x_train.shape[0]
x_train = x_train.astype("float32")
x_test  = x_test.astype("float32")
x_train = (x_train / 255) * 2 - 1
x_test  = (x_test / 255) * 2 - 1
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

generator = build_generator(noise_size)
discriminator = build_discriminator(image_size)
train_step = build_train_step(generator, discriminator)

for e in range(epoch):
    for s in range(steps):

        real_images = x_train[np.random.permutation(num_of_data)[:batch_size]]
        noise = np.random.normal(0.0, 1.0, (batch_size,) + noise_size)

        d_loss, g_loss = train_step(real_images, noise)
        print ("[{0}/{1}] [{2}/{3}] d_loss: {4:.4}, g_loss: {5:.4}".format(e, epoch, s, steps, d_loss, g_loss))

    # predict and save images
    image = generator.predict(np.random.normal(size=(10 * 10,) + noise_size))
    image = np.reshape(image, (10, 10, 32, 32, 3))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    image = np.reshape(image, (10 * 32, 10 * 32, 3))
    image = 255 * (image + 1) / 2
    image = image.astype("uint8")
    Image.fromarray(image, "RGB").save("e{:02d}_predicts.png".format(e))