""" 3-D Pix2Pix model"""
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, BatchNormalization, LeakyReLU,
                                     Dropout, Conv3DTranspose, ReLU, Input,
                                     Concatenate, ZeroPadding3D)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(Conv3D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(Conv3DTranspose(filters, size, strides=2, 
                               padding='same', kernel_initializer=initializer,
                               use_bias=False))
    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(.5))
    result.add(ReLU())

    return result


def Generator(SEQUENCE_LENGTH=64, output_channels=3):
    inputs = Input([SEQUENCE_LENGTH, 256, 256, 3])
    down_stack = [
        # no batch normalization in first layer, per paper
        # all ReLUs in encoder are leaky
        downsample(64, 4, apply_batchnorm=False),  # C64 (bs,  32, 128, 128, 64)
        downsample(128, 4),  # C128 (bs, 16, 64, 64, 128)
        downsample(256, 4),  # C256 (bs, 8, 32, 32, 256)
        downsample(512, 4),  # C512 (bs, 4, 16, 16, 512)
        downsample(512, 4),  # C512 (bs, 2, 8, 8, 512)
        # downsample(512, 4),  # C512 (bs, 1, 2, 2, 512)
        # downsample(512, 4),  # C512 (bs, 1, 1, 1, 512)
    ]

    up_stack = [
        # all ReLUs in decoder and not leaky
        upsample(512, 4, apply_dropout=True),  # C512 (bs, 4, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # C512 (bs, 4, 4, 4, 1024)
        # upsample(512, 4, apply_dropout=True),  # C512 (bs, 4, 8, 8, 1024)
        # upsample(512, 4),  # C512 (bs, 4, 16, 16, 1024)
        upsample(256, 4),  # C256 (bs, 4, 32, 32, 512)
        upsample(128, 4),  # C128 (bs, 4, 64, 64, 256)
   
    ]

    # final 3-channel convolutional layer per paper
    initializer = tf.random_normal_initializer(0, .02)
    last = Conv3DTranspose(output_channels, 4,
                           strides=2, padding='same',
                           kernel_initializer=initializer,
                           activation='tanh')  # (bs, 4, 256, 256, 3)
    x = inputs
#################################################################################
    # saving latent feature space
    z = list()
    # downsampling
    skips = list()
    for down in down_stack:
        x = down(x)
        skips.append(x)
#################################################################################
    z = skips[-1]
#################################################################################
    skips = reversed(skips[:-1])


    # upsampling
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
##################################################################################
    return tf.keras.Model(inputs=inputs, outputs=(x))


def Discriminator(SEQUENCE_LENGTH=64):
    initializer = tf.random_normal_initializer(0., .02)
    inp = Input(shape=[SEQUENCE_LENGTH, 256, 256, 3], name='input_image')
    tar = Input(shape=[SEQUENCE_LENGTH, 256, 256, 3], name='target_image')

    # this concatenate only differs from Concatenate
    # in that it is used with the keras functional api
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 64, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) 
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = ZeroPadding3D()(down3)
    conv = Conv3D(512, 4, strides=1,
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1)
    batchnorm1 = BatchNormalization()(conv)
    leaky_relu = LeakyReLU()(batchnorm1)
    zero_pad2 = ZeroPadding3D()(leaky_relu)
    last = Conv3D(1, 4, strides=1,
                  kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

##########################################################################################

def build_checkpoint():
    # instantiate a generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    # define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
    return checkpoint




