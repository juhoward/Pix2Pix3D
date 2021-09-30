""" utilities for making tf.dataset """
import tensorflow as tf
import glob
import numpy as np


IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 150
BUFFER_SIZE = 1
BATCH_SIZE = 1
SEQUENCE_LENGTH = 64


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def load(image_file, real_only=False):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)

    w = tf.shape(image)[1]
    w = w // 2

    h = tf.shape(image)[0]
    bar = (h - 192) // 2
    input_image = image[bar:(h-bar), :w, :]
    real_image = image[bar:(h-bar), w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if real_only:
        return real_image
    else:
        return input_image, real_image


# GAN specific loading functions
def resize(input_image, real_image, height=IMG_HEIGHT, width=IMG_WIDTH):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def normalize(input_image, real_image):
    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def make_dataset(data_dir, shuffle=False):
    train_dataset = tf.data.Dataset.list_files(data_dir + '/*.jpg', shuffle=False)
    train_dataset = train_dataset.map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(SEQUENCE_LENGTH)
    return train_dataset


# general image loading functions
def resize_single_img(real_image, height=IMG_HEIGHT, width=IMG_WIDTH):
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return real_image


def normalize_single_img(real_image):
    # normalizing the images to [0, 1]
    return real_image / 255.


def load_single_image(image_file):
    real_image = load(image_file, real_only=True)
    real_image = resize_single_img(real_image)
    # real_image = normalize_single_img(real_image)
    return real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def make_single_img_dataset(data_dir, sequence_length=64, shuffle=False):
    train_dataset = tf.data.Dataset.list_files(data_dir + '/*.jpg', shuffle=False)
    train_dataset = train_dataset.map(load_single_image,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle:
        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(sequence_length)
    return train_dataset


def next_frame_dataset(data_dir, BATCH_SIZE=1, SEQUENCE_LENGTH=32):
    """A generator that returns 32 concatenated images"""
    dataset = sorted(glob.glob(data_dir + "/*.jpg"))
    print('Found ', len(dataset), ' images.')
    counter = 0
    while True:
        input_images = np.zeros(
            (BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
        output_image = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
        if (counter + SEQUENCE_LENGTH >= len(dataset)):
            counter = 0
        input_imgs = list()
        for i in range(SEQUENCE_LENGTH):
            input_imgs.append(dataset[counter + i])
        imgs = [load_single_image(img) for img in sorted(input_imgs)]
        # concatenates images together
        input_images = np.concatenate(imgs, axis=2)
        # load last image in list
        output_image = load_single_image(input_imgs[-1])
        yield input_images[np.newaxis, ...], output_image[np.newaxis, ...]
        counter += SEQUENCE_LENGTH


def make_test_dataset(data_dir):
    test_dataset = tf.data.Dataset.list_files(data_dir + '/*.jpg', shuffle=False)
    # test_dataset = tf.sort(test_dataset)
    test_dataset = test_dataset.map(load_image_test,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset
