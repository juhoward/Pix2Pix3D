""" utilities for making tf.dataset """
import tensorflow as tf


def load_paths(data_dir):
    img_list = tf.data.Dataset.list_files(data_dir + '/*.jpg', shuffle=False)
    return img_list

def return_img(img_list):
    img = tf.io.read_file(img_list)
    img = tf.image.decode_png(img)
    img = tf.cast(img, tf.float32)
    return img

def resize(img, height=256, width=256):
    img = tf.image.resize(img, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img

def normalize(img):
    # normalizing the images to [-1, 1]
    img = (img / 127.5) - 1
    return img

def load_image(img_list):
    img = return_img(img_list)
    img = resize(img)
    img = normalize(img)
    return img

def make_datasets(vis_dir, ir_dir):
    vis_dataset = load_paths(vis_dir)
    ir_dataset = load_paths(ir_dir)
    vis_dataset = vis_dataset.map(
        load_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ir_dataset = ir_dataset.map(
        load_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    vis_dataset = vis_dataset.batch(64)
    ir_dataset = ir_dataset.batch(64)
    return vis_dataset, ir_dataset    