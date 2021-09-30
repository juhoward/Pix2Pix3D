""" 
This script is used to validate a trained pix2pix3D model.
To run it, activate an environment with tensorflow and opencv.

Run the script from the command line with the name of 
the checkpoint directory you want to load a model from.

Example: python evaluate_model.py cluster_5_100

A series of input / target / ouput png files will be written
to the output directory and used to create a continuous video
with opencv.
"""
import tensorflow as tf
from Pix2Pix_3D import (Generator, Discriminator)
from GAN_losses import (generator_loss, discriminator_loss)
from tf_dataset_utils import make_datasets
from plotting_utils import (generate_images, render_video)
import argparse
import os

tf.enable_eager_execution()

data_dir = '/opt/proj/arm-005II/datasets/KAIST-rgbt-ped-detection/data/kaist-rgbt/'
# training set
vis_data = data_dir + 'train/vis'
lwir_data = data_dir + 'train/lwir'
# validation set
vis_val = data_dir + 'val/visible'
lwir_val = data_dir + 'val/lwir'
checkpoint_dir = './training_checkpoints/'

parser = argparse.ArgumentParser()
parser.add_argument('model_name')
parser.add_argument('-tp', '--train_performance', action='store_true')
args = parser.parse_args(['day_suburban_1200'])

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
                                
latest = tf.train.latest_checkpoint(checkpoint_dir + args.model_name)
print('Restoring model from latest checkpoint: ', checkpoint_dir)
print('Checkpoint:' , latest)
checkpoint.restore(latest)

if args.train_performance:
    # observe performance on the training data
    vis_dataset, ir_dataset= make_datasets(vis_data, lwir_data)
    for n, (vis_batch, ir_batch) in enumerate(zip(vis_dataset.take(6), ir_dataset.take(6))):
        generate_images(generator, vis_batch, ir_batch, n, './output')
else:
    # observe performance on a validation set
    vis_dataset, ir_dataset= make_datasets(vis_val, lwir_val)
    for n, (vis_batch, ir_batch) in enumerate(zip(vis_dataset, ir_dataset)):
        generate_images(generator, vis_batch, ir_batch, n, './output')

render_video()