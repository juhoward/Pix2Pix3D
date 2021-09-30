import tensorflow as tf
from Pix2Pix_3D import (Generator, Discriminator, build_checkpoint)
from GAN_losses import (generator_loss, discriminator_loss)
from tf_dataset_utils import make_datasets
import datetime
import time
from time import process_time
import os
import numpy as np


# limit GPU usage
print('Available GPUs: ', tf.config.experimental.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


@tf.function
def train_step(input_image, target, epoch):
    #  create two gradient calculators for the generator and discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

# @tf.function
def fit(vis_ds, ir_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        # Train
        # low_loss = 11.0
        epoch = tf.Variable(epoch, dtype=np.int64)
        print("Epoch: ", epoch.numpy(), "/", epochs)
        for n, (vis_batch, ir_batch) in enumerate(zip(vis_ds, ir_ds)):
            vis_batch = tf.expand_dims(vis_batch, axis=0)
            ir_batch = tf.expand_dims(ir_batch, axis=0)
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            # reversed order
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(vis_batch, ir_batch, epoch)
        print('\nGen total loss: ', gen_total_loss.numpy(),
              ' Gen loss : ', gen_gan_loss.numpy(),
              ' Disc loss: ', disc_loss.numpy())
        if np.isnan(gen_total_loss.numpy()):
            break
        # # save checkpoint if it beats previous low loss
        # elif gen_total_loss.numpy() < low_loss:
        #     low_loss = gen_total_loss.numpy()
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        # else:
        #     continue        
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
        # save one last checkpoint
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

###########################################################################
data_dir = '/opt/proj/arm-005II/datasets/KAIST-rgbt-ped-detection/data/kaist-rgbt/'
vis_data = data_dir + 'train/visible'
ir_data = data_dir + 'train/lwir'
model_name = 'day_suburban_1200'
continue_train = False

# instantiate a generator and discriminator
generator = Generator()
discriminator = Discriminator()
# define optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=.5)

# tensorboard logs
log_dir = './logs/'
summary_writer = tf.summary.create_file_writer(
    log_dir + model_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# checkpoints
checkpoint_dir = './training_checkpoints/'

checkpoint_prefix = os.path.join(checkpoint_dir, model_name, 'ckpt')

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
if continue_train:
    new_model_name = 'new_mod'
    latest = tf.train.latest_checkpoint(checkpoint_dir + model_name)
    print('Restoring model from latest checkpoint: ', checkpoint_dir)
    print('Checkpoint:' , latest)
    checkpoint.restore(latest)
    checkpoint_prefix = os.path.join(checkpoint_dir, new_model_name, 'ckpt')

vis_dataset, ir_dataset= make_datasets(vis_data, ir_data)
# did you reload the optimizers??
fit(vis_dataset, ir_dataset, 1200)