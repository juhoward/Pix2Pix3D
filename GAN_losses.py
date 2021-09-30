""" Loss functions for Pix2Pix """
import tensorflow as tf

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

dist_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# , 
                                                    #  reduction=tf.keras.losses.Reduction.NONE)
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def dist_generator_loss(disc_generated_output, gen_output, target):
    gan_loss = dist_loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def dist_discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = dist_loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = dist_loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
