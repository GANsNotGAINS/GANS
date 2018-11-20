import tensorflow as tf
import tensorflow.contrib.layers as layers
import datetime
import tensorflow.contrib.slim as slim
import numpy as np

# Using MNIST dataset to test GAN code
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Treating the image as a flat array of pixels
image_dim = 784 # 28*28 pixels

# Making models and tensorboard directory
import os
if not os.path.exists('gan_models/'):
    os.makedirs('gan_models')
if not os.path.exists('gan_tensorboard/'):
    os.makedirs('gan_tensorboard')


# Core GAN code containing discriminator and generator model code
def Discriminator(img, batch_size, is_train=True, scope='Dis', reuse=True):
    # Need reuse params b/c we use discriminator twice
    with tf.variable_scope(scope, reuse=reuse) as scope:
        d1 = slim.fully_connected(img, 256, scope='d1_fc', activation_fn=tf.nn.relu)
        d2 = slim.fully_connected(d1, 1, scope='d2_fc', activation_fn=None)
        return tf.nn.sigmoid(d2), d2

def Generator(z, batch_size, img_shape, is_train=True, scope='Gen'):
    # Generate imgs [batch_size, h, w, c] given noise vector z
    with tf.variable_scope(scope) as scope:
        g1 = slim.fully_connected(z, 256, scope='g1_fc', activation_fn=tf.nn.relu)
        g2 = slim.fully_connected(g1, image_dim, scope='g2_fc', activation_fn=tf.nn.sigmoid)
        return g2

# Main function contains loss calculation, optimizer, and trainer code
def main():
    sess = tf.Session()
    batch_size = 50
    dim_z = 200
    learning_rate = 0.001

    img = tf.placeholder(shape=[batch_size, 784], name='img', dtype=tf.float32)
    is_train = tf.placeholder_with_default(True, [], name="is_train")
    # Generator Setup
    z = tf.random_uniform([batch_size, dim_z], minval=-1, maxval=1, dtype=tf.float32)
    generated_img = Generator(z, batch_size, img.get_shape().as_list())

    # Discriminator Setup
    real, real_logits = Discriminator(img, batch_size, reuse=False)
    fake, fake_logits = Discriminator(generated_img, batch_size, reuse=True) # Using generated image here

    # Loss Calculation
    G_alpha = 0.9
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                 logits=real_logits, labels=tf.zeros_like(real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                 logits=fake_logits, labels=tf.ones_like(fake)))
    discriminator_loss = d_loss_fake + d_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_logits, labels=tf.zeros_like(fake)))

    loss = tf.reduce_mean(generator_loss + discriminator_loss) # Total loss is sum of generator and discriminator loss
    # Summaries for Tensorboard. Losses and sample images are visualized.
    tf.summary.scalar("loss/dis_loss", tf.reduce_mean(discriminator_loss))
    tf.summary.scalar("loss/gen_loss", tf.reduce_mean(generator_loss))
    tf.summary.scalar("loss/GAN_loss", loss)
    tf.summary.image("real", tf.reshape(img, [batch_size, 28, 28, 1]), max_outputs=1)
    tf.summary.image("generated", tf.reshape(generated_img, [batch_size, 28, 28, 1]))

    logdir = "gan_tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)


    global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
    all_vars = tf.trainable_variables()
    d_var = [v for v in all_vars if v.name.startswith('Dis')]
    g_var = [v for v in all_vars if v.name.startswith(('Gen'))]

    dis_opt = tf.contrib.layers.optimize_loss(
        loss=discriminator_loss,
        global_step=global_step,
        learning_rate=learning_rate*0.5,
        optimizer=tf.train.AdamOptimizer(beta1=0.5),
        clip_gradients=20.0,
        name='dis_optimize_loss',
        variables=d_var
    )

    gen_opt = tf.contrib.layers.optimize_loss(
        loss=generator_loss,
        global_step=global_step,
        learning_rate=learning_rate*0.5,
        optimizer=tf.train.AdamOptimizer(beta1=0.5),
        clip_gradients=20.0,
        name='gen_optimize_loss',
        variables=g_var
    )

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Training 
    for idx in range(50000):
        img_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 784])
        # Train and run both optimizers. May want to consider different ratios of training
        _ = sess.run([dis_opt, gen_opt], {img: img_batch, is_train: True})

        if idx % 100 == 0:
            # Use validation set for outputting loss and accuracies
            img_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 784])
            GAN_loss, summary = sess.run([loss, summary_op], {img: img_batch, is_train: False})
            print("GAN Loss: {}".format(GAN_loss))
            writer.add_summary(summary, idx)
        if idx % 10000 == 0 and idx != 0:
            save_path = saver.save(sess, "gan_models/gan.ckpt", global_step=idx)
            print("Saving model")

if __name__ == '__main__':
    main()
