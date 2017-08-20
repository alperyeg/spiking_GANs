"""
Training a generative adversarial network to sample from a
Gaussian distribution, 1-D normal distribution
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
import plotting as plots
import quantities as pq

from scipy.stats import norm, poisson
from six.moves import range
import ops
from data_generation import DataDistribution, GeneratorDistribution

sns.set(color_codes=True)

# TODO proper tf.summary
# TODO save model as a checkpoint

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# The generator and discriminator networks are quite simple.
# The generator is a linear transformation passed through a non-linearity
# (a softplus function), followed by another linear transformation.
def generator(inpt, h_dim, gf_dim, is_training=True, model='simple'):
    if model == 'simple':
        return _generator_simple(inpt, h_dim)
    elif model == "dcgan":
        return _generator_dcgan(inpt, gf_dim, is_training)


def _generator_simple(inpt, h_dim):
    h0 = tf.nn.softplus(ops.linear(inpt, h_dim, 'g0_simple'))
    h1 = ops.linear(h0, 1, 'g1_simple')
    return h1


def _generator_dcgan(inpt, gf_dim, is_training):
    with tf.variable_scope('generator'):
        z, h0_w, h0_b = ops.linear(inpt, gf_dim * 8 * 4 * 4, scope='g0_dcgan',
                                   with_w=True)
        hs = [None]
        hs[0] = tf.reshape(z, [-1, 4, 4, gf_dim * 8])

        # batch norm parameter
        decay = 0.9
        epsilon = 1e-5
        # do batch norm
        bn = tf.contrib.layers.batch_norm(hs[0], decay=decay,
                                          updates_collections=None,
                                          epsilon=epsilon,
                                          center=True, scale=True,
                                          is_training=is_training,
                                          scope='g_bn0')
        hs[0] = tf.nn.relu(bn, is_training)

        i = 1  # Iteration number.
        depth_mul = 8  # Depth decreases as spatial component increases.
        size = 8  # Size increases as depth decreases.

        while size < len(inpt):
            hs.append(None)
            name = 'g{}_dcgan'.format(i)
            hs[i], _, _ = ops.conv2d_transpose(hs[i - 1],
                                           [batch_size, size, size,
                                            gf_dim * depth_mul], scope=name,
                                           with_w=True)
            bn = tf.contrib.layers.batch_norm(hs[i], decay=decay,
                                              updates_collections=None,
                                              epsilon=epsilon,
                                              center=True, scale=True,
                                              is_training=is_training,
                                              scope='g_bn{}'.format(i))

            hs[i] = tf.nn.relu(bn, is_training)
            i += 1
            depth_mul //= 2
            size *= 2

        hs.append(None)
        name = 'g{}_dcgan'.format(i)
        hs[i], _, _ = ops.conv2d_transpose(hs[i - 1],
                                       [batch_size, size, size, 1], scope=name,
                                       with_w=True)

    return tf.nn.tanh(hs[i])


# Make sure that the discriminator is more powerful than the generator,
# as otherwise it did not have sufficient capacity to learn to be able to
# distinguish accurately between generated and real samples.
# So make it a deeper neural network, with a larger number of dimensions.
def discriminator(inpt, h_dim, df_dim, batch_size_, minibatch_layer=True,
                  model='MLP', reuse=False):
    if model == 'MLP':
        return _discriminator_mlp(inpt, h_dim, minibatch_layer, reuse)
    elif model == 'CONV':
        return _discriminator_conv(inpt, df_dim, batch_size_, reuse)


def _discriminator_mlp(inpt, h_dim, minibatch_layer=True, reuse=False):
    """
    Discriminator using Multi-layer perceptron

    It uses tanh nonlinearities in all layers except the final one, which is
    a sigmoid (the output of which is interpreted as a probability).

    """
    with tf.variable_scope("Discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = tf.tanh(ops.linear(inpt, h_dim * 2, scope='d0_mlp'))
        h1 = tf.tanh(ops.linear(h0, h_dim * 2, scope='d1_mlp'))

        # without the minibatch layer, the discriminator
        # needs an additional layer # to have enough capacity to separate
        # the two distributions correctly
        if minibatch_layer:
            h2 = run_minibatch(h1)
        else:
            h2 = tf.tanh(ops.linear(h1, h_dim * 2, scope='d2_mlp'))

        h3 = ops.linear(h2, 1, scope='d3_mlp')
    return tf.sigmoid(h3), h3


def _discriminator_conv(inpt, df_dim, batch_size_, reuse=False):
    """
    Discriminator using Convolutional Network

    It uses leaky rectifier units in all layers except the final one, which is
    a sigmoid (the output of which is interpreted as a probability).


    """
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        # TODO add batch normalization
        # h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name='d_h1_conv'),
        # self.is_training))
        h0 = ops.lrelu(ops.conv2d(inpt, df_dim, scope='d0_conv'))
        h1 = ops.lrelu(ops.conv2d(h0, df_dim * 2, scope='d1_conv'))
        h2 = ops.lrelu(ops.conv2d(h1, df_dim * 4, scope='d2_conv'))
        h3 = ops.lrelu(ops.linear(h2, df_dim * 8, scope='d3_conv'))
        # h4 = ops.linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
        h4 = ops.linear(tf.reshape(h3, [batch_size_, -1]), 1, 'd_h4_lin')
        return tf.nn.sigmoid(h4), h4


def run_minibatch(inpt, num_kernels=5, kernel_dim=3):
    """
    * Take the output of some intermediate layer of the discriminator.
    * Multiply it by a 3D tensor to produce a matrix (of size num_kernels x 
    kernel_dim in the code below).
    * Compute the L1-distance between rows in this matrix across all samples 
    in a batch, and then apply a negative exponential.
    * The minibatch features for a sample are then the sum of these 
    exponentiated distances.
    * Concatenate the original input to the minibatch layer (the output of 
    the previous discriminator layer) with the newly created minibatch 
    features, and pass this as input to the next layer of the discriminator.

    :param inpt: 
    :param num_kernels: 
    :param kernel_dim: 
    :return: 
    """
    x = ops.linear(inpt, num_kernels * kernel_dim, scope='minibatch',
                   stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, axis=3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), axis=0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
    return tf.concat(values=[inpt, minibatch_features], axis=1)


def optimizer(loss, var_list, initial_learning_rate, name='GradientDescent',
              **kwargs):
    """

    :param loss:  tensor containing the value to minimize.
    :param var_list:  Optional list or tuple of Variable objects to update 
                      to minimize loss. 
    :param initial_learning_rate: tensor or a floating point value.
    :param name: str, name of the optimizer, Default is `GradientDescent`
    :param kwargs: Additional keywords
    :return: An Operation that updates the variables in var_list. 
    """
    if name == 'GradientDescent':
        return _gradient_descent_optimizer(loss, var_list,
                                           initial_learning_rate)
    elif name == 'MomentumOptimizer':
        return _momentum_optimizer(loss, var_list, initial_learning_rate)
    elif name == 'AdamOptimizer':
        return _adam_optimizer(loss, var_list, initial_learning_rate)


def _gradient_descent_optimizer(loss, var_list, initial_learning_rate):
    """
    GradientDescentOptimizer with exponential learning rate decay

    """
    # finding good optimization parameters require some tuning
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    optimizer_ = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer_


def _momentum_optimizer(loss, var_list, initial_learning_rate):
    """
    Momentum Optimizer with exponential learning rate decay

    """
    # Apply exponential decay to the learning rate; staircase to use integer
    #  division in a stepwise (=staircase) fashion
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True)
    optimizer_ = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                            momentum=0.6).minimize(loss,
                                                                   global_step=batch,
                                                                   var_list=var_list)
    return optimizer_


def _adam_optimizer(loss, var_list, initial_learning_rate):
    # TODO check if correct
    decay = 0.95
    beta1 = 0.35
    num_decay_steps = 150
    batch = tf.Variable(0)
    # TODO check learning rate, otherwise start with 0.0002
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True)
    optimizer_ = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=beta1).minimize(loss,
                                                              global_step=batch,
                                                              var_list=var_list)
    return optimizer_


class GAN(object):

    def __init__(self, data_distribution, gen, num_steps, batch_size, minibatch,
                 hidden_size=4, learning_rate=0.03,
                 z_dim=100, gf_dim=64, df_dim=64,
                 **kwargs):
        """
        :param data_distribution: class DataDistribution
        :param gen: tensor generator net
        :param num_steps: int
        :param batch_size: int
        :param minibatch: bool
        :param log_every: int
        :param anim_path: string
        :param hidden_size:
        :param learning_rate: 
        :param z_dim: int, Dimension of dim for Z
        :param gf_dim: int, Dimension of gen filters in first conv layer
        :param df_dim: int, Dimension of discr filters in first conv layer
        :param kwargs: additional parameters
        """
        self.input_shape = kwargs['input_shape']
        binned, _, _ = data_distribution.poisson_nonstat_sample(t_stop=10000 * pq.ms,
                                                                num_bins=self.input_shape[0],
                                                                num_sts=2 * num_steps)
        self.data = binned.to_array()
        self.gen = gen
        self.num_steps = num_steps
        self.pre_train_steps = kwargs['pre_train_steps']
        self.pre_train = kwargs['pre_train']
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = kwargs['log_every']
        self.hidden_size = hidden_size
        self.anim_path = kwargs['anim_path']
        self.anim_frames = []
        self.learning_rate = learning_rate
        self.loss_d_plot = []
        self.loss_g_plot = []
        self.training_run = kwargs['training_run']

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # can use a higher learning rate when not using the minibatch layer
        if self.minibatch:
            self.learning_rate = self.learning_rate / 100.0     # 0.005
            print(
                'minibatch active setting smaller learning rate of: {}'.format(
                    self.learning_rate))
        # TODO change tensor size to actual input size
        self._create_model()

    def _create_model(self):
        """
        Creates the model

        Does the pre-training and optimization steps, creates also the
        Generative and Discriminator Network.

        """
        # TODO in optimizing steps try other optimizers

        # Define the pre training
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=self.input_shape)
            # self.pre_labels = tf.placeholder(tf.float32, shape=tensor_size)
            pre_labels = tf.ones(shape=self.input_shape)
            D_pre = discriminator(self.pre_input, self.hidden_size, self.df_dim,
                                  self.batch_size, self.minibatch, model='MLP')
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a
        # noise distribution as input, and passes them through an MLP.
        with tf.variable_scope('Generator'):
            self.z = tf.placeholder(tf.float32, shape=self.input_shape)
            self.G = generator(self.z, self.hidden_size, self.gf_dim,
                               is_training=self.training_run, model='simple')

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated
        # samples (self.z).
        #
        # Here we create two copies of the discriminator network
        # (that share parameters), as you cannot use the same network with
        # different inputs in TensorFlow.
        # with tf.variable_scope('Disc') as scope:
        self.x = tf.placeholder(tf.float32, shape=self.input_shape)
        self.D1, self.D1_logits = discriminator(self.x, self.hidden_size,
                                                self.df_dim,
                                                self.batch_size,
                                                self.minibatch, reuse=False)
        # make a copy of D using same variables, but with G as input
        self.D2, self.D2_logits = discriminator(self.G, self.hidden_size,
                                                self.df_dim,
                                                self.batch_size,
                                                self.minibatch, reuse=True)

        # Define the loss for discriminator and generator networks
        # and create optimizers for both
        # self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        # self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits,
                                                    labels=tf.ones_like(self.D1)))
        self.loss_d_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits,
                                                    labels=tf.zeros_like(
                                                        self.D2)))
        self.loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits,
                                                    labels=tf.ones_like(self.D2)))

        self.loss_d = self.d_loss_real + self.loss_d_fake

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='Discriminator')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='Generator')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        """
        Draws samples from the data distribution and the noise distribution,
        and alternates between optimizing the parameters of the discriminator
        and the generator.
        """
        tensor_size = self.input_shape
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            if pre_train:
                # pretraining discriminator
                for step in range(self.pre_train_steps):
                    # d = (np.random.random(self.batch_size) - 0.5) * 10.0
                    d = self.data[self.pre_train_steps + step]
                    # labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                    # labels = self.data[step]
                    pretrain_loss, _ = session.run(
                        [self.pre_loss, self.pre_opt], {
                            self.pre_input: np.reshape(d, tensor_size)
                        })
                self.weightsD = session.run(self.d_pre_params)
                tf.summary.histogram('weightsD', self.weightsD[0])

                # copy weights from pre-training over to new D network
                for i, v in enumerate(self.d_params):
                    session.run(tf.assign(v, self.weightsD[i]))
                    # session.run(v.assign(self.weightsD[i]))

            for step in range(self.num_steps):
                # update discriminator
                x = np.sort(self.data[self.num_steps + step])
                z = self.gen.sample_int(tensor_size[0])
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, tensor_size),
                    self.z: np.reshape(z, tensor_size)
                })

                # update generator
                z = self.gen.sample_int(tensor_size[0])
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, tensor_size)
                })

                self.loss_d_plot.append(loss_d)
                self.loss_g_plot.append(loss_g)
                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

                if self.anim_path:
                    self.anim_frames.append(plots.samples(session, save=False,
                                                          lower_range=self.gen.lower_range,
                                                          upper_range=self.gen.upper_range,
                                                          batch_size=self.input_shape[0],
                                                          D1=self.D1,
                                                          G=self.G,
                                                          x=self.x,
                                                          data=self.data,
                                                          z=self.z))

            if self.anim_path:
                plots.save_animation(self.anim_path, self.anim_frames,
                                     lower_range=self.gen.lower_range,
                                     upper_range=self.gen.upper_range)
            else:
                plots.plot_distributions(session, save=False,
                                         lower_range=self.gen.lower_range,
                                         upper_range=self.gen.upper_range,
                                         batch_size=self.input_shape[0],
                                         D1=self.D1,
                                         G=self.G,
                                         x=self.x,
                                         z=self.z,
                                         data=self.data)
                plots.plot_training_loss(self.loss_g_plot, self.loss_d_plot)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs', session.graph)
            writer.close()


def main(**kwargs):
    model = GAN(DataDistribution(),
                GeneratorDistribution(kwargs['lower_range'],
                                      kwargs['upper_range']),
                num_steps=kwargs['num_steps'],
                pre_train_steps=kwargs['pre_train_steps'],
                pre_train=kwargs['pre_train'],
                batch_size=kwargs['batch_size'],
                input_shape=kwargs['input_shape'],
                minibatch=kwargs['minibatch'],
                log_every=kwargs['log_every'],
                anim_path=kwargs['anim_path'],
                checkpoint_dir=kwargs['checkpoint_dir'],
                training_run=kwargs['training_run']
                )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim', type=str, default=None,
                        help='name of the output animation file (default: none)')
    return parser.parse_args()


if __name__ == '__main__':
    num_steps = 2000
    pre_train_steps = 1000
    pre_train = True
    batch_size = 80
    input_shape = (80, 1)
    minibatch = False
    lower_range = 0
    upper_range = 20
    log_every = 100
    training_run = True
    anim_path = ""
    checkpoint_dir = "./"
    # TODO make a parameter dictionary for convenience
    main(num_steps=num_steps,
         pre_train_steps=pre_train_steps, pre_train=pre_train,
         batch_size=batch_size, input_shape=input_shape, minibatch=minibatch,
         lower_range=lower_range, upper_range=upper_range,
         log_every=log_every, anim_path=anim_path, training_run=training_run,
         checkpoint_dir=checkpoint_dir)
