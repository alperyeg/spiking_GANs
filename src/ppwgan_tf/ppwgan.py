from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from datetime import datetime
import time
import pickle
import os.path
import tensorflow as tf
import quantities as pq
import yaml

import statsmodels.api as sm
import scipy.stats as stats

# from Simulate_Poisson import IntensityHomogenuosPoisson, generate_sample
from BatchIterator import PaddedDataIterator, BucketedDataIterator
from Plotter import get_intensity, get_integral  # , get_integral_empirical
from elephant.spike_train_generation import homogeneous_poisson_process
from Utils import \
    sequence_filter  # , lambda_estimation, file2sequence, sequence2file

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

##############################################################################
# parameters
with open('config.yaml', 'r') as stream:
    try:
        params = yaml.load(stream)
        print(params)
    except yaml.YAMLError as err:
        print(err)

MODE = params['mode']  # wgan-lp
DATA = params['data']  # hawkes, selfcorrecting, gaussian, rnn
LAMBDA_LP = params['lambda_lp']  # Penality for Lipschtiz divergence
CRITIC_ITERS = params['critic_iters'] # How many critic iterations per generator iteration
BATCH_SIZE = params['batch_size']  # Batch size
MAX_STEPS = params['max_steps']
ITERS = params['iters']  # how many generator iterations to train for
SEED = params['manualSeed']  # set graph-level seed
SET_SEED = params['set_seed']
PRE_TRAIN = params['pre_train']
COST_ALL = params['cost_all']
G_DIFF = params['g_diff']
D_DIFF = params['d_diff']
MARK = params['mark']
ITERATION = params['iteration']
T = params['T']  # end time of simulation
SEQ_NUM = params['seq_num']  # number of sequences
DIM_SIZE = params['dim_size']

# DATA = sys.argv[1]
# SEQ_NUM = int(float(sys.argv[2]))
# LAMBDA_LP = float(sys.argv[3])

# Custom
DATA_PATH = os.path.join(params['dataroot'], params['dataname'])

if DATA in ['mimic', 'meme', 'citation', 'stock', "mixture1", "mixture2",
            "mixture3", "mixture4"]:
    REAL_DATA = True
else:
    REAL_DATA = False

if SET_SEED:
    tf.set_random_seed(SEED)
    np.random.seed(SEED)

##############################################################################
# prepare data

FILE_NAME = 'pickled_data_{}'.format(DATA)
if not os.path.isfile(FILE_NAME):
    t = time.time()
    dat = np.load(DATA_PATH).item()
    print('loading data done in {}'.format(time.time() - t))
    real_sequences = []
    for s in dat['spikes']:
        real_sequences.extend(s)
    # intensityPoisson = IntensityHomogenuosPoisson(lambda0)
    fake_sequences = [homogeneous_poisson_process(
        10 * pq.Hz, t_start=0 * pq.ms, t_stop=6000 * pq.ms)
        for _ in range(SEQ_NUM)]
    # generate_sample(intensityPoisson, T, 20000)
    # pickle.dump([real_sequences, fake_sequences], open(FILE_NAME, 'wb'))
else:
    real_sequences, fake_sequences = pickle.load(open(FILE_NAME, 'rb'))

# print(np.mean([len(item) for item in real_sequences])/T,
#       np.mean([len(item) for item in fake_sequences])/T)
if not REAL_DATA:
    real_sequences = real_sequences[:SEQ_NUM]
    fake_sequences = fake_sequences[:SEQ_NUM]
if DATA in ['citation', 'hawkes', 'selfcorrecting']:
    real_iterator = BucketedDataIterator(real_sequences, T, MARK, D_DIFF)
    fake_iterator = BucketedDataIterator(fake_sequences, T, MARK, G_DIFF)
else:
    real_iterator = PaddedDataIterator(real_sequences, T, MARK, D_DIFF)
    fake_iterator = PaddedDataIterator(fake_sequences, T, MARK, G_DIFF)


###############################################################################
# define model


def generator(rnn_inputs,  # dims batch_size x num_steps x input_size
              seqlen,
              cell_type='LSTM',
              num_layers=1,
              state_size=64,
              batch_size=BATCH_SIZE
              ):
    with tf.variable_scope("generator"):

        num_steps = tf.shape(rnn_inputs)[1]

        # RNN
        if cell_type == 'Basic':
            cell = tf.contrib.rnn.BasicRNNCell(state_size)
        elif cell_type == 'LSTM':
            # tuple of c_state and m_state
            cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)

        if cell_type == 'LSTM':
            cell = tf.contrib.rnn.MultiRNNCell(
                [cell] * num_layers, state_is_tuple=True)
        elif cell_type == 'Basic':
            cell = tf.contrib.rnn.MultiRNNCell(
                [cell] * num_layers, state_is_tuple=False)

        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)
        # dynamic_rnn produces rnn_outputs with shape [batch_size, num_steps, state_size]
        # the outputs is zero after seqlen if provided

        # reshape rnn_outputs
        # reshape and reverse reshape logically consistent
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])

        # Softmax layer
        with tf.variable_scope('FullConnect'):
            W = tf.get_variable('Wt', [state_size, 1])
            b = tf.get_variable(
                'bt', [1], initializer=tf.constant_initializer(0.0))
        logits_t = tf.matmul(rnn_outputs, W) + b
        logits_t = tf.nn.elu(logits_t) + 1  # abs, exp, or nothing is better
        if not D_DIFF and G_DIFF:  # depend on D_DIFF
            logits_t = tf.cumsum(logits_t, axis=1)

        if MARK:
            # Softmax layer
            with tf.variable_scope('softmax'):
                W = tf.get_variable('Wz', [state_size, DIM_SIZE])
                b = tf.get_variable(
                    'bz', [DIM_SIZE], initializer=tf.constant_initializer(0.0))
            logits_prob = tf.matmul(rnn_outputs, W) + b
            logits_prob = tf.nn.softmax(logits_prob)
            logits = tf.concat([logits_t, logits_prob], axis=1)

        if MARK:
            logits = tf.reshape(logits, [batch_size, num_steps, DIM_SIZE + 1])
        else:
            logits = tf.reshape(logits_t, [batch_size, num_steps, 1])

    return logits


def discriminator(rnn_inputs,  # dims batch_size x num_steps x input_size
                  seqlen,
                  lower_triangular_ones,
                  cell_type='LSTM',
                  num_layers=1,
                  state_size=64,
                  batch_size=BATCH_SIZE,
                  cost_all=COST_ALL,
                  scope_reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if scope_reuse:
            scope.reuse_variables()

        num_steps = tf.shape(rnn_inputs)[1]
        keep_prob = tf.constant(0.9)

        # RNN
        if cell_type == 'Basic':
            cell = tf.contrib.rnn.BasicRNNCell(state_size)
        elif cell_type == 'LSTM':
            # tuple of c_state and m_state
            cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)

        if cell_type == 'LSTM':
            cell = tf.contrib.rnn.MultiRNNCell(
                [cell] * num_layers, state_is_tuple=True)
        elif cell_type == 'Basic':
            cell = tf.contrib.rnn.MultiRNNCell(
                [cell] * num_layers, state_is_tuple=False)

        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)

        # Add dropout
        rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

        # reshape rnn_outputs
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])

        # Softmax layer
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, 1])
            b = tf.get_variable(
                'b', [1], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(rnn_outputs, W) + b

        seqlen_mask = tf.slice(
            tf.gather(lower_triangular_ones, seqlen - 1), [0, 0],
            [batch_size, num_steps])
        if cost_all:
            logits = tf.reshape(logits, [batch_size, num_steps])
            logits *= seqlen_mask
            # Average over actual sequence lengths.
            fval = tf.reduce_sum(logits, axis=1)
            fval /= tf.reduce_sum(seqlen_mask, axis=1)
        else:  # Select the Last Relevant Output
            index = tf.range(0, batch_size) * num_steps + (seqlen - 1)
            flat = tf.reshape(logits, [-1, 1])
            relevant = tf.gather(flat, index)
            fval = tf.reshape(relevant, [batch_size])
    return fval


if MARK:
    Z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 2])  # time,dim
    Z_one_hot = tf.one_hot(tf.cast(Z[:, :, 1], tf.int32), DIM_SIZE)
    Z_all = tf.concat([Z[:, :, :1], Z_one_hot], axis=2)
else:
    # [batch_size, num_steps]
    Z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 1])
    Z_all = Z

fake_seqlen = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
fake_data = generator(Z_all, fake_seqlen)
if MARK:
    fake_data_discrete = tf.argmax(fake_data[:, :, 1:], axis=2)

if MARK:
    X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 2])
    # one_hot depth on_value off_value
    X_one_hot = tf.one_hot(tf.cast(X[:, :, 1], tf.int32), DIM_SIZE)
    real_data = tf.concat([X[:, :, :1], X_one_hot], axis=2)
else:
    X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 1])
    real_data = X

real_seqlen = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
lower_triangular_ones = tf.constant(
    np.tril(np.ones([MAX_STEPS, MAX_STEPS])), dtype=tf.float32)
real_mask = tf.slice(tf.gather(lower_triangular_ones, real_seqlen - 1),
                     [0, 0], [BATCH_SIZE, tf.shape(real_data)[1]])
fake_mask = tf.slice(tf.gather(lower_triangular_ones, fake_seqlen - 1),
                     [0, 0], [BATCH_SIZE, tf.shape(fake_data)[1]])
real_mask = tf.expand_dims(real_mask, -1)
fake_mask = tf.expand_dims(fake_mask, -1)

D_fake = discriminator(fake_data, fake_seqlen, lower_triangular_ones)
D_real = discriminator(real_data, real_seqlen,
                       lower_triangular_ones, scope_reuse=True)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
G_loss = -tf.reduce_mean(D_fake)
tf.summary.scalar("G_loss", G_loss)

train_variables = tf.trainable_variables()
generator_variables = [
    v for v in train_variables if v.name.startswith("generator")]
discriminator_variables = [
    v for v in train_variables if v.name.startswith("discriminator")]

print(map(lambda x: x.op.name, train_variables))
print(map(lambda x: x.op.name, generator_variables))
print(map(lambda x: x.op.name, discriminator_variables))

min_steps = tf.minimum(tf.shape(fake_data)[1], tf.shape(real_data)[1])
pre_train_loss = tf.reduce_sum(
    tf.abs(fake_data[:, :min_steps, :] - real_data[:, :min_steps, :]))
pre_train_op = tf.train.RMSPropOptimizer(
    learning_rate=5e-5).minimize(pre_train_loss, var_list=generator_variables)

# WGAN Lipschitz constraint
if MODE == 'wgan-lp':
    length_ = tf.minimum(tf.shape(real_data)[1], tf.shape(fake_data)[1])
    lipschtiz_divergence = tf.abs(D_real - D_fake) / tf.sqrt(
        tf.reduce_sum(tf.square(
            real_data[:, :length_, :] - fake_data[:, :length_, :]),
            axis=[1, 2]) + 0.00001)

    lipschtiz_divergence = tf.reduce_mean((lipschtiz_divergence - 1) ** 2)
    D_loss += LAMBDA_LP * lipschtiz_divergence

    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss,
                                                           var_list=discriminator_variables)
    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss,
                                                           var_list=generator_variables)

##################################################################################
# run

saved_file = "wgan_{}_{}_{}_{}_{}_{}_{}".format(
    DATA, SEQ_NUM, ITERATION, LAMBDA_LP, datetime.now().day,
    datetime.now().hour, datetime.now().minute)
if not os.path.exists('logs/out/%s' % saved_file):
    os.makedirs('logs/out/%s' % saved_file)

gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=1.0, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('./log/%s/'%saved_file,sess.graph)

stop_indicator = False

n_t = 30
ts_real, intensity_real = get_intensity(real_sequences, T, n_t)

# pre-train
if PRE_TRAIN:
    for it in range(80):  # 4000
        real_batch = real_iterator.next_batch(BATCH_SIZE)
        fake_batch = fake_iterator.next_batch(BATCH_SIZE)
        pre_loss_curr, _ = sess.run([pre_train_loss, pre_train_op],
                                    feed_dict={Z: fake_batch[0],
                                               fake_seqlen: fake_batch[1],
                                               X: real_batch[0],
                                               real_seqlen: real_batch[1]})
        if it % 100 == 0:
            print('pre_train_loss:{}'.format(pre_loss_curr))

# GAN train
for it in range(ITERS):
    for _ in range(CRITIC_ITERS):
        real_batch = real_iterator.next_batch(BATCH_SIZE)
        fake_batch = fake_iterator.next_batch(BATCH_SIZE)
        D_loss_curr, _ = sess.run([D_loss, disc_train_op],
                                  feed_dict={Z: fake_batch[0],
                                             fake_seqlen: fake_batch[1],
                                             X: real_batch[0],
                                             real_seqlen: real_batch[1]})

    fake_batch = fake_iterator.next_batch(BATCH_SIZE)
    G_loss_curr, Summary_curr, _ = sess.run([G_loss, merged, gen_train_op],
                                            feed_dict={Z: fake_batch[0],
                                                       fake_seqlen: fake_batch[
                                                           1]})

    # train_writer.add_summary(Summary_curr,global_step=it)

    if it == 0:
        if REAL_DATA:
            pass
            # integral_intensity = get_integral_empirical(real_sequences, intensity_real,T,n_t)
        elif DATA != "rmtpp":
            integral_intensity = get_integral(real_sequences, DATA)
            integral_intensity = np.asarray(integral_intensity)
            fig = sm.qqplot(integral_intensity, stats.expon,
                            distargs=(), loc=0, scale=1, line='45')
            plt.grid()
            fig.savefig('logs/out/{}/real.png'.format(saved_file))
            plt.close()

    if it % 1000 == 0:
        sequences_generator = []
        for _ in range(int(2000 / BATCH_SIZE)):
            sequences_gen = sess.run(fake_data, feed_dict={
                Z: fake_batch[0], fake_seqlen: fake_batch[1]})
            shape_gen = sequences_gen.shape
            sequences_gen = np.reshape(
                sequences_gen, (shape_gen[0], shape_gen[1]))
            if D_DIFF:
                sequences_gen = np.cumsum(sequences_gen, axis=1)
            sequences_gen = sequence_filter(
                sequences_gen, fake_batch[1])  # remove padding tokens
            sequences_generator += sequences_gen

        ts_gen, intensity_gen = get_intensity(sequences_generator, T, n_t)
        deviation = np.linalg.norm(
            intensity_gen - intensity_real) / np.linalg.norm(intensity_real)

        print(
            'Iter: {}; D loss: {:.4}; G_loss: {:.4}; data:{}; deviation: {}'.format(
                it, D_loss_curr, G_loss_curr, DATA, deviation))
        plt.plot(ts_real, intensity_real, label='real')
        plt.plot(ts_gen, intensity_gen, label='generated')
        plt.legend(loc=1)
        plt.xlabel('time')
        plt.ylabel('intensity')
        plt.savefig('logs/out/{}/{}_{}.png'
                    .format(saved_file, str(it).zfill(3), deviation),
                    bbox_inches='tight')
        plt.close()

        if not REAL_DATA and DATA != "rmtpp":
            integral_intensity = get_integral(sequences_generator, DATA)
            integral_intensity = np.asarray(integral_intensity)
            fig = sm.qqplot(integral_intensity, stats.expon,
                            distargs=(), loc=0, scale=1, line='45')
            res, slope_intercept = stats.probplot(
                integral_intensity, dist=stats.expon)
            plt.grid()
            fig.savefig('logs/out/{}/{}.png'.format(saved_file, it))
            plt.close()

            if np.abs(1 - slope_intercept[0]) < 1e-1 and deviation < 1e-1:
                stop_indicator = True
        elif deviation < 1e-2:
            stop_indicator = True

    if it == ITERS - 1 or stop_indicator or it % 1000 == 0:
        sequences_generator = []
        for _ in range(int(SEQ_NUM / BATCH_SIZE)):
            sequences_gen = sess.run(fake_data, feed_dict={
                Z: fake_batch[0], fake_seqlen: fake_batch[1]})
            shape_gen = sequences_gen.shape
            sequences_gen = np.reshape(
                sequences_gen, (shape_gen[0], shape_gen[1]))
            if D_DIFF:
                sequences_gen = np.cumsum(sequences_gen, axis=1)
            sequences_gen = sequence_filter(
                sequences_gen, fake_batch[1])  # remove padding tokens
            sequences_generator += sequences_gen
        # sequence2file(sequences_generator, 'wgan_{}_{}_{}_{}'.format(
        #     DATA, SEQ_NUM, ITERATION, LAMBDA_LP))
        np.save(os.path.join('logs/out', saved_file,
                             'result_{}_{}_{}_{}_{}.npy'.format(DATA, SEQ_NUM,
                                                                ITERATION,
                                                                LAMBDA_LP, it)),
                sequences_generator)
        if it == ITERS - 1 or stop_indicator:
            break
