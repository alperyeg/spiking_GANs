from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os.path
import sys
import scipy.stats as stats
import argparse
import os
import random
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import datetime

from torch.autograd import Variable
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter

from Simulate_Poisson import generate_sample, generate_samples_marked
from data_generation import DataDistribution

try:
    JOB_ID = int(os.environ['SLURM_JOB_ID'])
    ARRAY_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    ARRAY_ID = 10

# TODO change to config file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    help='cifar10 | lsun | imagenet | folder | lfw | fake | '
                         'step_rate | variability | pattern',
                    default='step_rate')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--dataname', required=False,
                    help='Name of dataset, use together with dataroot to '
                         'navigate to the dataset', type=str)
parser.add_argument('--lambda_lp', required=False, default=0.1,
                    help='Penalty for Lipschtiz divergence')
parser.add_argument('--critic_iters', required=False, default=5,
                    help='How many critic iterations per generator iteration')
parser.add_argument('--batch_size', required=False, default=256, type=int)
parser.add_argument('--max_steps', required=False, default=300)
parser.add_argument('--epochs', required=False, default=25,
                    help='How many generator iterations to train for')
parser.add_argument('--iters', required=False, default=20000,
                    help='How many generator iterations to train for')
parser.add_argument('--manualSeed', required=False,
                    default=123, help='set graph-level seed')
parser.add_argument('--T', required=False, default=15.0,
                    help='End time of simulation')
parser.add_argument('--seq_num', required=False, default=20000,
                    help='Number of sequences')
parser.add_argument('--mode', required=False, default='wgan-lp',
                    help='Type of Network to use: wgan-lp | dcgan')
parser.add_argument('--data', required=False, default='hawkes',
                    help='hawkes | selfcorrecting | gaussian| rnn')
parser.add_argument('--cuda', action='store_true', help='Enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('--gen', default='',
                    help="path to Generator (to continue training)")
parser.add_argument('--disc', default='',
                    help="path to Discriminator (to continue training)")
parser.add_argument('--outf', default='./logs/run_{}_rate{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), ARRAY_ID),
                    help='folder to output images and model checkpoints')
parser.add_argument('--encoding', type=bool, help='load encoded data',
                    default=False)
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)

opt = parser.parse_args()
with open('params.yaml', 'r') as stream:
    try:
        params = yaml.load(stream)
        print(params)
    except yaml.YAMLError as err:
        print(err)
print(opt)

# other parameters
# TODO move to param config file
PRE_TRAIN = True
COST_ALL = True
G_DIFF = True
D_DIFF = True
MARK = False
ITERATION = 0
DIM_SIZE = 1
ngpu = int(opt.ngpu)

# Cuda
# cuda_available = torch.cuda.is_available()
# device = torch.device("cuda" if cuda_available else "cpu")
# tensor = tensor.to(device)

if opt.manualSeed is None:
    opt.manualSeed = 123
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print(
        "WARNING: You have a CUDA device, "
        "so you should probably run with cuda")


# Define a dictionary to save results
save_dict = {}

# Define writer object for tensorboard
writer = SummaryWriter(log_dir=os.path.join(opt.outf, 'tensorboard'))


def load_data(dataset_name, encoding=False, array_id=10):
    if encoding:
        try:
            fname = dataset_name
            dat = np.load(fname).item()
        except (KeyError, FileNotFoundError):
            fname = './logs/data/data_NS10000_IS64_type-{}_encoded-{}_rate{}.npy'.format(
                dataset_name, encoding, array_id)
        print("Loaded dataset: {}".format(fname))
        norm_data = dat['normed_data']
        num_samples = len(norm_data)
        encoded_data = dat['encoded_data']
        # Convert list to float32
        tensor_all = torch.from_numpy(
            np.array(norm_data, dtype=np.float32))
        raw_tensor = torch.from_numpy(
            np.array(encoded_data, dtype=np.float32).reshape(num_samples, 1,
                                                             opt.batch_size,
                                                             opt.batch_size))
        save_dict['encoded_data'] = encoded_data
    else:
        try:
            fname = dataset_name
            dat = np.load(fname).item()
        except (FileNotFoundError, KeyError):
            fname = './logs/data/data_NS10000_IS64_type-{0}_rate{1}.npy'.format(
                dataset_name, array_id)
            dat = np.load(fname).item()
        print("Loaded dataset: {}".format(fname))
        binned_data = dat['binned_data']
        norm_data = dat['normed_data']
        num_samples = len(binned_data)
        # Save original binned data too
        save_dict['binned_data'] = binned_data
        # Convert list to float32
        tensor_all = torch.from_numpy(np.array(norm_data, dtype=np.float32))
        raw_tensor = torch.from_numpy(np.array(binned_data, dtype=np.float32))
    # Free space
    del dat
    # preprocess(tensor_all)
    # Define targets as ones
    # label smoothing, i.e. set labels to 0.9
    targets = torch.ones(num_samples) - 0.1
    # Create dataset
    ds = TensorDataset(tensor_all, targets)
    return ds, raw_tensor


if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
# Load datasets with type (step_rate | variability | pattern | ...)
else:
    print('loading data')
    t = time.time()
    dataset, tensor_raw = load_data(dataset_name=os.path.join(opt.dataroot,
                                                              opt.dataname),
                                    encoding=opt.encoding,
                                    array_id=ARRAY_ID)
    print('done loading data, in sec: {}'.format(time.time() - t))
    vutils.save_image(tensor_raw,
                      '{}/real_samples_normalized.png'.format(opt.outf),
                      normalize=True)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=int(opt.workers))


########################
# define models
########################
# TODO check if weights are correctly initialized
# custom weights initialization called on discriminator and generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('softmax') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self,
                 cell_type='LSTM',
                 num_layers=1,
                 state_size=64,
                 batch_size=opt.batch_size,
                 n_gpu=1):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.num_layers = num_layers
        self.state_size = state_size
        self.batch_size = batch_size

        # RNN
        if cell_type == 'Basic':
            self.rnn = nn.RNN(input_size=state_size, hidden_size=state_size,
                              num_layers=num_layers, nonlinearity='tanh')
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=state_size, hidden_size=state_size,
                               num_layers=num_layers)
        self.h0, self.c0 = init_hidden(num_layers, batch_size, state_size)

    def forward(self, rnn_inputs):
        # TODO careful here, check
        num_steps = rnn_inputs.shape[1]
        # TODO: beware, rnn_inputs needs to be a vector of
        # shape (seq_len, batch_input_size)
        output, hidden = self.rnn(self.rnn_inputs, (self.h0, self.c0))
        # add softmax layer
        # TODO check in corresponding tf code following line
        # also possible: self.softmax = nn.Softmax(dim=0);
        # out = self.softmax(output)
        out = F.softmax(output, dim=0)

        if not D_DIFF and G_DIFF:  # depend on D_DIFF
            W = torch.randn((self.state_size, 1))
            b = torch.zeros([1])
            # logits_t = torch.matmul(output, W) + b
            logits_t = torch.mm(output, W) + b
            logits_t = F.elu(logits_t) + 1
            logits_t = torch.cumsum(logits_t, dim=1)
            out = logits_t

        if MARK:
            W = torch.randn((self.state_size, 1))
            b = torch.zeros([1])
            logits_t = torch.mm(output, W) + b
            # redeclare W, b
            W = torch.randn((self.state_size, DIM_SIZE))
            b = torch.zero([DIM_SIZE])
            logits_prob = torch.mm(output, W) + b
            logits_prob = nn.Softmax(logits_prob)
            logits = torch.cat([logits_t, logits_prob], dim=1)
            logits.resize_(self.batch_size, num_steps, DIM_SIZE + 1)
            out = logits
        else:
            out.resize(self.batch_size, num_steps, 1)
        return out, hidden


def init_hidden(n_layers=1, mb_size=1, h_dim=64):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(n_layers, mb_size, h_dim),
            torch.zeros(n_layers, mb_size, h_dim))


class Discriminator(nn.Module):
    def __init__(self,  # dims batch_size x num_steps x input_size
                 lower_triangular,
                 cell_type='LSTM',
                 num_layers=1,
                 state_size=64,
                 batch_size=opt.batch_size,
                 cost_all=COST_ALL,
                 n_gpu=1,
                 ):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        self.num_layers = num_layers
        self.state_size = state_size
        self.batch_size = batch_size
        self.cost_all = cost_all
        self.lower_triangular_ones = lower_triangular

        keep_prob = 0.9

        # RNN
        if cell_type == 'Basic':
            self.rnn = nn.RNN(input_size=state_size, hidden_size=state_size,
                              num_layers=num_layers, nonlinearity='tanh')
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=state_size, hidden_size=state_size,
                               num_layers=num_layers, dropout=keep_prob)
        self.h0, self.c0 = init_hidden(num_layers, self.batch_size, state_size)
        self.dropout = nn.Dropout(p=keep_prob)

    def forward(self, rnn_inputs, seq_len):
        # TODO careful here, check
        num_steps = rnn_inputs.shape[1]

        output, hidden = self.rnn(self.rnn_inputs, (self.h0, self.c0))

        # Add dropout
        rnn_outputs = self.dropout(output)
        # reshape rnn_outputs
        rnn_outputs = torch.reshape(rnn_outputs, (-1, self.state_size))

        # TODO try with softmax layer from nn package
        # Softmax layer
        W = torch.randn((self.state_size, 1))
        b = torch.zeros([1])
        # logits = torch.matmul(rnn_outputs, W) + b
        logits = torch.mm(rnn_outputs, W) + b

        # TODO check if correct with start and length
        seqlen_mask = torch.gather(
            input=torch.Tensor(self.lower_triangular_ones),
            index=seq_len - 1, dim=0).narrow(dim=0, start=self.batch_size,
                                             length=num_steps)

        if self.cost_all:
            logits = torch.reshape(logits, (self.batch_size, num_steps))
            logits *= seqlen_mask
            # Average over actual sequence lengths.
            f_val = torch.sum(input=logits, dim=1)
            f_val /= torch.sum(input=seqlen_mask, dim=1)
        else:
            # Select the Last Relevant Output
            index = torch.range(0, self.batch_size) * num_steps + (seq_len - 1)
            flat = torch.reshape(logits, (-1, 1))
            relevant = torch.gather(flat, index=index, dim=0)
            f_val = torch.reshape(relevant, self.batch_size)
        return f_val


def make_one_hot(labels, n_class=2):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.LongTensor or torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    n_class : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    if opt.gpu:
        y_one_hot = torch.cuda.FloatTensor(labels.size(0), n_class,
                                           labels.size(2),
                                           labels.size(3)).zero_()
    else:
        y_one_hot = torch.FloatTensor(labels.size(0), n_class, labels.size(2),
                                      labels.size(3)).zero_()
    target = y_one_hot.scatter_(1, labels.data, 1)
    return target


def one_hot(batch_size, classes, dim=1):
    y = torch.LongTensor(batch_size, 1).random_() % classes
    y_one_hot = torch.FloatTensor(batch_size, classes)
    return y_one_hot.scatter_(dim, y, 1)


def lipschitz_divergence(real_data, fake_data, d_real, d_fake):
    length_ = min(real_data.shape[1], fake_data.shape[1])
    # Apply the Lipschitz constraint
    # TODO check/correct dimensions
    ld = torch.abs(d_real - d_fake / torch.sqrt(torch.sum((real_data[:, :length_, :] - fake_data[:, :length_, :])**2) + 0.00001))
    ld = torch.mean((ld - 1) ** 2)
    return ld


lower_triangular_ones = torch.FloatTensor(
    np.tril(np.ones([opt.max_steps, opt.max_steps])))

discriminator = Discriminator(lower_triangular_ones, n_gpu=int(opt.ngpu))
discriminator.apply(weights_init)

generator = Generator(n_gpu=int(opt.ngpu))
generator.apply(weights_init)

if opt.gen != '':
    generator.load_state_dict(torch.load(opt.gen))
print(generator)
if opt.disc != '':
    discriminator.load_state_dict(torch.load(opt.disc))
print(discriminator)


# TODO: define labels, noise, optimizer (as empty tensors)
fake_seqlen = torch.IntTensor(opt.batch_size)
real_seqlen = torch.IntTensor(opt.batch_size)
label = torch.FloatTensor(opt.batch_size)
# label smoothing
real_label = 0.9    # before 1.0
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1
fixed_noise = torch.FloatTensor(opt.batch_size)

if opt.cuda:
    discriminator.cuda()
    generator.cuda()
    # criterionD.cuda()
    # criterionG.cuda()
    label = label.cuda()
    # input_, = input_.cuda()
    # noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    one = one.cuda()
    mone = mone.cuda()

# setup optimizer
optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

# # Pre train options #
# pre train loss
pre_train_loss = nn.L1Loss()
# pre train optimizers
pre_train_optimG = optim.RMSprop(generator.parameters(), lr=5e-5)
pre_train_optimD = optim.RMSprop(discriminator.parameters(), lr=5e-5)

# TODO training steps, everything else comes below here
if MARK:
    # TODO
    z_one_hot = make_one_hot()
    # Z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 2]) # time,dim
    # Z_one_hot = tf.one_hot( tf.cast(Z[:,:,1],tf.int32),DIM_SIZE )
    # Z_all = tf.concat([Z[:,:,:1],Z_one_hot],axis=2)
else:
    # TODO check if correct shape of [batch_size, num_steps]
    z_all = torch.Tensor(opt.batch_size, 1)
# if MARK:
#     Z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 2]) # time,dim
#     Z_one_hot = tf.one_hot( tf.cast(Z[:,:,1],tf.int32),DIM_SIZE )
#     Z_all = tf.concat([Z[:,:,:1],Z_one_hot],axis=2)
# else:
#     Z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 1]) # [batch_size, num_steps]
#     Z_all = Z
#
# fake_seqlen = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
# fake_data = generator(Z_all,fake_seqlen)
# if MARK:
#     fake_data_discrete = tf.argmax(fake_data[:,:,1:], axis=2) #
#
# if MARK:
#     X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 2])
#     X_one_hot = tf.one_hot( tf.cast(X[:,:,1],tf.int32),DIM_SIZE) # one_hot depth on_value off_value
#     real_data = tf.concat([X[:,:,:1],X_one_hot],axis=2)
# else:
#     X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, 1])
#     real_data = X
#
# real_mask = torch.gather(input=torch.Tensor(lower_triangular_ones),
#                          index=real_seqlen - 1, dim=0).narrow(dim=0,
#                                                               start=opt.batch_size,
#                                                               length=real_data.shape[1])
# fake_mask = tf.slice(tf.gather(lower_triangular_ones, fake_seqlen - 1),[0, 0], [BATCH_SIZE,tf.shape(fake_data)[1]])
# real_mask = tf.expand_dims(real_mask,-1)
# fake_mask = tf.expand_dims(fake_mask,-1)

# saved_file = "wgan_{}_{}_{}_{}_{}_{}_{}".format(opt.data, opt.seq_num,
#                                                 opt.epochs, opt.lambda_lp,
#                                                 datetime.now().day,
#                                                 datetime.now().hour,
#                                                 datetime.now().minute)
# if not os.path.exists('out/%s' % saved_file):
#     os.makedirs('out/%s' % saved_file)

stop_indicator = False
n_t = 30
# ts_real, intensity_real = get_intensity(real_sequences, T, n_t)

# pre-train
if PRE_TRAIN:
    pre_stop = 40
    for i, data in enumerate(dataloader):
        # clear gradients
        discriminator.zero_grad()

        # Train discriminator
        pre_disc_input = data
        if opt.cuda:
            pre_disc_input = pre_disc_input.cuda()
        # TODO could also try to wrap with torch.Tensor
        pre_disc_inputv = Variable(pre_disc_input)
        # Train with real labels
        label.resize_(opt.batch_size).fill_(real_label)
        pre_outputD = discriminator(pre_disc_inputv, pre_disc_inputv.shape[1])
        pre_errD_real = pre_train_loss(pre_outputD, label)
        pre_errD_real.backward()
        pre_D_x = pre_outputD.data.mean()
        # TODO save
        print(pre_D_x)

        # Train with fake labels
        label.fill_(fake_label)
        noise = DataDistribution.poisson_stationary_sample(10, 6000,
                                                           binned=False)
        with torch.no_grad:
            # totally freeze G, training D
            noise = torch.from_numpy(np.array(noise[0]))
        # generate fake with generator
        pre_fake = generator(noise)
        output = discriminator(pre_fake.detach())
        pre_errD_fake = pre_train_loss(output, label)
        pre_errD_fake.backward()
        pre_D_G_z1 = output.data.mean()
        pre_errD = pre_errD_real + pre_D_G_z1
        pre_train_optimD.step()
        # TODO save pre train values

        # Train Generator
        # clear gradients
        generator.zero_grad()
        label = label.fill_(real_label)
        output = discriminator(pre_fake)
        pre_errG = pre_train_loss(output, label)
        pre_errG.backward()
        pre_D_G_z2 = output.data.mean()
        pre_train_optimG.step()

        if i % 10 == 0:
            print('[{}/{}] pre_train_loss: pre_errD {}, pre_errG {}, '
                  'pre_D_x {}, pre_G_z1 {}, pre_G_z2 {}'.format(
                      i, pre_stop, pre_errD.data[0], pre_errG.data[0], pre_D_x,
                      pre_D_G_z1, pre_D_G_z2))

        # stop pre training
        if i == pre_stop:
            break

# Train the GAN
for epoch in range(opt.epochs):
    # also possible data_iter = iter(dataloader); data = data_iter.next(),
    # then add random sampler
    for i, data in enumerate(dataloader):
        # reset requires_grad
        # set to False below in generator update
        for p in discriminator.parameters():
            p.requires_grad = False
        # clear gradients
        discriminator.zero_grad()

        # Train discriminator (critic)
        # TODO maybe train the discriminator a few steps more (critic_iters)
        input_ = data
        if opt.cuda:
            input_ = input_.cuda()
        inputv = Variable(input_)
        # Train with real
        label.resize_(opt.batch_size).fill_(real_label)
        errD_real = discriminator(inputv, inputv.shape[1])
        errD_real = errD_real.mean()
        errD_real.backward(mone)

        # Train with fake
        noise = DataDistribution.poisson_stationary_sample(10, 6000,
                                                           binned=False)
        with torch.no_grad:
            # totally freeze G, training D
            noise = torch.from_numpy(np.array(noise[0]))
        # generate fake with generator
        fake = generator(noise)
        errD_fake = discriminator(fake.detach())
        errD_fake = errD_fake.mean()
        errD_fake.backward(one)
        lipschitz_div = lipschitz_divergence(inputv, noise, errD_real,
                                             errD_fake)
        errD = errD_fake - errD_real + opt.labda_lp * lipschitz_div
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in discriminator.parameters():
            p.requires_grad = False
        generator.zero_grad()
        errG = discriminator(fake)
        errG = errG.mean()
        errG.backward(mone)
        gen_cost = -errG
        optimizerG.step()

        print('[{}/{}] train_loss: errD {}, errG {}, errD_fake {}'
              ', errD_real {} gen_cost {}'.format(i, len(dataloader),
                                                  errD.data[0],
                                                  errG.data[0], errD_fake,
                                                  errD_real, gen_cost))
        if i % 100 == 0:
            fake = generator(fixed_noise)
            vutils.save_image(fake.data,
                              '%s/fake_samples_normalized_epoch_%03d.png' % (
                                  opt.outf, epoch),
                              normalize=True)
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d.png' % (
                                  opt.outf, epoch),
                              normalize=False)

    # do checkpointing
    checkpoint_path = os.path.join(opt.outf, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(generator.state_dict(),
               '%s/generator_epoch_%d.pth' % (checkpoint_path, epoch))
    torch.save(discriminator.state_dict(),
               '%s/discriminator_epoch_%d.pth' % (checkpoint_path, epoch))

outf = './logs/run_{}_rate{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), ARRAY_ID)
with open(os.path.join(outf, 'params.yaml'), 'w') as f:
    yaml.dump(params, f)
