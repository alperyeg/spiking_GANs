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

from datetime import datetime
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter

# TODO change to config file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    help='cifar10 | lsun | imagenet | folder | lfw | fake | '
                         'step_rate | variability | pattern',
                    default='step_rate')
parser.add_argument('--lambda_lp', required=False, default=0.1,
                    help='Penalty for Lipschtiz divergence')
parser.add_argument('--critic_iters', required=False, default=5,
                    help='How many critic iterations per generator iteration')
parser.add_argument('--batch_size', required=False, default=256)
parser.add_argument('--max_steps', required=False, default=300)
parser.add_argument('--iters', required=False, default=2000,
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


if opt.data in ['mimic', 'meme', 'citation', 'stock', "mixture1", "mixture2",
                "mixture3", "mixture4"]:
    REAL_DATA = True
else:
    REAL_DATA = False

if opt.manualSeed is None:
    opt.manualSeed = 123
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

# TODO prepare or load data


########################
# define models
########################
# TODO check if weights are correctly initialized
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, rnn_inputs,
                 seq_len,
                 cell_type='LSTM',
                 num_layers=1,
                 state_size=64,
                 batch_size=opt.batch_size,
                 n_gpu=1):
        super(Generator, self).__init__()
        self.rnn_inputs = rnn_inputs
        self.n_gpu = n_gpu
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.state_size = state_size
        self.batch_size = batch_size
        # TODO careful here, check
        self.num_steps = rnn_inputs.shape[1]

        # RNN
        if cell_type == 'Basic':
            self.rnn = nn.RNN(input_size=state_size, hidden_size=state_size,
                              num_layers=num_layers, nonlinearity='tanh')
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=state_size, hidden_size=state_size,
                               num_layers=num_layers, nonlinearity='tanh')
        self.h0, self.c0 = self.hidden(num_layers, self.batch_size, state_size)

    def forward(self):
        # TODO: beware, rnn_inputs needs to be a vector of
        # shape (seq_len, batch_input_size)
        output, hidden = self.rnn(self.rnn_inputs, (self.h0, self.c0))
        # add softmax layer
        # TODO check in corresponding tf code following line
        out = F.softmax(output, dim=0)

        if not D_DIFF and G_DIFF:  # depend on D_DIFF
            W = torch.FloatTensor(self.state_size, 1)
            b = torch.zeros([1])
            logits_t = torch.matmul(output, W) + b
            logits_t = F.elu(logits_t) + 1
            logits_t = torch.cumsum(logits_t, dim=1)
            out = logits_t

        if MARK:
            W = torch.FloatTensor(self.state_size, 1)
            b = torch.zeros([1])
            logits_t = torch.matmul(output, W) + b
            # redeclare W, b
            W = torch.FloatTensor(self.state_size, DIM_SIZE)
            b = torch.FloatTensor([DIM_SIZE])
            nn.init.constant(b, 0)
            logits_prob = torch.matmul(output, W) + b
            logits_prob = nn.Softmax(logits_prob)
            logits = torch.cat([logits_t, logits_prob], dim=1)
            logits.resize_(self.batch_size, self.num_steps, DIM_SIZE + 1)
            out = logits
        else:
            out.resize(self.batch_size, self.num_steps, 1)
        return out, hidden

    @staticmethod
    def init_hidden(n_layers=1, mb_size=1, h_dim=64):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(n_layers, mb_size, h_dim),
                torch.zeros(n_layers, mb_size, h_dim))


class Discriminator(nn.Module):
    def __init__(self, rnn_inputs,  # dims batch_size x num_steps x input_size
                 seq_len,
                 lower_triangular_ones,
                 cell_type='LSTM',
                 num_layers=1,
                 state_size=64,
                 batch_size=opt.batch_size,
                 cost_all=COST_ALL,
                 n_gpu=1):
        super(Discriminator, self).__init__()
        self.rnn_inputs = rnn_inputs
        self.n_gpu = n_gpu
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.state_size = state_size
        self.batch_size = batch_size
        self.cost_all = cost_all

        # TODO careful here, check
        self.num_steps = rnn_inputs.shape[1]
        keep_prob = torch.FloatTensor([0.9])

        # RNN
        if cell_type == 'Basic':
            self.rnn = nn.RNN(input_size=state_size, hidden_size=state_size,
                              num_layers=num_layers, nonlinearity='tanh')
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=state_size, hidden_size=state_size,
                               num_layers=num_layers, nonlinearity='tanh')
        self.h0, self.c0 = self.hidden(num_layers, self.batch_size, state_size)

    def forward(self, *input):


netG = Generator(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
