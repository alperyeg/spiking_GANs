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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from datetime import datetime
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    help='cifar10 | lsun | imagenet | folder | lfw | fake | '
                         'step_rate | variability | pattern',
                    default='step_rate')
parser.add_argument('--lambda_lp', required=False, default=0.1,
                    help='Penality for Lipschtiz divergence')
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
print(opt)

# other parameters
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
                 n_gpu=1):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.seqlen = seq_len
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.state_size = state_size
        self.batch_size = opt.batch_size
        self.num_steps = rnn_inputs.shape[1]

        # RNN
        if cell_type == 'Basic':
            cell = nn.RNN(state_size * num_layers)
            cell = tf.contrib.rnn.BasicRNNCell(state_size)
        elif cell_type == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(state_size,
                                           state_is_tuple=True)  # tuple of c_state and m_state

        if cell_type == 'LSTM':
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                               state_is_tuple=True)
        elif cell_type == 'Basic':
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                               state_is_tuple=False)

        # self.main = nn.Sequential

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            out = self.main(inpt)
        return out


netG = Generator(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
