from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import utils
import datetime
import time

from torch.utils.data import TensorDataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter

JOB_ID = int(os.environ['SLURM_JOB_ID'])
ARRAY_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
# ARRAY_ID = 12

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    help='cifar10 | lsun | imagenet | folder | lfw | fake | '
                         'step_rate | variability',
                    default='step_rate')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=30,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.4,
                    help='beta1 for adam. default=0.4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--outf', default='./logs/run_{}_rate{}'.format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), ARRAY_ID),
    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError as ose:
    print(ose)

if opt.manualSeed is None:
    opt.manualSeed = 123
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

# Define a dictionary to save results
save_dict = {}

# Define writer object for tensorboard
writer = SummaryWriter(log_dir=os.path.join(opt.outf, 'tensorboard'))

if torch.cuda.is_available() and not opt.cuda:
    print(
        "WARNING: You have a CUDA device, so you should probably run with --cuda")

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
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    print('in cifar10 dataset')
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
    nc = 3
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())

# Load datasets with type (step_rate | variability)
else:
    print('loading data')
    t = time.time()
    try:
        fname = './data_NS10000_IS64_type-step_rate_rate{}.npy'.format(
            ARRAY_ID)
        data = np.load(fname).item()
        binned_data = data['binned_data']
        norm_data = data['normed_data']
    except (FileNotFoundError, KeyError):
        fname = './logs/data/data_NS10000_IS64_type-step-rate.npy'
        data = np.load(fname).item()
        binned_data = data['binned_data']
        norm_data = data['normed_data']
    print('done loading data, in sec: {}'.format(time.time() - t))
    num_samples = len(binned_data)
    # Save original binned data too
    save_dict['binned_data'] = binned_data
    # Free space
    del data
    # Convert list to float32
    tensor_all = torch.from_numpy(np.array(norm_data, dtype=np.float32))
    tensor_raw = torch.from_numpy(np.array(binned_data, dtype=np.float32))
    # preprocess(tensor_all)
    # Define targets as ones
    # TODO try label smoothing, i.e. set labels e.g. to 0.9
    targets = torch.ones(num_samples)
    # Create dataset
    dataset = TensorDataset(tensor_all, targets)
    nc = 1

    vutils.save_image(tensor_raw,
                      '{}/real_samples_normalized.png'.format(opt.outf),
                      normalize=True)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True,
                                         num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            out = self.main(inpt)
        return out


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _net_D(nn.Module):
    def __init__(self, ngpu):
        super(_net_D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inpt):
        if isinstance(inpt.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, inpt, range(self.ngpu))
        else:
            out = self.main(inpt)

        return out.view(-1, 1).squeeze(1)


netD = _net_D(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input_ = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
# fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
fixed_noise = np.random.poisson(lam=2, size=(
    opt.batchSize, nz, 1, 1)).astype(np.float32)
fixed_noise = torch.from_numpy(np.divide(fixed_noise, np.max(fixed_noise)))
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input_, label = input_.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# New dictionary entries to save
save_dict['D_x'] = []
save_dict['G_z1'] = []
save_dict['G_z2'] = []
save_dict['errD'] = []
save_dict['errD_fake'] = []
save_dict['errD_real'] = []
save_dict['errG'] = []
save_dict['fake_data'] = []


for epoch in range(opt.niter):
    # Create lists per epochs
    [save_dict[elem].append([epoch]) for elem in save_dict if
     not elem in ['binned_data', 'num_samples']]
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input_.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input_)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        save_dict['D_x'][epoch].append((i, output.data))
        writer.add_scalar('data/D_x', D_x, epoch)
        writer.add_histogram('histogram/D_x', output.data, epoch)

        # train with fake
        noise = np.random.poisson(lam=2, size=(
            batch_size, nz, 1, 1)).astype(np.float32)
        noise = torch.from_numpy(np.divide(noise, noise.max()))
        noise.resize_(batch_size, nz, 1, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        save_dict['G_z1'][epoch].append((i, output.data))
        save_dict['errD'][epoch].append((i, errD.data[0]))
        save_dict['errD_real'][epoch].append((i, errD_real.data[0]))
        save_dict['errD_fake'][epoch].append((i, errD_fake.data[0]))
        writer.add_scalar('data/G_z1', D_G_z1, epoch)
        writer.add_scalar('data/errD', errD.data[0], epoch)
        writer.add_scalar('data/errD_real', errD_real.data[0], epoch)
        writer.add_scalar('data/errD_fake', errD_fake.data[0], epoch)
        writer.add_scalar('data/G_z1', D_G_z1, epoch)
        writer.add_histogram('histogram/G_z1', output.data, epoch)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(
            label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        save_dict['G_z2'][epoch].append((i, output.data))
        save_dict['errG'][epoch].append((i, errG.data[0]))
        writer.add_scalar('data/errG', errG.data[0], epoch)
        writer.add_scalar('data/G_z2', D_G_z2, epoch)
        writer.add_histogram('histogram/G_z2', output.data, epoch)

        print(
            '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, opt.niter, i, len(dataloader),
               errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                              '%s/fake_samples_normalized_epoch_%03d.png' % (
                                  opt.outf, epoch),
                              normalize=True)
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d.png' % (
                                  opt.outf, epoch),
                              normalize=False)
            '''
            1-dim: list with all the data, listed according the epochs
            2-dim: list containing lists of integer and tuple, 
                integer indicates the epoch, the tuple contains the step index 
                and the output data,
                [int, tuple]
            3-dim: tuple of integer and data as torch.FloatTensor, the integer
                indicates the step index of the corresponding batch in the loop
                (int, FloatTensor)
            4-dim: FloatTensor of shape 64x1x64x64: 
            5-dim: 
                64 samples x 
                Channel number (here always only 1) x 
                64x64 normalized binned data  
            '''
            save_dict['fake_data'][epoch].append((i, fake.data))

    # do checkpointing
    checkpoint_path = os.path.join(opt.outf, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (checkpoint_path,
                                                            epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (checkpoint_path,
                                                            epoch))
utils.save_samples(save_dict, path=opt.outf, filename='results.npy')
writer.close()
