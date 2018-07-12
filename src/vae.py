from __future__ import print_function
import argparse
import torch
import torch.utils.data
import yaml
import os
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset


# parameters
params = {}
with open('config.yaml', 'r') as stream:
    try:
        params = yaml.load(stream)
        print(params)
    except yaml.YAMLError as err:
        print(err)

BATCH_SIZE = params['batch_size']  # Batch size
MAX_STEPS = params['max_steps']
ITERS = params['iters']  # how many generator iterations to train for
SEED = params['manualSeed']  # set graph-level seed
DATA_PATH = os.path.join(params['dataroot'], params['dataname'])
EPOCHS = params['epoch']
ENCODE = params['encoding']
IMAGESIZE = params['imageSize']
log_interval = 10

# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a dictionary to save results
save_dict = {}


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
        tensor_train = torch.from_numpy(
            np.array(norm_data[:int(len(norm_data)/2)], dtype=np.float32))
        tensor_test = torch.from_numpy(
            np.array(norm_data[int(len(norm_data)/2):], dtype=np.float32))

        # tensor_all = torch.from_numpy(
        #     np.array(norm_data, dtype=np.float32))
        raw_tensor = torch.from_numpy(
            np.array(encoded_data, dtype=np.float32).reshape(num_samples, 1,
                                                             IMAGESIZE,
                                                             IMAGESIZE))
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
        # tensor_all = torch.from_numpy(np.array(norm_data, dtype=np.float32))
        raw_tensor = torch.from_numpy(np.array(binned_data, dtype=np.float32))
    # Free space
    del dat
    # preprocess(tensor_all)
    # Define targets as ones
    # label smoothing, i.e. set labels to 0.9
    # targets = torch.ones(num_samples) - 0.1
    # Create dataset
    ds_train = TensorDataset(tensor_train)
    ds_test = TensorDataset(tensor_test)
    return ds_train, ds_test, raw_tensor


kwargs = {'num_workers': 1,
          'pin_memory': True} if torch.cuda.is_available else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('logs/data/mnist', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=BATCH_SIZE, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('logs/data/mnist', train=False,
#                    transform=transforms.ToTensor()),
#     batch_size=BATCH_SIZE, shuffle=True, **kwargs)

train_data, test_data, raw = load_data(DATA_PATH, encoding=ENCODE)
assert train_data, 'train_data is empty'
assert test_data, 'test_data is empty'
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                          shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data[0].to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                if 1000 % epoch == 0:
                    save_image(comparison.cpu(),
                               'logs/figures/reconstruction_' + str(epoch) + '.png', nrow=n)
                    np.save('reconstruction_{}.npy'.format(epoch),
                            comparison.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(BATCH_SIZE, 20).to(device)
        sample = model.decode(sample).cpu()
        if 1000 % epoch == 0:
            np.save('sample_{}.npy'.format(epoch),
                    sample.numpy())
            save_image(sample.view(BATCH_SIZE, 1, 28, 28),
                       'logs/figures/sample_' + str(epoch) + '.png')
