import numpy as np
import os
import time
import quantities as pq
import utils
import argparse
import uuid
import multiprocessing as mp
from data_generation import DataDistribution
from STP_generation import generate_sts
from utils import encoder


parser = argparse.ArgumentParser()
parser.add_argument('--data_type', required=False, type=str,
                    help='kind of rate (step_rate | variability | pattern)',
                    default='step_rate')
parser.add_argument('--encoding', required=False,
                    help='if input should be encoded',
                    default=False, action='store_true')
parser.add_argument('--generate', required=True, type=bool,
                    help='if the data should be generated',
                    default=True)
parser.add_argument('--filename', required=False, type=str,
                    help='Name of the file to be saved')
parser.add_argument('--path', required=False, type=str, default='logs/data',
                    help='Path of the file to be saved')
# parser.add_argument('--xi', required=False, type=int, help='Pattern size if'
#                     '`data_type` is `pattern`', default=15)

opt = parser.parse_args()
print(opt)

'''
Supported data types are so far (step_rate | variability | pattern)
'''

# Number of data samples
num_samples = 10000
imageSize = 32
xi = 15
save_dict = {}
try:
    JOB_ID = int(os.environ['SLURM_JOB_ID'])
    ARRAY_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    ARRAY_ID = 10
print(ARRAY_ID)
print('generating data')
t = time.time()

# Matrices to store, shape: samples x C x H x W
# C: channels, H: height, W: width
binned_data = np.empty((num_samples, 1, imageSize, imageSize),
                       dtype=np.float32)
norm_data = np.empty((num_samples, 1, imageSize, imageSize),
                     dtype=np.float32)
raw_data = []
encoded_data = []


def generate_data(data_type, index, encode=False):
    d = None
    if data_type == 'step_rate':
        d = DataDistribution.poisson_nonstat_sample(
            t_stop=3000 * pq.ms,
            rate2=ARRAY_ID * pq.ms,
            num_bins=64,
            num_sts=64,
            binned=(not encode),
            dt=3 * pq.ms)
    elif data_type == 'variability':
        d = DataDistribution.gen_nonstat_sample(6, t=10000 * pq.ms,
                                                sample_period=10 * pq.ms,
                                                num_bins=64,
                                                num_sts=64,
                                                binned=(not encode))
    elif data_type == 'pattern':
        np.random.seed(index)
        d = DataDistribution.generate_stp_data(n_neurons=1, rate=10 * pq.Hz,
                                               occurr=7, xi=xi,
                                               t_stop=6 * pq.s,
                                               delay=0 * pq.ms)
        d = d['patterns']
        # sts = generate_sts(data_type=6, T=6000 * pq.ms, N=1)[0]
        # sts_final = stp + sts
        # d = d['data']
    return d


def create_norm_data(encoded_data, method='None'):
        norm_data = np.zeros((len(encoded_data), 1, imageSize, imageSize),
                             dtype=np.float32)
        norm_value = 0
        for ed in encoded_data:
            # calc max value in the data set
            max_val = np.max(ed)
            if max_val > norm_value:
                norm_value = max_val
        np.random.seed(0)
        # normalize
        for i, ed in enumerate(encoded_data):
            if method == 'inject':
                # inject in a random position but together the arrays
                # inbetween the matrix
                rnd = np.random.randint(low=0, high=(imageSize - xi))
                norm_data[i][0][rnd:rnd + ed.shape[0]] = np.divide(ed, norm_value)
            else:
                dat = np.array(ed).reshape((1, xi, imageSize))
                norm_data[i] = np.divide(dat, norm_value)
        return norm_data, norm_value


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        t = time.time()
        if opt.generate:
            data_all = [pool.apply_async(generate_data, args=(opt.data_type, i,
                                                              opt.encoding))
                        for i in range(num_samples)]
            # data = generate_data(data_type=opt.data_type,
            # encode=opt.encoding)
            if opt.encoding:
                for data in data_all:
                    raw_data.append(data.get())
                    encoded_data.extend(
                        encoder(data.get(), imageSize, 200 * pq.ms,
                                min_spikes=7, start_val=0))
            else:
                for i, data in enumerate(data_all):
                    dat = data[0].to_array().ravel()
                    raw_data.append(data[1])
                    # Reshape to required format
                    dat = dat.reshape((1, imageSize, imageSize))
                    binned_data[i] = dat
                    # Normalize data
                    dat = np.divide(dat, np.max(dat))
                    # data = (data - data.mean()) / data.std()
                    norm_data[i] = dat
        else:
            # TODO load data
            pass
        print('done generating data, in sec: {}'.format(time.time() - t))

    if opt.encoding:
        # normalize
        norm_data, norm_value = create_norm_data(encoded_data, method='inject')
        # save
        save_dict['normed_data'] = norm_data
        save_dict['num_samples'] = num_samples
        save_dict['imageSize'] = imageSize
        save_dict['spikes'] = raw_data[:100]
        save_dict['encoded_data'] = encoded_data
        save_dict['normed_value'] = norm_value

    else:
        save_dict['binned_data'] = binned_data
        save_dict['normed_data'] = norm_data
        save_dict['num_samples'] = num_samples
        save_dict['imageSize'] = imageSize
        save_dict['spikes'] = raw_data
        save_dict['data_type'] = opt.data_type

    if opt.filename:
        fname = opt.filename
    else:
        fname = 'train-data_NS{}_IS{}_type-{}_encoded-{}_rate{}.npy'.format(
            num_samples, imageSize, opt.data_type, opt.encoding,
            ARRAY_ID)
    u = str(uuid.uuid4())
    path = os.path.join(opt.path, u)
    if not os.path.exists(path):
        os.makedirs(path)
    utils.save_samples(save_dict, path=path, filename=fname)
