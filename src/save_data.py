import numpy as np
import os
import time
import quantities as pq
import utils
import argparse
from data_generation import DataDistribution
from utils import encoder


parser = argparse.ArgumentParser()
parser.add_argument('--data_type', required=False, type=str,
                    help='kind of rate (step_rate | variability)',
                    default='step_rate')
parser.add_argument('--encoding', required=False, type=str,
                    help='if input should be encoded',
                    default=False)
parser.add_argument('--generate', required=True, type=bool,
                    help='if the data should be generated',
                    default=True)

opt = parser.parse_args()

'''
Supported data types are so far (step_rate | variability)
'''

# Number of data samples
num_samples = 10000
imageSize = 64
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


def generate_data(data_type, encode=False):
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
    return d


for i in range(num_samples):
    if opt.generate:
        data = generate_data(data_type=opt.data_type, encode=opt.encoding)
        if opt.encoding:
            raw_data.append(data)
            encoded_data.extend(encoder(data, imageSize, 500 * pq.ms, 10))
        else:
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
    norm_data = np.empty((len(encoded_data), 1, imageSize, imageSize),
                         dtype=np.float32)
    norm_value = []
    for i, ed in enumerate(encoded_data):
        dat = np.array(ed).reshape((1, imageSize, imageSize))
        norm_data[i] = np.divide(dat, np.max(dat))
        norm_value.append(np.max(dat))
    save_dict['normed_data'] = norm_data
    save_dict['num_samples'] = num_samples
    save_dict['imageSize'] = imageSize
    save_dict['spikes'] = raw_data
    save_dict['encoded_data'] = encoded_data
    save_dict['normed_values'] = norm_value

else:
    save_dict['binned_data'] = binned_data
    save_dict['normed_data'] = norm_data
    save_dict['num_samples'] = num_samples
    save_dict['imageSize'] = imageSize
    save_dict['spikes'] = raw_data
    save_dict['data_type'] = opt.data_type
utils.save_samples(save_dict, path='.',
                   filename='data_NS{}_IS{}_type-{}_encoded-{}_rate{}.npy'.format(
                       num_samples, imageSize, opt.data_type, opt.encoding,
                       ARRAY_ID))
