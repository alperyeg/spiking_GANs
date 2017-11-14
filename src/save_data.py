import numpy as np
import neo 
import torch
import time
import quantities as pq
import utils
from data_generation import DataDistribution

'''
Supported data types are so far (step_rate | variability) 
'''
data_type = 'variability'

# Number of data samples
num_samples = 10000
imageSize = 64
save_dict = {}
print('generating data')
t = time.time()
# Matrices to store, shape: samples x C x H x W
# C: channels, H: height, W: width
binned_data = np.empty((num_samples, 1, imageSize, imageSize),
                       dtype=np.float32)
norm_data = np.empty((num_samples, 1, imageSize, imageSize),
                     dtype=np.float32)
raw_data = []
for i in range(num_samples):
    if data_type == 'step_rate':
        data, spikes, _ = DataDistribution.poisson_nonstat_sample(
            t_stop=10000 * pq.ms,
            num_bins=64,
            num_sts=64)
    elif data_type == 'variability':
        data, spikes = DataDistribution.gen_nonstat_sample(6, t=10000 * pq.ms,
                                                           sample_period=10 * pq.ms,
                                                           num_bins=64,
                                                           num_sts=64)
    data = data.to_array().ravel()
    raw_data.append(spikes)
    # Reshape to required format
    data = data.reshape((1, imageSize, imageSize))
    binned_data[i] = data
    # Normalize data
    data = np.divide(data, np.max(data))
    # data = (data - data.mean()) / data.std()
    norm_data[i] = data
print('done generating data, in sec: {}'.format(time.time() - t))

save_dict['binned_data'] = binned_data
save_dict['normed_data'] = norm_data
save_dict['num_samples'] = num_samples
save_dict['imageSize'] = imageSize
save_dict['spikes'] = raw_data
save_dict['data_type'] = data_type
utils.save_samples(save_dict, path='.', filename='data_NS{}_IS{}_type-{}.npy'.format(num_samples, 
                                                                                     imageSize, data_type))
