import numpy as np
import time
import quantities as pq
import utils
from mpi4py import MPI
from data_generation import DataDistribution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

'''
Supported data types are so far (step_rate | variability)
'''
data_type = 'variability'
# Number of data samples
num_samples = 10000
imageSize = 64
save_dict = {}
# print('generating data')
# t = time.time()

# Matrices to store, shape: samples x C x H x W
# C: channels, H: height, W: width
binned_data = np.empty((num_samples, 1, imageSize, imageSize),
                       dtype=np.float32)
recv_data = np.empty((num_samples, 1, imageSize, imageSize),
                     dtype=np.float32)
# raw_data = []
if rank != 0:
    for i in range((rank - 1) * 42, (rank) * 42): 
        if i == 10000:
            break
        if i % 10 == 0:
            print(rank, i)
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
            # raw_data.append(spikes)
            # Reshape to required format
            data = data.reshape((1, imageSize, imageSize))
            recv_data[rank - 1] = data
            comm.Send(recv_data, dest=0)
# print('done generating data, in sec: {}'.format(time.time() - t))

if rank == 0:
    for i in range(1, size):
        comm.Recv(recv_data, source=i)
        binned_data[i - 1] = recv_data[i - 1]

if comm.rank == 0:
    normed_data = np.empty((num_samples, 1, imageSize, imageSize),
                           dtype=np.float32)
    for i, bst in enumerate(binned_data):
        # Normalize data
        normed_data[i] = np.divide(bst, np.max(bst))
        # data = (data - data.mean()) / data.std()
        
    save_dict['binned_data'] = binned_data
    save_dict['normed_data'] = normed_data
    save_dict['num_samples'] = num_samples
    save_dict['imageSize'] = imageSize
    # save_dict['spikes'] = raw_data
    save_dict['data_type'] = data_type
    utils.save_samples(save_dict, path='.', filename='data_NS{}_IS{}_type-{}.npy'.format(num_samples,
                                                                                         imageSize, data_type))
