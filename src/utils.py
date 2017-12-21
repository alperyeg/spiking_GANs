import os
import numpy as np
import tensorflow as tf
import quantities as pq
import neo


def save(ckpt_dir, step, saver, sess, model_name):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    saver.save(sess,
               os.path.join(ckpt_dir, model_name),
               global_step=step)


def load(ckpt_dir, saver, sess):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        return True
    else:
        return False


def save_samples(samples, path, filename):
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(os.path.join(path, filename), samples)


def convert_to_spiketrains(binned_data, binsize, rho, units='ms'):
    """
    Convert given `binned_data` to a `neo.SpikeTrain`

    :param binned_data: The binned and generated data as a numpy matrix
    :param binsize: Size of the bin, of the original data
    :param rho: Scale up factor, max normalizing coefficient
    :param units: quantities, Default is 'ms'
    :return spiketrains: Returns scaled up spike trains from generated data
    """
    spiketrains = []
    binsize = binsize.rescale(units).magnitude
    for row in binned_data:
        actual_bin = 0
        st = []
        for b in row:
            upscaled_bin = np.abs(int(b * rho))
            st.extend(np.random.uniform(low=actual_bin,
                                        high=actual_bin + binsize,
                                        size=upscaled_bin))
            actual_bin += binsize
        neo_spiketrain = neo.SpikeTrain(st, t_start=0, t_stop=actual_bin,
                                        units=units)
        spiketrains.append(neo_spiketrain)
    return spiketrains
