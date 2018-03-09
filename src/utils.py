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


def encode_input(spiketrains, rows, columns, dt):
    # TODO more documentation, optimize?
    """
    Encodes a matrix for given list of `spiketrains`
    :param spiketrains: list of spiketrains
    :param rows: Size of the rows of matrix `M`
    :param columns: Size of the columns of matrix `M`
    :param dt: Time resolution
    :return: Matrix `M` and the last spike index for each spiketrains
    """
    M = np.zeros((rows, columns))
    spike_index = []
    for i, spk in enumerate(spiketrains):
        s = 0
        # intermediate result
        res = 0
        steps = 0
        for j in range(columns):
            if s < len(spk):
                if j == 0:
                    res = spk[s]
                    M[i, j] = res
                    s += 1
                    steps = dt
                else:
                    # copy if smaller
                    if res + steps < spk[s]:
                        M[i, j] = res
                        steps += dt
                    else:
                        # check for refractory period violation
                        if spk[s] - res >= dt:
                            M[i, j] = spk[s]
                            res = spk[s]
                            steps = dt
                        else:
                            M[i, j] = res
                            steps += dt
                        s += 1
            else:
                M[i, j] = res
        spike_index.append(s)
    return M, spike_index
