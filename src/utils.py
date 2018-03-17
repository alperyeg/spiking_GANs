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


def encode_input(spiketrains, rows, columns, dt=1 * pq.ms, refrac=2 * pq.ms):
    # TODO more documentation
    # TODO optimize?
    """
    Encodes a matrix for given list of `spiketrains`
    :param refrac: quantity object: min refractory period
    :param spiketrains: list of spiketrains
    :param rows: Size of the rows of matrix `M`
    :param columns: Size of the columns of matrix `M`
    :param dt: quantity object: Time resolution of the step to go
    :return: Encoded matrix `M`
    """
    M = np.zeros((rows, columns))
    if not isinstance(dt, pq.Quantity):
        raise ValueError("dt must be a Quantity object")
    else:
        dt = dt.rescale(spiketrains[0].units)
    refrac = refrac.rescale(dt.units)
    dt = float(dt.magnitude)
    refrac = float(refrac.magnitude)
    t_start = float(spiketrains[0].t_start.magnitude)
    for i, spk in enumerate(spiketrains):
        s = 0
        spk = spk.magnitude
        # intermediate result
        res = 0
        steps = dt
        for j in range(columns):
            if s < len(spk):
                if j == 0:
                    if t_start + dt >= spk[s]:
                        res = spk[s]
                        s += 1
                    else:
                        res = t_start
                    M[i, j] = res
                    # s += 1
                    steps += dt
                else:
                    # copy if smaller
                    if res + steps < spk[s]:
                        M[i, j] = res
                        steps += dt
                    else:
                        # check for refractory period violation
                        if spk[s] - res >= refrac:
                            M[i, j] = spk[s]
                            res = spk[s]
                            steps = dt
                        else:
                            M[i, j] = res
                            steps += dt
                        s += 1
            else:
                M[i, j] = res
    return M


def encoder(spiketrains, cols, dt):
    # TODO type of sliding window, atm not sliding, rather jumping
    # TODO add possibility to calculate dt: cols/max_spike
    """
    Encodes the input for a given set of spiketrains, via sliding window
    :param spiketrains: neo.SpikeTrain objects
    :param cols: number of columns for the matrix 
    :param dt: time resolution
    :return: ms: list of encoded matrices, shape: `[windows, len(spiketrains),
    cols]`, where `windows` is the number of windows to cover all spikes with
    for `cols` size
    """
    # get row of matrix
    rows = len(spiketrains)
    # max_spike = max(max(spks, key=max))
    # get index of longest spiketrain
    m = spiketrains.index(max(spiketrains, key=len))
    longest_spiketrain = len(spiketrains[m])
    # how many windows in total
    windows = int(
        np.ceil(
            longest_spiketrain / (cols * dt.rescale(spiketrains[0].units))))
    en = encode_input(spiketrains, rows, windows * cols, dt)
    # store all encoded inputs
    ms = []
    [ms.append(en[:, w * cols:cols * (w + 1)]) for w in range(windows)]
    return ms
