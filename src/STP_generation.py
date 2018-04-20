# -*- coding: utf-8 -*-
import stocmod as stoc
import quantities as pq
import time
import numpy as np
import neo
import random

t0 = time.time()


def generate_stp(occurr, xi, t_stop, delays, t_start=0 * pq.s):
    '''
    Generate a spatio-temporal-pattern (STP). One pattern consists in a
    repeated sequence of spikes with fixed inter spikes intervals (delays).
    The starting time of the repetitions of the pattern are randomly generated.
    '''
    # Generating all the first spikes of the repetitions
    s1 = np.sort(
        np.random.uniform(
            high=(t_stop - t_start - delays[-1]).magnitude, size=occurr))

    # Using matrix algebra to add all the delays
    s1_matr = (s1 * np.ones([xi - 1, occurr])).T
    delays_matr = np.ones(
        [occurr, 1]) * delays.rescale(t_stop.units).magnitude.reshape(
        [1, xi - 1])
    ss = s1_matr + delays_matr

    # Stacking the first and successive spikes
    stp = np.hstack((s1.reshape(occurr, 1), ss))

    # Transofm in to neo SpikeTrain
    stp = [
        neo.core.SpikeTrain(
            t * t_stop.units + t_start, t_stop, t_start=t_start) for t in
        stp.T]
    return stp


def generate_stp_data(n_neurons, rate, occurr, xi, t_stop, delay):
    """
    Generate independent data with embedded STPs. The underlying process is an homogeneous stationary
    multi-dimensional Poisson process.

    `occurr` patterns with size `xi` are merged into the first `xi` spiketrains.

    Parameters
    ----------
    n_neurons: int
        Number of neurons
    rate: pq.Quantitiy
        Firing rate
    occurr: int
        Pattern occurence in the data
    xi: int
        Size of pattern
    t_stop: pq.Quantitiy
       Stop time of the spiketrain
    delay: float
      Delay between the patterns, 0 is synchrony

    Examples
    --------
    >>> # generation of data
    >>> stp_data = generate_stp_data(
    >>> n_neurons=64, rate=10 * pq.Hz, occurr=5, xi=3, t_stop=5 * pq.s, delay=0 * pq.ms)

    """
    rate_patt = t_stop.simplified.magnitude / float(occurr) * pq.Hz
    rates = [rate - rate_patt] * xi + [rate] * (n_neurons - xi)
    for i in range(n_neurons):
        # Generate the independent background of sts
        # sts_rep = {'data': [], 'patterns': []}
        # np.random.seed(i + xi + occurr)
        sts = [stocmod.homogeneous_poisson_process(
            rate=r, t_stop=t_stop, t_start=0 * pq.s) for r in rates]
        # Iterating different complexities of the patterns
        # Generating the stp
        if delay.magnitude > 0:
            stp = generate_stp(occurr=occurr,
                               xi=xi,
                               t_stop=t_stop,
                               delays=np.arange(delay.magnitude, delay.magnitude * (xi),
                                                delay.magnitude) * delay.units)
        if delay.magnitude == 0:
            stp = generate_stp(occurr=occurr,
                               xi=xi,
                               t_stop=t_stop,
                               delays=np.zeros(xi-1) * delay.units)
        # Merging the stp in the first xi sts
        sts_pool = [0] * xi
        for st_id, st in enumerate(stp):
            # st.annotate(xi=xi, occ=occurr, t_stop=t_stop, rate=rate, delay=delay, n_dataset=i)
            sts_pool[st_id] = stocmod._pool_two_spiketrains(st, sts[st_id])

        # Storing datasets containing stps
        sts_rep = {'data': sts_pool + sts[xi:], 'patterns': stp}
    return sts_rep


def generate_sts(data_type, N=100, T=1000 * pq.ms, sampl_period=10 * pq.ms):
    """
    Generate a list of parallel spike trains with different statistics.

    The data are composed of background spiking activity plus possibly
    a repeated sequence of synchronous events (SSE).
    The background activity depends on the value of data_type.
    The size and occurrence count of the SSE is specified by sse_params.

    Parameters
    ----------
    data_type : int
        An integer specifying the type of background activity.
        At the moment the following types of background activity are
        supported (note: homog = across neurons; stat = over time):
        0 | 3 : 100 indep Poisson | Gamma(5), homog, stat (15 Hz)
        7 indep Poisson, homog, stat (20 Hz)
        8 indep Poisson, homog, stat (25 Hz)
        1 | 4 : 100 indep Poisson | Gamma(5), homog, nonstat-step (10/60/10 Hz)
        2 | 5 : 100 indep Poisson | Gamma(5), heterog (5->25 Hz), stat
        6 : 100 indep Poisson, rate increase with latency variability
        9 : 100 indep Poisson, heterog, nonstat: 14/100/14 Hz+0.2*i Hz, i=1->100

    N: int (optional; works only for data_type = 0 !!!)
        total number of neurons in the model. The default is N=100.
    T: quantity.Quantity
        Simulation time. Default is 1000 * pq.ms
    sampl_period: quantity.Quantity
        Sampling period of the rate profile. Default is 10 * pq.ms

    Returns
    ------
    sts : list of SpikeTrains
        a list of spike trains
    params : dict
        a dictionary of simulation parameters (to be enriched...)

    """
    params = {'nr_neurons': N, 'simul_time': T}

    # Indep Poisson / Gamma(5), homog, stat (15 Hz)
    if data_type == 0 or data_type == 3:
        rate = 15 * pq.Hz
        shape = 1 if data_type == 0 else 5
        # Define a rate profile
        sts = stoc.gamma_thinning(
            rate=rate, shape=shape, t_stop=T, N=N)
    # Indep Poisson, homog, stat (20 Hz)
    elif data_type == 7:
        rate = 20 * pq.Hz
        shape = 1
        # Define a rate profile
        sts = stoc.gamma_thinning(
            rate=rate, shape=shape, t_stop=T, N=N)
    # Indep Poisson, homog, stat (25 Hz)
    elif data_type == 8:
        rate = 25 * pq.Hz
        shape = 1
        # Define a rate profile
        sts = stoc.gamma_thinning(
            rate=rate, shape=shape, t_stop=T, N=N)

    # Indep Poisson / Gamma(5), homog, nonstat-step (10/60/10 Hz)
    elif data_type == 1 or data_type == 4:
        shape = 1 if data_type == 1 else 5  # regularity parameter for Gamma
        a0, a1 = 10 * pq.Hz, 60 * pq.Hz  # baseline and transient rates
        t1, t2 = 600 * pq.ms, 700 * pq.ms  # time segment of transient rate

        # Define a rate profile
        times = sampl_period.units * np.arange(
            0, T.rescale(sampl_period.units).magnitude, sampl_period.magnitude)

        rate_profile = np.zeros(times.shape)
        rate_profile[np.any([times < t1, times > t2], axis=0)] = a0.magnitude
        rate_profile[np.all([times >= t1, times <= t2], axis=0)] = a1.magnitude
        rate_profile = rate_profile * a0.units
        rate_profile = neo.AnalogSignal(
            rate_profile, sampling_period=sampl_period)
        sts = stoc.gamma_nonstat_rate(
            rate_profile, shape=shape, N=N)

    # Indep Poisson / Gamma(5), heterog (5->15 Hz), stat
    elif data_type == 2 or data_type == 5:
        rate_min = 5 * pq.Hz  # min rate. Ensures that there is >=1 spike
        rate_max = 25 * pq.Hz  # max rate
        rate = np.linspace(rate_min.magnitude, rate_max.magnitude, N) * pq.Hz
        shape = 1 if data_type == 2 else 5  # regularity parameter for Gamma

        # Define a rate profile
        sts = stoc.gamma_thinning(rate=rate, shape=shape, t_stop=T, N=N)
        random.shuffle(sts)
    # Indep Poisson, latency variability
    elif data_type == 6:

        l = 20  # 20 groups of neurons
        w = 5  # of 5 neurons each
        t0 = 50 * pq.ms  # the first of which increases the rate at time t0
        t00 = 500 * pq.ms  # and again at time t00
        ratechange_dur = 5 * pq.ms  # old: 10ms  # for a short period
        a0, a1 = 14 * pq.Hz, 100 * pq.Hz  # old: 10/60 Hz; from rate a0 to a1
        ratechange_delay = 5 * pq.ms  # old: 10ms; followed with delay by next group

        # Define a rate profile
        times = sampl_period.units * np.arange(
            0, T.rescale(sampl_period.units).magnitude, sampl_period.magnitude)
        sts = []
        rate_profiles = []
        for i in range(N):
            t1 = t0 + (i // w) * ratechange_delay
            t2 = t1 + ratechange_dur
            t11 = t00 + (i // w) * ratechange_delay
            t22 = t11 + ratechange_dur
            # print t1, t2, t11, t22
            rate_profile = np.zeros(times.shape)
            rate_profile[np.any([times < t1, times > t2], axis=0)] = \
                a0.magnitude
            rate_profile[np.all([times >= t1, times <= t2], axis=0)] = \
                a1.magnitude
            rate_profile[np.all([times >= t11, times <= t22], axis=0)] = \
                a1.magnitude
            rate_profile = rate_profile * a0.units
            rate_profile = neo.AnalogSignal(
                rate_profile, sampling_period=sampl_period)
            # print np.where(np.diff(rate_profile)>0*pq.Hz)
            sts += stoc.poisson_nonstat(rate_profile, N=1)

    # Indep Poisson, heterog, nonstat: 10/60/10 Hz + .05 * i, i=-50,...,50
    elif data_type == 9:
        # Define a rate profile
        times = sampl_period.units * np.arange(
            0, T.rescale(sampl_period.units).magnitude, sampl_period.magnitude)

        a0, a1 = 10 * pq.Hz, 60 * pq.Hz  # avg baseline and transient rates
        t1, t2 = 600 * pq.ms, 700 * pq.ms  # time segment of transient rate
        minrate = 5 * pq.Hz
        drates = np.linspace(  # dev of each train from avg baseline
            minrate - a0, a0 - minrate, N)
        rate_profile = np.zeros(times.shape)
        rate_profile[np.any([times < t1, times > t2], axis=0)] = a0.magnitude
        rate_profile[np.all([times >= t1, times <= t2], axis=0)] = a1.magnitude
        rate_profile = rate_profile * a0.units
        rate_profile = neo.AnalogSignal(
            rate_profile, sampling_period=sampl_period)  # avg rate profile
        rate_profiles = [rate_profile + dr for dr in drates]  # each profile
        sts = [stoc.poisson_nonstat_thinning(rate_profiles[i], N=1,
                                             cont_sign_method='step')[0] for i
               in range(N)]

    else:
        raise ValueError(
            'data_type %d not supported. Provide int from 0 to 10' % data_type)
    return sts, params
