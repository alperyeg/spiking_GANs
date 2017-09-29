import numpy as np
import random
import warnings
import quantities as pq
import neo
# import jelephant.core.neo_tools as neo_tools
# import jelephant.core.surrogates as surr
import scipy


def shift_spiketrain(spiketrain, t):
    '''
    Shift the times of a SpikeTrain by an amount t.

    Shifts also the SpikeTrain's attributes t_start and t_stop by t.
    Retains the SpikeTrain's waveforms, sampling_period, annotations.

    Paramters
    ---------
    spiketrain : SpikeTrain
        the spike train to be shifted
    t : Quantity
        the amount by which to shift the spike train

    Returns
    -------
    SpikeTrain : SpikeTrain
       a new SpikeTrain, whose times and attributes t_start, t_stop are those
       of the input spiketrain  shifted by an amount t. Waveforms, sampling
       period and annotations are also retained.
    '''
    st = spiketrain
    st_shifted = neo.SpikeTrain(
        st.view(pq.Quantity) + t, t_start=st.t_start + t,
        t_stop=st.t_stop + t, waveforms=st.waveforms)
    st_shifted.sampling_period = st.sampling_period
    st_shifted.annotations = st.annotations

    return st_shifted


def poisson(rate, t_stop, t_start=0 * pq.s, n=None, decimals=None):
    """
    Generates one or more independent Poisson spike trains.

    Parameters
    ----------
    rate : Quantity or Quantity array
        Expected firing rate (frequency) of each output SpikeTrain.
        Can be one of:
        *  a single Quantity value: expected firing rate of each output
           SpikeTrain
        *  a Quantity array: rate[i] is the expected firing rate of the i-th
           output SpikeTrain
    t_stop : Quantity
        Single common stop time of each output SpikeTrain. Must be > t_start.
    t_start : Quantity (optional)
        Single common start time of each output SpikeTrain. Must be < t_stop.
        Default: 0 s.
    decimals : int or None (optional)
        Precision, i.e., number of decimal places, for the spikes in the
        SpikeTrains. To create spike times as whole numbers, i.e., no decimal
        digits, use decimals = 0. If set to None, no rounding takes place and
        default computer precision will be used.
        Default: None
    n : int or None (optional)
        If rate is a single Quantity value, n specifies the number of
        SpikeTrains to be generated. If rate is an array, n is ignored and the
        number of SpikeTrains is equal to len(rate).
        Default: None


    Returns
    -------
    list of neo.SpikeTrain
        Each SpikeTrain contains one of the independent Poisson spike trains,
        either n SpikeTrains of the same rate, or len(rate) SpikeTrains with
        varying rates according to the rate parameter. The time unit of the
        SpikeTrains is given by t_stop.


    Example
    -------

    >>> import numpy as np
    >>> import quantities as pq
    >>> np.random.seed(1)
    >>> print stocmod.poisson(rate = 3*pq.Hz, t_stop=1*pq.s)

    [<SpikeTrain(array([ 0.14675589,  0.30233257]) * s, [0.0 s, 1.0 s])>]

    >>> print stocmod.poisson(rate = 3*pq.Hz, t_stop=1*pq.s, decimals=2)

    [<SpikeTrain(array([ 0.35]) * s, [0.0 s, 1.0 s])>]

    >>> print stocmod.poisson(rate = 3*pq.Hz, t_stop=1*pq.s, decimals=2, n=2)

    [<SpikeTrain(array([ 0.14,  0.42,  0.56,  0.67]) * s, [0.0 s, 1.0 s])>,
     <SpikeTrain(array([ 0.2]) * s, [0.0 s, 1.0 s])>]

    >>> # Return the spike counts of 3 generated spike trains
    >>> print [len(x) for x in stocmod.poisson(
            rate = [20,50,80]*pq.Hz, t_stop=1*pq.s)]

    [17, 38, 66]
    """
    # Check that the provided input is Hertz of return error
    try:
        for r in rate.reshape(-1, 1):
            r.rescale('Hz')
    except ValueError:
        raise ValueError('rate argument must have rate unit (1/time)')

    # Check t_start < t_stop and create their strip dimensions
    if not t_start < t_stop:
        raise ValueError(
            't_start (=%s) must be < t_stop (=%s)' % (t_start, t_stop))
    stop_dl = t_stop.simplified.magnitude
    start_dl = t_start.simplified.magnitude

    # Set number N of output spike trains (specified or set to len(rate))
    if n is not None:
        if not (type(n) == int and n > 0):
            raise ValueError('n (=%s) must be a positive integer' % str(n))
        N = n
    else:
        if rate.ndim == 0:
            N = 1
        else:
            N = len(rate.flatten())
            if N == 0:
                raise ValueError('No rate specified.')

    rate_dl = rate.simplified.magnitude.flatten()

    # Check rate input parameter
    if len(rate_dl) == 1:
        if rate_dl < 0:
            raise ValueError('rate (=%s) must be non-negative.' % rate)
        rates = np.array([rate_dl] * N)
    else:
        rates = rate_dl.flatten()
        if any(rates < 0):
            raise ValueError('rate must have non-negative elements.')

    if N != len(rates):
        warnings.warn('rate given as Quantity array, n will be ignored.')

    # TODO: ref period does not work!    
    # Generate the array of (random, Poisson) number of spikes per spiketrain
    num_spikes = np.random.poisson(rates * (stop_dl - start_dl))
    if isinstance(num_spikes, int):
        num_spikes = np.array([num_spikes])
    spiketimes = [
        (stop_dl - start_dl) * np.sort(np.random.random(ns)) +
            start_dl for ns in num_spikes]
    #    spiketimes = spiketimes + (np.array([0]+list(np.diff(spiketimes) > 0.001)))
    for st_idx, st in enumerate(spiketimes):
        st = np.array(st)
        if len(st>0):
            ref_per = np.random.rand() / 1000. + 0.001
            spiketimes[st_idx] = np.hstack(
                (st[0], st[np.where(np.diff(st) > ref_per)[0]+1]))

    # Create the Poisson spike trains
    series = [neo.SpikeTrain(
        spk*t_stop.units, t_start=t_start, t_stop=t_stop)  for spk in spiketimes]

    # Round to decimal position, if requested
    if decimals is not None:
        series = [neo.SpikeTrain(
            s.round(decimals=decimals), t_start=t_start, t_stop=t_stop)
            for s in series]

    return series


def poisson_nonstat(rate_signal, N=1, method='time_rescale'):
    '''
    Generates an ensemble of non-stationary Poisson processes with identical
    intensity.

    Parameters
    ----------
    rate_signal : neo.AnalogSignal or list
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate or the list of
        the different signal for each neuron
    N : int
        ensemble sizen number of spike trains n output, in case rate_signa is
        a list of different signal, N spike trains for each different rate
        profiles are generated
        Default: N=1
    method : string
        The method used to generate the non-stationary poisson process:
        *'time_rescale': method based on the time rescaling theorem
        (ref. Brwn et al. 2001)
        *'thinning': thinning method of a stationary poisson process
        (ref. Sigman Notes 2013)
        Default:'time_rescale'
    -------
    spiketrains : list(list(float))
        list of spike trains
    '''
    methods_dic = {
        'time_rescale': poisson_nonstat_time_rescale,
        'thinning': poisson_nonstat_thinning}
    if method not in methods_dic:
        raise ValueError("Unknown method selected.")
    method_use = methods_dic[method]
    if type(rate_signal) == neo.core.analogsignal.AnalogSignal:
        if N is None:
                sts = method_use(rate_signal)
        else:
            sts = method_use(rate_signal, N=N)
    else:
        sts = []
        for r in rate_signal:
            sts = sts + method_use(r, N=N)
    return sts


def sip_poisson(
        M, N, T, rate_b, rate_c, jitter=0 * pq.s, tot_coinc='det',
        start=0 * pq.s, min_delay=0 * pq.s, decimals=4,
        return_coinc=False, output_format='list'):
    """
    Generates a multidimensional Poisson SIP (single interaction process)
    plus independent Poisson processes

    A Poisson SIP consists of Poisson time series which are independent
    except for simultaneous events in all of them. This routine generates
    a SIP plus additional parallel independent Poisson processes.

    **Args**:
      M [int]
          number of Poisson processes with coincident events to be generated.
          These will be the first M processes returned in the output.
      N [int]
          number of independent Poisson processes to be generated.
          These will be the last N processes returned in the output.
      T [float. Quantity assignable, default to sec]
          total time of the simulated processes. The events are drawn between
          0 and T. A time unit from the 'quantities' package can be assigned
          to T (recommended)
      rate_b [float | iterable. Quantity assignable, default to Hz]
          overall mean rate of the time series to be generated (coincidence
          rate rate_c is subtracted to determine the background rate). Can be:
          * a float, representing the overall mean rate of each process. If
            so, it must be higher than rate_c.
          * an iterable of floats (one float per process), each float
            representing the overall mean rate of a process. If so, the first
            M entries must be larger than rate_c.
      rate_c [float. Quantity assignable, default to Hz]
          coincidence rate (rate of coincidences for the M-dimensional SIP).
          The SIP time series will have coincident events with rate rate_c
          plus independent 'background' events with rate rate_b-rate_c.
      jitter [float. Quantity assignable, default to sec]
          jitter for the coincident events. If jitter == 0, the events of all
          M correlated processes are exactly coincident. Otherwise, they are
          jittered around a common time randomly, up to +/- jitter.
      tot_coinc [string. Default to 'det']
          whether the total number of injected coincidences must be determin-
          istic (i.e. rate_c is the actual rate with which coincidences are
          generated) or stochastic (i.e. rate_c is the mean rate of coincid-
          ences):
          * 'det', 'd', or 'deterministic': deterministic rate
          * 'stoc', 's' or 'stochastic': stochastic rate
      start [float <T. Default to 0. Quantity assignable, default to sec]
          starting time of the series. If specified, it must be lower than T
      min_delay [float <T. Default to 0. Quantity assignable, default to sec]
          minimum delay between consecutive coincidence times.
      decimals [int| None. Default to 4]
          number of decimal points for the events in the time series. E.g.:
          decimals = 0 generates time series with integer elements,
          decimals = 4 generates time series with 4 decimals per element.
          If set to None, no rounding takes place and default computer
          precision will be used
      return_coinc [bool]
          whether to retutrn the coincidence times for the SIP process
      output_format [str. Default: 'list']
          the output_format used for the output data:
          * 'gdf' : the output is a np ndarray having shape (2,-1). The
                    first column contains the process ids, the second column
                    represents the corresponding event times.
          * 'list': the output is a list of M+N sublists. Each sublist repres-
                    ents one process among those generated. The first M lists
                    contain the injected coincidences, the last N ones are
                    independent Poisson processes.
          * 'dict': the output is a dictionary whose keys are process IDs and
                    whose values are np arrays representing process events.

    **OUTPUT**:
      realization of a SIP consisting of M Poisson processes characterized by
      synchronous events (with the given jitter), plus N independent Poisson
      time series. The output output_format can be either 'gdf', list or
      dictionary (see output_format argument). In the last two cases a time
      unit is assigned to the output times (same as T's. Default to sec).

      If return_coinc == True, the coincidence times are returned as a second
      output argument. They also have an associated time unit (same as T's.
      Default to sec).

    .. note::
        See also: poisson(), msip_poisson(), genproc_mip_poisson(),
                  genproc_mmip_poisson()

    *************************************************************************
    EXAMPLE:

    >>> import quantities as qt
    >>> import jelephant.core.stocmod as sm
    >>> sip, coinc = sm.sip_poisson(M=10, N=0, T=1*qt.sec, \
            rate_b=20*qt.Hz,  rate_c=4, return_coinc = True)

    *************************************************************************
    """

    # return empty objects if N=M=0:
    if N == 0 and M == 0:
        if output_format == 'list':
            return [] * T.units
        elif output_format == 'gdf':
            return np.array([[]] * 2).T
        elif output_format == 'dict':
            return {}

    # Assign time unit to jitter, or check that its existing unit is a time
    # unit
    jitter = abs(jitter)

    # Define the array of rates from input argument rate. Check that its length
    # matches with N
    if rate_b.ndim == 0:
        if rate_b < 0:
            raise ValueError(
                'rate_b (=%s) must be non-negative.' % str(rate_b))
        rates_b = np.array(
            [rate_b.magnitude for _ in xrange(N + M)]) * rate_b.units
    else:
        rates_b = np.array(rate_b).flatten() * rate_b.units
        if not all(rates_b >= 0):
            raise ValueError('*rate_b* must have non-negative elements')
        elif N + M != len(rates_b):
            raise ValueError(
                "*N* != len(*rate_b*). Either specify rate_b as"
                "a vector, or set the number of spike trains by *N*")

    # Check: rate_b>rate_c
    if np.any(rates_b < rate_c):
        raise ValueError('all elements of *rate_b* must be >= *rate_c*')

    # Check min_delay < 1./rate_c
    if not (rate_c == 0 or min_delay < 1. / rate_c):
        raise ValueError(
            "'*min_delay* (%s) must be lower than 1/*rate_c* (%s)." %
            (str(min_delay), str((1. / rate_c).rescale(min_delay.units))))

    # Check that the variable decimals is integer or None.
    if decimals is not None and type(decimals) != int:
        raise ValueError(
            'decimals type must be int or None. %s specified instead' %
            str(type(decimals)))

    # Generate the N independent Poisson processes
    if N == 0:
        independ_poisson_trains = [] * T.units
    else:
        independ_poisson_trains = poisson(
            rate=rates_b[M:], t_stop=T, t_start=start, decimals=decimals)
        # Convert the trains from neo SpikeTrain objects to  simpler Quantity
        # objects
        independ_poisson_trains = [
            pq.Quantity(ind.base) * ind.units
            for ind in independ_poisson_trains]

    # Generate the M Poisson processes there are the basis for the SIP
    # (coincidences still lacking)
    if M == 0:
        embedded_poisson_trains = [] * T.units
    else:
        embedded_poisson_trains = poisson(
            rate=rates_b[:M] - rate_c, t_stop=T, t_start=start, n=M,
            decimals=decimals)
        # Convert the trains from neo SpikeTrain objects to simpler Quantity
        # objects
        embedded_poisson_trains = [
            pq.Quantity(emb.base) * emb.units
            for emb in embedded_poisson_trains]

    # Generate the array of times for coincident events in SIP, not closer than
    # min_delay. The array is generated as a quantity from the Quantity class
    # in the quantities module
    if tot_coinc in ['det', 'd', 'deterministic']:
        Nr_coinc = int(((T - start) * rate_c).rescale(pq.dimensionless))
        while 1:
            coinc_times = start + \
                np.sort(np.random.random(Nr_coinc)) * (T - start)
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
    elif tot_coinc in ['s', 'stoc', 'stochastic']:
        while 1:
            coinc_times = poisson(rate=rate_c, t_stop=T, t_start=start, n=1)[0]
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
        # Convert coinc_times from a neo SpikeTrain object to a Quantity object
        # pq.Quantity(coinc_times.base)*coinc_times.units
        coinc_times = coinc_times.view(pq.Quantity)
        # Set the coincidence times to T-jitter if larger. This ensures that
        # the last jittered spike time is <T
        for i in range(len(coinc_times)):
            if coinc_times[i] > T - jitter:
                coinc_times[i] = T - jitter

    # Replicate coinc_times M times, and jitter each event in each array by
    # +/- jitter (within (start, T))
    embedded_coinc = coinc_times + \
        np.random.random((M, len(coinc_times))) * 2 * jitter - jitter
    embedded_coinc = embedded_coinc + \
        (start - embedded_coinc) * (embedded_coinc < start) - \
        (T - embedded_coinc) * (embedded_coinc > T)

    # Inject coincident events into the M SIP processes generated above, and
    # merge with the N independent processes
    sip_process = [
        np.sort(np.concatenate((
            embedded_poisson_trains[m].rescale(T.units),
            embedded_coinc[m].rescale(T.units))) * T.units)
        for m in xrange(M)]

    # Append the independent spike train to the list of trains
    sip_process.extend(independ_poisson_trains)

    # Convert back sip_process and coinc_times from Quantity objects to
    # neo.SpikeTrain objects
    sip_process = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in sip_process]
    coinc_times = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in embedded_coinc]

    # Return the processes in the specified output_format
    if output_format == 'list':
        if not return_coinc:
            output = sip_process  # [np.sort(s) for s in sip_process]
        else:
            output = sip_process, coinc_times
    elif output_format == 'gdf':
        neuron_ids = np.concatenate(
            [np.ones(len(s)) * (i + 1) for i, s in enumerate(sip_process)])
        spike_times = np.concatenate(sip_process)
        ids_sortedtimes = np.argsort(spike_times)
        output = np.array(
            (neuron_ids[ids_sortedtimes], spike_times[ids_sortedtimes])).T
        if return_coinc:
            output = output, coinc_times
    elif output_format == 'dict':
        dict_sip = {}
        for i, s in enumerate(sip_process):
            dict_sip[i + 1] = s
        if not return_coinc:
            output = dict_sip
        else:
            output = dict_sip, coinc_times

    return output


def sip_nonstat(
        M, N, rate_b, rate_c, jitter=0 * pq.s, tot_coinc='det',
        start=0 * pq.s, min_delay=0 * pq.s,
        return_coinc=False, output_format='list'):
    """
    Generates a multidimensional Poisson SIP (single interaction process)
    plus independent Poisson processes, all with non stationary firing rate
    profiles.

    A Poisson SIP consists of Poisson time series which are independent
    except for simultaneous events in all of them. This routine generates
    a SIP plus additional parallel independent Poisson processes.

    **Args**:
      M : int
          number of Poisson processes with coincident events to be generated.
          These will be the first M processes returned in the output.
      N : int
          number of independent Poisson processes to be generated.
          These will be the last N processes returned in the output.
      rate_b : neo.AnalogSignal or list
          overall mean rate of the time series to be generated (coincidence
          rate rate_c is subtracted to determine the background rate). Can be:
          * an AnalogSignal, representing the overall mean rate profile of each
          process. If so, it must be higher than rate_c in each time point.
          * an iterable of AnalogSignals (one signal per process), each
            representing the overall mean rate profile of a process. If so,
            the first M entries must be larger than rate_c in each time point.
      rate_c : Quantity (1/time)
          coincidence rate (rate of coincidences for the M-dimensional SIP).
          The SIP time series will have coincident events with rate rate_c
          plus independent 'background' events with rate rate_b-rate_c.
      jitter : Quantity (time)
          jitter for the coincident events. If jitter == 0, the events of all
          M correlated processes are exactly coincident. Otherwise, they are
          jittered around a common time randomly, up to +/- jitter.
      tot_coinc : string
          whether the total number of injected coincidences must be determin-
          istic (i.e. rate_c is the actual rate with which coincidences are
          generated) or stochastic (i.e. rate_c is the mean rate of coincid-
          ences):
          * 'det', 'd', or 'deterministic': deterministic rate
          * 'stoc', 's' or 'stochastic': stochastic rate
          Default: 'det'
      min_delay Quantity (time)
          minimum delay between consecutive coincidence times.
      return_coinc [bool]
          whether to retutrn the coincidence times for the SIP process
          Default=False
      output_format string
          the output_format used for the output data:
          * 'gdf' : the output is a np ndarray having shape (2,-1). The
                    first column contains the process ids, the second column
                    represents the corresponding event times.
          * 'list': the output is a list of M+N sublists. Each sublist repres-
                    ents one process among those generated. The first M lists
                    contain the injected coincidences, the last N ones are
                    independent Poisson processes.
          * 'dict': the output is a dictionary whose keys are process IDs and
                    whose values are np arrays representing process events.
          Default: 'list'
    **OUTPUT**:
      realization of a SIP consisting of M Poisson processes characterized by
      synchronous events (with the given jitter), plus N independent Poisson
      time series. The output output_format can be either 'gdf', list or
      dictionary (see output_format argument). In the last two cases a time
      unit is assigned to the output times (same as T's. Default to sec).

      If return_coinc == True, the coincidence times are returned as a second
      output argument. They also have an associated time unit (same as T's.
      Default to sec).

    .. note::
        See also: poisson(), sip_poisson(), msip_poisson(), sip_nonstat
        mip_gen(), genproc_mmip_poisson()

    *************************************************************************
    EXAMPLE:

    >>> import quantities as qt
    >>> import jelephant.core.stocmod as sm
    >>> sip, coinc = sm.sip_poisson(M=10, N=0, T=1*qt.sec, \
            rate_b=20*qt.Hz,  rate_c=4, return_coinc = True)

    *************************************************************************
    """

    # return empty objects if N=M=0:
    if N == 0 and M == 0:
        if output_format == 'list':
            return [] * pq.s
        elif output_format == 'gdf':
            return np.array([[]] * 2).T
        elif output_format == 'dict':
            return {}

    # Assign time unit to jitter, or check that its existing unit is a time
    # unit
    jitter = abs(jitter)

    # Define the array of rates from input argument rate. Check that its length
    # matches with N
    if type(rate_b) == neo.core.analogsignal.AnalogSignal:
        rate_b = [rate_b for i in range(N + M)]
    elif N + M != len(rate_b):
        raise ValueError(
            "*N* != len(*rate_b*). Either specify rate_b as"
            "a vector, or set the number of spike trains by *N*")
    # Check: rate_b>rate_c
    if np.min([np.min(r) for r in rate_b]) < rate_c.rescale(
        rate_b[0].units).magnitude:
        raise ValueError('all elements of *rate_b* must be >= *rate_c*')

    # Check min_delay < 1./rate_c
    if not (rate_c == 0 or min_delay < 1. / rate_c):
        raise ValueError(
            "'*min_delay* (%s) must be lower than 1/*rate_c* (%s)." %
            (str(min_delay), str((1. / rate_c).rescale(min_delay.units))))

    # Check that the variable decimals is integer or None.

    T = rate_b[0].t_stop
    start = rate_b[0].t_start
    # Generate the N independent Poisson processes
    if N == 0:
        independ_poisson_trains = [] * T.units
    else:
        independ_poisson_trains = poisson_nonstat(
            rate_signal=rate_b[M:])
        # Convert the trains from neo SpikeTrain objects to  simpler Quantity
        # objects
        independ_poisson_trains = [
            pq.Quantity(ind.base) * ind.units
            for ind in independ_poisson_trains]

    # Generate the M Poisson processes there are the basis for the SIP
    # (coincidences still lacking)
    if M == 0:
        embedded_poisson_trains = [] * T.units
    else:
        embedded_poisson_trains = poisson_nonstat(
            rate_signal=[r - rate_c for r in rate_b[:M]])
        # Convert the trains from neo SpikeTrain objects to simpler Quantity
        # objects
        embedded_poisson_trains = [
            pq.Quantity(emb.base) * emb.units
            for emb in embedded_poisson_trains]

    # Generate the array of times for coincident events in SIP, not closer than
    # min_delay. The array is generated as a quantity from the Quantity class
    # in the quantities module
    if tot_coinc in ['det', 'd', 'deterministic']:
        Nr_coinc = int(((T - start) * rate_c).rescale(pq.dimensionless))
        while 1:
            coinc_times = start + \
                np.sort(np.random.random(Nr_coinc)) * (T - start)
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
    elif tot_coinc in ['s', 'stoc', 'stochastic']:
        while 1:
            coinc_times = poisson(
                rate=rate_c, t_stop=T, t_start=start, n=1)[0]
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
        # Convert coinc_times from a neo SpikeTrain object to a Quantity object
        # pq.Quantity(coinc_times.base)*coinc_times.units
        coinc_times = coinc_times.view(pq.Quantity)
        # Set the coincidence times to T-jitter if larger. This ensures that
        # the last jittered spike time is <T
        for i in range(len(coinc_times)):
            if coinc_times[i] > T - jitter:
                coinc_times[i] = T - jitter

    # Replicate coinc_times M times, and jitter each event in each array by
    # +/- jitter (within (start, T))
    embedded_coinc = coinc_times + \
        np.random.random((M, len(coinc_times))) * 2 * jitter - jitter
    embedded_coinc = embedded_coinc + \
        (start - embedded_coinc) * (embedded_coinc < start) - \
        (T - embedded_coinc) * (embedded_coinc > T)

    # Inject coincident events into the M SIP processes generated above, and
    # merge with the N independent processes
    sip_process = [
        np.sort(np.concatenate((
            embedded_poisson_trains[m].rescale(T.units),
            embedded_coinc[m].rescale(T.units))) * T.units)
        for m in xrange(M)]

    # Append the independent spike train to the list of trains
    sip_process.extend(independ_poisson_trains)

    # Convert back sip_process and coinc_times from Quantity objects to
    # neo.SpikeTrain objects
    sip_process = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in sip_process]
    coinc_times = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in embedded_coinc]

    # Return the processes in the specified output_format
    if output_format == 'list':
        if not return_coinc:
            output = sip_process  # [np.sort(s) for s in sip_process]
        else:
            output = sip_process, coinc_times
    elif output_format == 'gdf':
        neuron_ids = np.concatenate(
            [np.ones(len(s)) * (i + 1) for i, s in enumerate(sip_process)])
        spike_times = np.concatenate(sip_process)
        ids_sortedtimes = np.argsort(spike_times)
        output = np.array(
            (neuron_ids[ids_sortedtimes], spike_times[ids_sortedtimes])).T
        if return_coinc:
            output = output, coinc_times
    elif output_format == 'dict':
        dict_sip = {}
        for i, s in enumerate(sip_process):
            dict_sip[i + 1] = s
        if not return_coinc:
            output = dict_sip
        else:
            output = dict_sip, coinc_times

    return output


def msip_poisson(
        M, N, T, rate_b, rate_c, jitter=0 * pq.s, tot_coinc='det',
        start=0 * pq.s, min_delay=0 * pq.s, decimals=4, return_coinc=False,
        output_format='list'):
    """
    Generates Poisson multiple single-interaction-processes (mSIP) plus
    independent Poisson processes.

    A Poisson SIP consists of Poisson time series which are independent
    except for events occurring simultaneously in all of them. This routine
    generates multiple, possibly overlapping SIP plus additional parallel
    independent Poisson processes.

    **Args**:
      M [iterable | iterable of iterables]
          The list of neuron tags composing SIPs that have to be generated.
          Can be:
          * an iterable of integers: each integer is a time series ID, the
            list represents one SIP. A single SIP is generated this way.
          * an iterable of iterables: each internal iterable must contain
            integers, and represents a SIP that has to be generated.
            Different SIPs can be overlapping
      N [int | iterable]
          Refers to the full list of time series to be generated. Can be:
          * an integer, representing the number of Poisson processes to be
            generated. If so, the time series IDs will be integers from 1 to
            N.
          * an iterable, representing the full list of time series IDs to be
            generated.
      T [float. Quantity assignable, default to sec]
          total time of the simulated processes. The events are drawn between
          0 and T. A time unit from the 'quantities' package can be assigned
          to T (recommended)
      rate_b [float | iterable. Quantity assignable, default to Hz]
          overall mean rate of the time series to be generated (coincidence
          rate rate_c is subtracted to determine the background rate). Can be:
          * a float, representing the overall mean rate of each process. If
            so, it must be higher than each entry in rate_c.
          * an iterable of floats (one float per process), each float
            representing the overall mean rate of a process. For time series
            embedded in a SIP, the corresponding entry in rate_b must be
            larger than that SIP's rate (see rate_c).
      rate_c [float. Quantity assignable, default to Hz]
          coincidence rate (rate of coincidences for the M-dimensional SIP).
          Each SIP time series will have coincident events with rate rate_c,
          plus independent background events with rate rate_b-rate_c.
      jitter [float. Quantity assignable, default to sec]
          jitter for the coincident events. If jitter == 0, the events of all
          M correlated processes are exactly coincident. Otherwise, they are
          jittered around a common time randomly, up to +/- jitter.
      tot_coinc [string. Default to 'det']
          whether the total number of injected coincidences must be determin-
          istic (i.e. rate_c is the actual rate with which coincidences are
          generated) or stochastic (i.e. rate_c is the mean rate of coincid-
          ences):
          * 'det', 'd', or 'deterministic': deterministic rate
          * 'stoc', 's' or 'stochastic': stochastic rate
      start [float <T. Default to 0. Quantity assignable, default to sec]
          starting time of the series. If specified, it must be lower than T
      min_delay [float <T. Default to 0. Quantity assignable, default to sec]
          minimum delay between consecutive coincidence times of a SIP.
          This does not affect coincidences from two different SIPs, which
          can fall arbitrarily closer to each other.
      decimals [int| None. Default to 4]
          number of decimal points for the events in the time series. E.g.:
          decimals = 0 generates time series with integer elements,
          decimals = 4 generates time series with 4 decimals per element.
          If set to None, no rounding takes place and default computer
          precision will be used
      return_coinc [bool]
          whether to retutrn the coincidence times for the SIP process
      output_format   : [string. Default to 'gdf']
          the output_format used for the output data:
          * 'gdf' : the output is a np ndarray having shape (2,-1). The
                    first column contains the process ids, the second column
                    represents the corresponding event times.
          * 'list': the output is a list of M+N sublists. Each sublist repres-
                    ents one process among those generated. The first M lists
                    contain the injected coincidences, the last N ones are
                    independent Poisson processes.
          * 'dict': the output is a dictionary whose keys are process IDs and
                    whose values are np arrays representing process events.

    **OUTPUT**:
      Realization of mSIP plus independent Poisson time series. M and N
      determine the number of SIP assemblies and overall time series,
      respectively.
      The output output_format can be either 'gdf', list or dictionary
      (see output_format argument). In the last two cases a time unit is
      assigned to the output times (same as T's. Default to sec).

      If return_coinc == True, the mSIP coincidences are returned as an
      additional output variable. They are represented a list of lists, each
      sublist containing the coincidence times of a SIP. They also have an
      associated time unit (same as T's. Default to sec).

    **See also**:
      poisson(), sip_poisson(), genproc_mip_poisson(),
      genproc_mmip_poisson()

    *************************************************************************
    EXAMPLE:

    >>> import quantities as qt
    >>> import jelephant.core.stocmod as sm
    >>>
    >>> M = [1,2,3], [4,5]
    >>> N = 6
    >>> T = 1*qt.sec
    >>> rate_b, rate_c = 5 * qt.Hz, [2,3] *qt.Hz
    >>>
    >>> msip, coinc = sm.msip_poisson(M=M, N=N, T=T, rate_b=rate_b, \
            rate_c=rate_c, return_coinc = True, output_format='list')

    *************************************************************************
    """

    # Create from M the list all_units of all unit IDs to be generated, and
    # check N
    if hasattr(N, '__iter__'):
        all_units = N
    elif type(N) == int and N > 0:
        all_units = range(1, N + 1)
    else:
        raise ValueError(
            'N (=%s) must be a positive integer or an iterable' %
            str(N))

    # Create from M the list all_sip of all SIP assemblies to be generated, and
    # check M
    if hasattr(M, '__iter__'):
        if all([hasattr(m, '__iter__') for m in M]):
            all_sip = M
        elif all([type(m) == int for m in M]):
            all_sip = [M]
        else:
            raise ValueError(
                "M must be either a list of lists (one for every SIP) or "
                "a list of integers (a single SIP)")
    else:
        raise ValueError(
            "M must be either a list of lists (one for every SIP)"
            " or a list of integers (a single SIP)")

    # Check that the list of all units includes that of all sip-embedded units
    if not all([set(all_units).issuperset(sip) for sip in all_sip]):
        raise ValueError(
            "The set of all units (defined by N) must include each SIP"
            " (defined by M)")

    # Create the array of coincidence rates (one rate per SIP). Check the
    # number of elements and their non-negativity
    if rate_c.ndim == 0:
        rates_c = np.array([rate_c.magnitude for sip in all_sip]) * \
            rate_c.units
    else:
        rates_c = np.array(rate_c).flatten() * rate_c.units
        if not all(rates_c >= 0):
            raise ValueError('variable rate_c must have non-negative elements')
        elif len(all_sip) != len(rates_c):
            raise ValueError(
                "length of rate_c (=%d) and number of SIPs (=%d) mismatch" %
                (len(rate_c), len(all_sip)))

    # Define the array of rates from input argument rate. Check that its length
    # matches with N
    if rate_b.ndim == 0:
        if rate_b < 0:
            raise ValueError(
                "rate_b (=%s) must be non-negative." %
                str(rate_b))
        rates_b = np.array([rate_b.magnitude for _ in all_units]) * \
            rate_b.units
    else:
        rates_b = np.array(rate_b).flatten() * rate_b.units
        if not all(rates_b >= 0):
            raise ValueError("variable rate_b must have non-negative elements")
        elif len(all_units) != len(rates_b):
            raise ValueError(
                "the length of rate_b (=%d) must match the number "
                "of units (%d)" % (len(rates_b), len(all_units)))

    # Compute the background firing rate (total rate - coincidence rate) and
    # simulate background activity as a list...
    rates_bg = rates_b
    for sip_idx, sip in enumerate(all_sip):
        for n_id in sip:
            rates_bg[n_id - 1] -= rates_c[sip_idx]

    # Simulate the background activity and convert from neo SpikeTrain to
    # Quantity object
    background_activity = poisson(
        rate=rates_bg, t_stop=T, t_start=start, decimals=decimals)
    background_activity = [
        pq.Quantity(bkg.base) * bkg.units for bkg in background_activity]

    # Add SIP-like activity (coincidences only!) to background activity, and
    # list for each SIP its coincidences
    sip_coinc = []
    for sip, sip_rate in zip(all_sip, rates_c):
        sip_activity, coinc_times = sip_poisson(
            M=len(sip), N=0, T=T, rate_b=sip_rate, rate_c=sip_rate,
            jitter=jitter, tot_coinc=tot_coinc, start=start,
            min_delay=min_delay, decimals=decimals,
            return_coinc=True, output_format='list')
        sip_coinc.append(coinc_times)
        for i, n_id in enumerate(sip):
            background_activity[n_id - 1] = np.sort(
                np.concatenate(
                    [background_activity[n_id - 1], sip_activity[i]]) *
                T.units)

    # Convert background_activity from a Quantity object back to a neo
    # SpikeTrain object
    background_activity = [
        neo.SpikeTrain(bkg, t_start=start, t_stop=T).rescale(T.units)
        for bkg in background_activity]

    # Return the processes in the specified output_format
    if output_format == 'list':
        if not return_coinc:
            return background_activity
        else:
            return background_activity, sip_coinc
    elif output_format == 'gdf':
        neuron_ids = np.concatenate([
            np.ones(len(s)) * (i + 1)
            for i, s in enumerate(background_activity)])
        spike_times = np.concatenate(background_activity)
        ids_sortedtimes = np.argsort(spike_times)
        if not return_coinc:
            return np.array((
                neuron_ids[ids_sortedtimes], spike_times[ids_sortedtimes])).T
        else:
            return (
                np.array((
                    neuron_ids[ids_sortedtimes],
                    spike_times[ids_sortedtimes])).T,
                sip_coinc)
    elif output_format == 'dict':
        dict_sip = {}
        for i, s in enumerate(background_activity):
            dict_sip[i + 1] = s
        if not return_coinc:
            return dict_sip
        else:
            return dict_sip, sip_coinc


def mip_nonstat(
        M, N, rate_b, rate_c, pi, jitter=0 * pq.s, tot_coinc='det',
        start=0 * pq.s, min_delay=0 * pq.s,
        return_coinc=False, output_format='list'):
    """
    Generates a multidimensional Poisson MIP (multiple interaction process)
    plus independent Poisson processes, all  with non stationary firing rate
    profiles.

    A Poisson MIP consists of Poisson time series which are independent
    except for simultaneous events, in which, differently from the SIP in which
    is constant, the number of neurons involved is distribuited as a binomial
    variable op parameter (pi,M) in all of them. This routine generates
    a MIP plus additional parallel independent Poisson processes.

    **Args**:
      M : int
          number of Poisson processes with coincident events to be generated.
          These will be the first M processes returned in the output.
      N : int
          number of independent Poisson processes to be generated.
          These will be the last N processes returned in the output.
      rate_b : neo.AnalogSignal or list
          overall mean rate of the time series to be generated (coincidence
          rate rate_c is subtracted to determine the background rate). Can be:
          * an AnalogSignal, representing the overall mean rate profile of each
          process. If so, it must be higher than rate_c in each time point.
          * an iterable of AnalogSignals (one signal per process), each
            representing the overall mean rate profile of a process. If so,
            the first M entries must be larger than rate_c in each time point.
      rate_c : Quantity (1/time)
          coincidence rate (rate of coincidences for the M-dimensional SIP).
          The SIP time series will have coincident events with rate rate_c
          plus independent 'background' events with rate rate_b-rate_c.
      pi : float (0<=pi<=1)
          The probability that one single neuron patecipate to the simultaneous
          event
      jitter : Quantity (time)
          jitter for the coincident events. If jitter == 0, the events of all
          M correlated processes are exactly coincident. Otherwise, they are
          jittered around a common time randomly, up to +/- jitter.
      tot_coinc : string
          whether the total number of injected coincidences must be determin-
          istic (i.e. rate_c is the actual rate with which coincidences are
          generated) or stochastic (i.e. rate_c is the mean rate of coincid-
          ences):
          * 'det', 'd', or 'deterministic': deterministic rate
          * 'stoc', 's' or 'stochastic': stochastic rate
          Default: 'det'
      min_delay Quantity (time)
          minimum delay between consecutive coincidence times.
      return_coinc : bool
          whether to retutrn the coincidence times for the SIP process
          Default=False
      output_format string
          the output_format used for the output data:
          * 'gdf' : the output is a np ndarray having shape (2,-1). The
                    first column contains the process ids, the second column
                    represents the corresponding event times.
          * 'list': the output is a list of M+N sublists. Each sublist repres-
                    ents one process among those generated. The first M lists
                    contain the injected coincidences, the last N ones are
                    independent Poisson processes.
          * 'dict': the output is a dictionary whose keys are process IDs and
                    whose values are np arrays representing process events.
          Default: 'list'
    **OUTPUT**:
      realization of a SIP consisting of M Poisson processes characterized by
      synchronous events (with the given jitter), plus N independent Poisson
      time series. The output output_format can be either 'gdf', list or
      dictionary (see output_format argument). In the last two cases a time
      unit is assigned to the output times (same as T's. Default to sec).

      If return_coinc == True, the coincidence times are returned as a second
      output argument. They also have an associated time unit (same as T's.
      Default to sec).

    .. note::
        See also: poisson(), sip_poisson(), msip_poisson(), sip_nonstat
        mip_gen(), genproc_mmip_poisson()

    *************************************************************************
    EXAMPLE:

    >>> import quantities as qt
    >>> import jelephant.core.stocmod as sm
    >>> sip, coinc = sm.sip_poisson(M=10, N=0, T=1*qt.sec, \
            rate_b=20*qt.Hz,  rate_c=4, return_coinc = True)

    *************************************************************************
    """

    # return empty objects if N=M=0:
    if N == 0 and M == 0:
        if output_format == 'list':
            return [] * pq.s
        elif output_format == 'gdf':
            return np.array([[]] * 2).T
        elif output_format == 'dict':
            return {}

    # Assign time unit to jitter, or check that its existing unit is a time
    # unit
    jitter = abs(jitter)

    # Define the array of rates from input argument rate. Check that its length
    # matches with N
    if type(rate_b) == neo.core.analogsignal.AnalogSignal:
        rate_b = [rate_b for i in range(N + M)]
    elif N + M != len(rate_b):
        raise ValueError(
            "*N* != len(*rate_b*). Either specify rate_b as"
            "a vector, or set the number of spike trains by *N*")
    # Check: rate_b>rate_c
    if np.min([np.min(r) for r in rate_b]) < rate_c.rescale(
        rate_b[0].units).magnitude:
        raise ValueError('all elements of *rate_b* must be >= *rate_c*')

    # Check min_delay < 1./rate_c
    if not (rate_c == 0 or min_delay < 1. / rate_c):
        raise ValueError(
            "'*min_delay* (%s) must be lower than 1/*rate_c* (%s)." %
            (str(min_delay), str((1. / rate_c).rescale(min_delay.units))))

    # Check that the variable decimals is integer or None.

    T = rate_b[0].t_stop
    start = rate_b[0].t_start
    # Generate the N independent Poisson processes
    if N == 0:
        independ_poisson_trains = [] * T.units
    else:
        independ_poisson_trains = poisson_nonstat(
            rate_signal=rate_b[M:])
        # Convert the trains from neo SpikeTrain objects to  simpler Quantity
        # objects
        independ_poisson_trains = [
            pq.Quantity(ind.base) * ind.units
            for ind in independ_poisson_trains]

    # Generate the M Poisson processes there are the basis for the MIP
    # (coincidences still lacking)
    if M == 0:
        embedded_poisson_trains = [] * T.units
    else:
        embedded_poisson_trains = poisson_nonstat(
            rate_signal=[r - rate_c for r in rate_b[:M]])
        # Convert the trains from neo SpikeTrain objects to simpler Quantity
        # objects
        embedded_poisson_trains = [
            pq.Quantity(emb.base) * emb.units
            for emb in embedded_poisson_trains]
    # Generate the array of times for coincident events in SIP, not closer than
    # min_delay. The array is generated as a quantity from the Quantity class
    # in the quantities module
    if tot_coinc in ['det', 'd', 'deterministic']:
        Nr_coinc = int(((T - start) * rate_c).rescale(pq.dimensionless))
        while 1:
            coinc_times = start + \
                np.sort(np.random.random(Nr_coinc)) * (T - start)
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
    elif tot_coinc in ['s', 'stoc', 'stochastic']:
        while 1:
            coinc_times = poisson(
                rate=rate_c, t_stop=T, t_start=start, n=1)[0]
            if len(coinc_times) < 2 or min(np.diff(coinc_times)) >= min_delay:
                break
        # Convert coinc_times from a neo SpikeTrain object to a Quantity object
        # pq.Quantity(coinc_times.base)*coinc_times.units
        coinc_times = coinc_times.view(pq.Quantity)
        # Set the coincidence times to T-jitter if larger. This ensures that
        # the last jittered spike time is <T
        for i in range(len(coinc_times)):
            if coinc_times[i] > T - jitter:
                coinc_times[i] = T - jitter
    # Replicate coinc_times M times, and jitter each event in each array by
    # +/- jitter (within (start, T))
    embedded_coinc = coinc_times + \
        np.random.random((M, len(coinc_times))) * 2 * jitter - jitter
    embedded_coinc = embedded_coinc + \
        (start - embedded_coinc) * (embedded_coinc < start) - \
        (T - embedded_coinc) * (embedded_coinc > T)
    #matrix of index of neurons partecipating to each simultaneous event
    delete_mat = np.random.random((M, len(coinc_times))) < pi

#    embedded_coinc = embedded_coinc[delete_mat]

    # Inject coincident events into the M MIP processes generated above, and
    # merge with the N independent processes
    mip_process = [
        np.sort(np.concatenate((
            embedded_poisson_trains[m].rescale(T.units),
            embedded_coinc[m][delete_mat[m]].rescale(T.units))) * T.units)
        for m in xrange(M)]

    # Append the independent spike train to the list of trains
    mip_process.extend(independ_poisson_trains)

    # Convert back sip_process and coinc_times from Quantity objects to
    # neo.SpikeTrain objects
    mip_process = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in mip_process]
    coinc_times = [
        neo.SpikeTrain(t, t_start=start, t_stop=T).rescale(T.units)
        for t in embedded_coinc]

    # Return the processes in the specified output_format
    if output_format == 'list':
        if not return_coinc:
            output = mip_process  # [np.sort(s) for s in sip_process]
        else:
            output = mip_process, coinc_times
    elif output_format == 'gdf':
        neuron_ids = np.concatenate(
            [np.ones(len(s)) * (i + 1) for i, s in enumerate(mip_process)])
        spike_times = np.concatenate(mip_process)
        ids_sortedtimes = np.argsort(spike_times)
        output = np.array(
            (neuron_ids[ids_sortedtimes], spike_times[ids_sortedtimes])).T
        if return_coinc:
            output = output, coinc_times
    elif output_format == 'dict':
        dict_sip = {}
        for i, s in enumerate(mip_process):
            dict_sip[i + 1] = s
        if not return_coinc:
            output = dict_sip
        else:
            output = dict_sip, coinc_times

    return output


def poisson_cos(t_stop, a, b, f, phi=0, t_start=0 * pq.s):
    '''
    Generate a non-stationary Poisson spike train with cosine rate profile r(t)
    given as:
    $$r(t)= a * cos(f*2*\pi*t + phi) + b$$

    Parameters
    ----------
    t_stop : Quantity
        Stop time of the output spike trains
    a : Quantity
        Amplitude of the cosine rate profile. The unit should be of frequency
        type.
    b : Quantity
        Baseline amplitude of the cosine rate modulation. The rate oscillates
        between b+a and b-a. The unit should be of frequency type.
    f : Quantity
        Frequency of the cosine oscillation.
    phi : float (optional)
        Phase offset of the cosine oscillation.
        Default: 0
    t_start : Quantity (optional)
        Start time of each output SpikeTrain.
        Default: 0 s

    Returns
    -------
    SpikeTrain
        Poisson spike train with the expected cosine firing rate profile.
    '''

    # Generate Poisson spike train at maximum rate
    max_rate = (a + b)
    poiss = poisson(rate=max_rate, t_stop=t_stop, t_start=t_start)[0]

    # Calculate rate profile at each spike time
    cos_arg = (2 * f * np.pi * poiss).simplified.magnitude
    rate_profile = b + a * np.cos(cos_arg + phi)

    # Accept each spike at time t with probability r(t)/max_rate
    u = np.random.uniform(size=len(poiss)) * max_rate
    spike_train = poiss[u < rate_profile]

    return spike_train


def _sample_int_from_pdf(a, n):
    '''
    Draw n independent samples from the set {0,1,...,L}, where L=len(a)-1,
    according to the probability distribution a.
    a[j] is the probability to sample j, for each j from 0 to L.


    Parameters
    -----
    a [array|list]
        Probability vector (i..e array of sum 1) that at each entry j carries
        the probability to sample j (j=0,1,...,len(a)-1).

    n [int]
        Number of samples generated with the function

    Output
    -------
    array of n samples taking values between 0 and n=len(a)-1.
    '''

    # a = np.array(a)
    A = np.cumsum(a)  # cumulative distribution of a
    u = np.random.uniform(0, 1, size=n)
    U = np.array([u for i in a]).T  # copy u (as column vector) len(a) times
    return (A < U).sum(axis=1)


def _pool_two_spiketrains(a, b, range='inner'):
    '''
    Pool the spikes of two spike trains a and b into a unique spike train.

    Parameters
    ----------
    a, b : neo.SpikeTrains
        Spike trains to be pooled

    range: str, optional
        Only spikes of a and b in the specified range are considered.
        * 'inner': pool all spikes from min(a.tstart_ b.t_start) to
           max(a.t_stop, b.t_stop)
        * 'outer': pool all spikes from max(a.tstart_ b.t_start) to
           min(a.t_stop, b.t_stop)
        Default: 'inner'

    Output
    ------
    neo.SpikeTrain containing all spikes in a and b falling in the
    specified range
    '''

    unit = a.units
    times_a_dimless = list(a.view(pq.Quantity).magnitude)
    times_b_dimless = list(b.rescale(unit).view(pq.Quantity).magnitude)
    times = (times_a_dimless + times_b_dimless) * unit

    if range == 'outer':
        start = min(a.t_start, b.t_start)
        stop = max(a.t_stop, b.t_stop)
        times = times[times > start]
        times = times[times < stop]
    elif range == 'inner':
        start = max(a.t_start, b.t_start)
        stop = min(a.t_stop, b.t_stop)
    else:
        raise ValueError('range (%s) can only be "inner" or "outer"' % range)
    pooled_train = neo.SpikeTrain(
        times=sorted(times.magnitude), units=unit, t_start=start, t_stop=stop)
    return pooled_train


def _pool_spiketrains(trains, range='inner'):
    '''
    Pool spikes from any number of spike trains into a unique spike train.

    Parameters
    ----------
    trains [list]
        list of spike trains to merge

    range: str, optional
        Only spikes of a and b in the specified range are considered.
        * 'inner': pool all spikes from min(a.tstart_ b.t_start) to
           max(a.t_stop, b.t_stop)
        * 'outer': pool all spikes from max(a.tstart_ b.t_start) to
           min(a.t_stop, b.t_stop)
        Default: 'inner'

    Output
    ------
    neo.SpikeTrain containing all spikes in trains falling in the
    specified range
    '''

    merge_trains = trains[0]
    for t in trains[1:]:
        merge_trains = _pool_two_spiketrains(merge_trains, t, range=range)
    start, stop = merge_trains.t_start, merge_trains.t_stop
    merge_trains = sorted(merge_trains)
    merge_trains = np.squeeze(merge_trains)
    merge_trains = neo.SpikeTrain(
        merge_trains, t_stop=stop, t_start=start, units=trains[0].units)
    return merge_trains


def _mother_proc_cpp_stat(A, T, r, start=0 * pq.ms):
    '''
    Generate the hidden ("mother") Poisson process for a Compound Poisson
    Process (CPP).


    Parameters
    ----------
    r : Quantity, Hz
        Homogeneous rate of the n spike trains that will be genereted by the
        CPP function
    A : array
        Amplitude distribution. A[j] represents the probability of a
        synchronous event of size j.
        The sum over all entries of a must be equal to one.
    T : Quantity (time)
        The stopping time of the mother process
    start : Quantity (time). Optional, default is 0 ms
        The starting time of the mother process


    Output
    ------
    Poisson spike train representing the mother process generating the CPP
    '''

    N = len(A) - 1
    exp_A = np.dot(A, range(N + 1))  # expected value of a
    exp_mother = (N * r) / float(exp_A)  # rate of the mother process
    return poisson(rate=exp_mother, t_stop=T, t_start=start)[0]


def _mother_proc_cpp_cos(A, T, a, b, w, phi, start=0 * pq.ms):
    '''
    Generate the hidden ("mother") Poisson process for a non-stationary
    Compound Poisson Process (CPP) with oscillatory rates
                    r(t)=a * cos(w*2*\pi*t + phi) + b

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        stop time of the output spike trains
    a : Quantity (1/time)
        amplitude of the cosine rate profile
    b : Quantity (1/time)
        baseline of the cosine rate modulation. In a full period, the
        rate oscillates between b+a and b-a.
    w : Quantity (1/time)
        frequency of the cosine oscillation
    phi : float
        phase of the cosine oscillation
    start : Quantity (time). Optional, default to 0 s
        start time of each output spike trains

    Output
    ------
    Poisson spike train representing the mother process generating the CPP
    '''

    N = len(A) - 1  # Number of spike train in the CPP
    exp_A = float(np.dot(A, xrange(N + 1)))  # Expectation of A
    spike_train = poisson_cos(
        t_stop=T, a=N * a / exp_A, b=N * b / exp_A, f=w, phi=phi,
        t_start=start)

    return spike_train


def _cpp_hom_stat(A, T, r, start=0 * pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    A : array
        Amplitude distribution. A[j] represents the probability of a
        synchronous event of size j.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        The end time of the output spike trains
    r : Quantity (1/time)
        Average rate of each spike train generated
    start : Quantity (time). Optional, default to 0 s
        The start time of the output spike trains

    Output
    ------
    List of n neo.SpikeTrains, having average firing rate r and correlated
    such to form a CPP with amplitude distribution a
    '''

    # Generate mother process and associated spike labels
    mother = _mother_proc_cpp_stat(A=A, T=T, r=r, start=start)
    labels = _sample_int_from_pdf(A, len(mother))

    N = len(A) - 1  # Number of trains in output

    try:  # Faster but more memory-consuming approach
        M = len(mother)  # number of spikes in the mother process
        spike_matrix = np.zeros((N, M), dtype=bool)
        # for each spike, take its label l
        for spike_id, l in enumerate(labels):
            # choose l random trains
            train_ids = random.sample(xrange(N), l)
            # and set the spike matrix for that train
            for train_id in train_ids:
                spike_matrix[train_id, spike_id] = True  # and spike to True

        times = [[] for i in range(N)]
        for train_id, row in enumerate(spike_matrix):
            times[train_id] = mother[row].view(pq.Quantity)

    except MemoryError:  # Slower (~2x) but less memory-consuming approach
        print('memory case')
        times = [[] for i in range(N)]
        for t, l in zip(mother, labels):
            train_ids = random.sample(xrange(N), l)
            for train_id in train_ids:
                times[train_id].append(t)

    trains = [neo.SpikeTrain(
        times=t, t_start=start, t_stop=T) for t in times]
#        times=t, units=T.units, t_start=start, t_stop=T) for t in times]

    return trains


def _cpp_het_stat(A, T, r, start=0.*pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        The end time of the output spike trains
    r : Quantity (1/time)
        Sequence of values (of length len(A)-1), each indicating the
        firing rate of one process in output
    start : Quantity (time). Optional, default to 0 s
        The start time of the output spike trains

    Output
    ------
    List of neo.SpikeTrains with different firing rates, forming
    a CPP with amplitude distribution A
    '''

    # Computation of Parameters of the two CPPs that will be merged
    # (uncorrelated with heterog. rates + correlated with homog. rates)
    N_output = len(r)  # number of output spike trains
    N = len(r[r != 0*pq.Hz])  # Number of non-empty spike trains
    if N < N_output:
        # store the indx of rates=0Hz
        idx_zeros = np.arange(N_output)[r == 0*pq.Hz]
        if len(idx_zeros) == N_output:
            # if all the rates are 0Hz return all empty spike trains
            return [
                neo.SpikeTrain([]*T.units, t_stop=T) for i in range(N_output)]
        r = r[r != 0*pq.Hz]  # consider only the positive rates
        # Check if the maximum number of correlated neurons is lower than the
        # number of non-empty neurons
        if sum(A[N + 1:]) > 0:
            raise ValueError(
                'The number of positive entries of A is larger than the' +
                'number of positive rates')
        A = A[:N + 1]
    A_exp = np.dot(A, xrange(N + 1))  # expectation of A
    r_sum = np.sum(r)  # sum of all output firing rates
    r_min = np.min(r)  # minimum of the firing rates
    r1 = r_sum - N * r_min  # rate of the uncorrelated CPP
    r2 = r_sum / float(A_exp) - r1  # rate of the correlated CPP
    r_mother = r1 + r2  # rate of the hidden mother process

    # Check the analytical constraint for the amplitude distribution
    if A[1] < (r1 / r_mother).rescale(pq.dimensionless).magnitude:
        raise ValueError('A[1] too small / A[i], i>1 too high')

    # Compute the amplitude distrib of the correlated CPP, and generate it
    a = [(r_mother * i) / float(r2) for i in A]
    a[1] = a[1] - r1 / float(r2)
    CPP = _cpp_hom_stat(a, T, r_min, start)

    # Generate the independent heterogeneous Poisson processes
    POISS = [poisson(i - r_min, T, start)[0] for i in r]

    # Pool the correlated CPP and the corresponding Poisson processes
    out = [_pool_two_spiketrains(CPP[i], POISS[i]) for i in range(N)]
    if len(out) < N_output:
        for i in idx_zeros:
            # add an empty spike train for the rate=0*pq.Hz
            out.insert(i, neo.SpikeTrain([]*T.units, t_stop=T))
    return out


def cpp(A, t_stop, rate, t_start=0 * pq.s, jitter=None):
    '''
    Generate a Compound Poisson Process (CPP) with a given amplitude
    distribution A and stationary marginal rates r.

    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of len(A)-1
    spike trains with a correlation structure determined by the amplitude
    distribution A: A[j] is the probability that a spike occurs synchronously
    in any j spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to j of the output spike trains with
    probability A[j].

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    t_stop : Quantity (time)
        The end time of the output spike trains
    rate : Quantity (1/time)
        Average rate of each spike train generated. Can be:
        * single-valued: if so, all spike trains will have same rate rate
        * a sequence of values (of length len(A)-1), each indicating the
          firing rate of one process in output
    t_start : Quantity (time). Optional, default to 0 s
        The t_start time of the output spike trains
    jitter : None or Quantity
        If None the corelations are perfectly synchronous, in the case jitter
        is a quantity object all the spike trains are shifted of a random in
        the interval [-jitter, +jitter].
        Default: None

    Returns
    -------
    List of SpikeTrain
        SpikeTrains with specified firing rates forming the CPP with amplitude
        distribution A.
    '''
    if abs(sum(A)-1) > np.finfo('float').eps:
        raise ValueError(
            'A must be a probability vector, sum(A)= %i !=1' % int(sum(A)))
    if any([a < 0 for a in A]):
        raise ValueError(
            'A must be a probability vector, all the elements of must be >0')
    if rate.ndim == 0:
        cpp = _cpp_hom_stat(A=A, T=t_stop, r=rate, start=t_start)
    else:
        cpp = _cpp_het_stat(A=A, T=t_stop, r=rate, start=t_start)
    if jitter is None:
        return cpp
    else:
        cpp = [
            surr.train_shifting(cp, shift=jitter, edges=']')[0] for cp in cpp]
        return cpp


def mip_gen(M, N, t_stop, rate_b, rate_c, pi, t_start=0*pq.s, jitter=None):
    """
    Generates a multidimensional Poisson MIP (multiple interaction process)
    plus independent Poisson processes

    A Poisson MIP consists of Poisson time series which are independent
    except for simultaneous events in whichparteipate a random number(binomial)
    of them. This routine generates a MIP plus additional parallel independent
    Poisson processes.

    Parameters
    ----------
    M : int
        number of Poisson processes with coincident events to be generated.
        These will be the first M processes returned in the output.
    N : int
        number of independent Poisson processes to be generated.
        These will be the last N processes returned in the output.
    t_stop : Quantity (time)
        total time of the simulated processes. The events are drawn between
        t_start and t_stop
        to T (recommended)
    rate_b : Quantity (1/time)
        overall mean rate of the time series to be generated (coincidence
        rate rate_c is subtracted to determine the background rate)
    rate_c : Quantity (1/time)
        coincidence rate (rate of coincidences for the M-dimensional SIP).
        The SIP time series will have coincident events with rate rate_c
        plus independent 'background' events with rate rate_b-rate_c.
    pi : float (0<=pi<=1)
        parameter of the Binomial distribution of number of neurons
        partecipating at each synchronous event. It as to be such that 0<=pi<=1
        representing the probability of synaptic failure. In case of pi=1 the
        process generate is equivalent to SIP model, in case pi=0 to
        independent Poisson processes
    t_start : Quantity (time)
        starting time of the series. If specified, it must be lower than
        t_stop
    jitter : None or Quantity
        If None the corelations are perfectly synchronous, in the case jitter
        is a quantity object all the spike trains are shifted of a random in
        the interval [-jitter, +jitter].
        Default: None
    Returns
    -------
    List of SpikeTrains
      Realization of a MIP consisting of M Poisson processes characterized by
      synchronous events, plus N independent Poisson time series.
    """
    trains = []
    a = [0] * (M + 1)
    rate_b_m = rate_b.magnitude
    rate_c_m = rate_c.magnitude

    #Computation of the ampitude of the MIP
    a[1] = 1 - rate_c_m / float(M * rate_b_m - (M * pi - 1) * rate_c_m)
    a[-1] = rate_c_m / float(M * rate_b_m - (M * pi - 1) * rate_c_m)
    a_MIP = [
        (scipy.special.binom(M, k) * pi**k * (1-pi)**(M-k))*a[-1] for
        k in range(M+1)]
    a_MIP[1] = a_MIP[1] + a[1]

    #Generation of the MIP
    mip = cpp(
        A=a_MIP, t_stop=t_stop, rate=rate_b, t_start=t_start, jitter=None)

    #Generation of the independent Poisson processes
    if N == 0:
        poiss = []
    else:
        poiss = poisson(rate_b, t_stop=t_stop, t_start=t_start, n=N)

    trains = mip + poiss
    return trains


def cpp_cos(A, T, a, b, w, phi, start=0 * pq.s):
    '''
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and non-stationary firing rates
                    r(t)=a * cos(w*2*\pi*t + phi) + b
    for all output spike trains.

    The CPP process is composed of n=length(A)-1 parallel Poisson spike
    trains with correlation structure determined by A: A[j] is the
    probability that a spike is occurring synchronously in j spike trains


    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    T : Quantity (time)
        stop time of the output spike trains
    a : Quantity (1/time)
        amplitude of the cosine rate profile
    b : Quantity (1/time)
        baseline of the cosine rate modulation. The rate oscillates
        between b+a and b-a.
    w : Quantity (1/time)
        frequency of the cosine oscillation
    phi : float
        phase of the cosine oscillation
    start : Quantity (time). Optional, default to 0 s
        start time of each output spike trains

    Output
    ------
    list of spike trains with same sinusoidal rates profile, and forming
    a CPP with specified amplitude distribution.
    '''

    N = len(A) - 1  # number of trains in output
    mother = _mother_proc_cpp_cos(A, T, a, b, w, phi, start=0 * pq.ms)
    labels = _sample_int_from_pdf(A, len(mother))

    try:  # faster but more memory-consuming approach
        M = len(mother)  # number of spikes in the mother process
        spike_matrix = np.zeros((N, M), dtype=bool)

        for spike_id, l in enumerate(labels):  # for each spike label l,
            train_ids = random.sample(xrange(N), l)  # choose l random trains
            for train_id in train_ids:  # and for each of them
                spike_matrix[train_id, spike_id] = True  # set copy to True

        trains = [mother[row] for row in spike_matrix]

    except MemoryError:  # slower (~2x) but less memory-consuming approach
        times = [[] for i in range(N)]
        for t, l in zip(mother, labels):
            train_ids = random.sample(xrange(N), l)
            for train_id in train_ids:
                times[train_id].append(t)

        trains = [neo.SpikeTrain(
            times=t, units=T.units, t_start=start, t_stop=T) for t in times]

    return trains


def poisson_nonstat_thinning(rate_signal, N=1, cont_sign_method='step'):
    '''
    Generate non-stationary Poisson SpikeTrains with a common rate profile.


    Parameters
    -----
    rate_signal : AnalogSignal
        An AnalogSignal representing the rate profile evolving over time.
        Note that, if rate_profile

    cont_sign_method : str, optional
        The approximation method used to make continuous the analog signal:
        * 'step': the signal is approximed in each nterval of rate_signal.times
          with the value of the signal at the left extrem of the interval
        * 'linear': linear interpolation is used
        Default: 'step'

    Output
    -----
    Poisson SpikeTrain with profile rate lambda(t)= rate_signal
    '''
    if any(rate_signal < 0) or not rate_signal.size:
        raise ValueError(
            'rate must be a positive non empty signal, representing the'
            'rate at time t')
    # Define the interpolation method
    else:
        methods_dic = {
            'linear': _analog_signal_linear_interp,
            'step': _analog_signal_step_interp}

        if cont_sign_method not in methods_dic:
            raise ValueError("Unknown method selected.")

        interp = methods_dic[cont_sign_method]

        #Generate n hidden Poisson SpikeTrains with rate equal to the peak rate
        lambda_star = max(rate_signal)
        poiss = poisson(
            rate=lambda_star, t_stop=rate_signal.t_stop,
            t_start=rate_signal.t_start, n=N)

        # For each SpikeTrain, retain spikes according to uniform probabilities
        # and add the resulting spike train to the list sts
        sts = []
        for st in poiss:
            # Compute the rate profile at each spike time by interpolation
            lamb = interp(signal=rate_signal, times=st.magnitude * st.units)

            # Accept each spike at time t with probability r(t)/max_rate
            u = np.random.uniform(size=len(st)) * lambda_star
            spiketrain = st[u < lamb]
            sts.append(spiketrain)

        return sts


def _analog_signal_linear_interp(signal, times):
    '''
    Compute the linear interpolation of a signal at desired times.

    Given a signal (e.g. an AnalogSignal) AS taking value s0 and s1 at two
    consecutive time points t0 and t1 (t0 < t1), the value s of the linear
    interpolation at time t: t0 <= t < t1 is given by:

                s = ((s1 - s0) / (t1 - t0)) * t + s0,
    for any time t between AS.t_start and AS.t_stop

    NOTE: If AS has sampling period dt, its values are defined at times
    t[i] = s.t_start + i * dt. The last of such times is lower than s.t_stop:
    t[-1] = s.t_stop - dt. For the interpolation at times t such that
    t[-1] <= t <= AS.t_stop, the value of AS at AS.t_stop is taken to be that
    at time t[-1].


    Parameters
    -----
    times : Quantity vector(time)
        The time points for which the interpolation is computed

    signal : neo.core.AnalogSignal
        The analog signal containing the discretization of the funtion to
        interpolate


    Output
    -----
    Quantity array representing the values of the interpolated signal at the
    times given by times
    '''
    dt = signal.sampling_period

    t_start = signal.t_start
    t_stop = signal.t_stop.rescale(signal.times.units)

    # Extend the signal (as a dimensionless array) copying the last value
    # one time, and extend its times to t_stop
    signal_extended = np.hstack([signal.magnitude, signal[-1].magnitude])
    times_extended = np.hstack([signal.times, t_stop]) * signal.times.units

    time_ids = np.floor(((times - t_start) / dt).rescale(
        pq.dimensionless).magnitude).astype('i')

    # Compute the slope m of the signal at each time in times
    y1 = signal_extended[time_ids]
    y2 = signal_extended[time_ids + 1]
    m = (y2 - y1) / dt

    # Interpolate the signal at each time in times by linear interpolation
    # TODO: return as an IrregularlySampledSignal?
    out = (y1 + m * (times - times_extended[time_ids])) * signal.units
    return out.rescale(signal.units)


def _analog_signal_step_interp(signal, times):
    '''
    Compute the step-wise interpolation of a signal at desired times.

    Given a signal (e.g. an AnalogSignal) AS taking value s0 and s1 at two
    consecutive time points t0 and t1 (t0 < t1), the value s of the step-wise
    interpolation at time t: t0 <= t < t1 is given by s=s0, for any time t
    between AS.t_start and AS.t_stop.


    Parameters
    -----
    times : Quantity vector(time)
        The time points for which the interpolation is computed

    signal : neo.core.AnalogSignal
        The analog signal containing the discretization of the funtion to
        interpolate


    Output
    -----
    Quantity aray representing the values of the interpolated signal at the
    times given by times
    '''
    dt = signal.sampling_period

    # Compute the ids of the signal times to the left of each time in times
    time_ids = np.floor(
        ((times - signal.t_start) / dt).rescale(pq.dimensionless).magnitude
        ).astype('i')

    # TODO: return as an IrregularlySampledSignal?
    return(signal.magnitude[time_ids] * signal.units).rescale(signal.units)


def _cumrate(intensity, dt):
    '''
    Cumulative intensity function.

    Parameters
    ----------
    intensity : array(float)
        intensity function (instantaneous rate)
    dt : float
        time resolution

    Output
    ------
    crf : array(float)
        cumulative intensity function

    (Tetzlaff, 2009-02-09)

    '''
    # integral of intensity
    crf = dt.magnitude * np.cumsum(intensity.magnitude)
    return crf


def _invcumrate(crf, csteps=1000):
    '''
    Inverse of the cumulative intensity function.

    Parameters
    ----------
    crf : array(float)
        cumulative intensity function (see cumrate())
    csteps : int, default csteps=1000
        number of steps between min. and max. spike count

    Returns:
    -------
    icrf : array(float)
        inverse of cumulative intensity function
    dc : float
        spike count resolution
    D : float
        expected number of spikes at simulation end

    (Tetzlaff, 2009-02-09)

    '''

    D = crf[-1]  # cumulative spike-count at time T
    dc = D / csteps  # spike-count resolution
    icrf = np.nan * np.ones(csteps, 'f')

    k = 0
    for i in range(csteps):  # loop over spike-count grid
        ## find smallest k such that crf[k]>i*dc
        while crf[k] <= i * dc:
            k += 1

        if k == 0:
            icrf[i] = 0.0
        else:
            # interpolate between crf[pl] and crf[pr]
            m = 1. / (crf[k] - crf[k - 1])  # approximated slope of icrf
            icrf[i] = np.float(k - 1) + m * (
                np.float(i * dc) - crf[k - 1])  # interpolated value of icrf

    return icrf, dc, D


def _poisson_nonstat_single(icrf, dc, D, dt, refr_period = False):
    '''
    Generates an inhomogeneous Poisson process for a given intensity
    (rate function).

    Parameters
    ----------
    icrf  : array(float)
        inverse of cumulative intensity function (see invcumrate())
    dc : float
        spike count resolution (see invcumrate())
    D : float
        expected number of spikes at simulation end (see invcumrate())
    dt     : float
                    time resolution

    Returns
    -------
    spiketimes : array(float)
        array of spike times

    (Tetzlaff, 2009-02-09)

    '''
    # number of spikes in interval [0,T]
    nspikes = np.random.poisson(D)

    # uniform distribution of nspikes spikes in [0,D]
    counts = D * np.sort(np.random.rand(nspikes))

    ind = np.where(np.ceil(counts/dc) + 1 <= len(icrf))
    t1 = icrf[np.floor(counts[ind] / dc).astype('i')]
    t2 = icrf[np.floor(counts[ind] / dc).astype('i') + 1]
    m = t2 - t1
    spiketimes = t1 + m * (counts[ind] / dc + 1 - np.ceil(counts[ind] / dc))
    spiketimes = np.array(spiketimes)
#    spiketimes = spiketimes + (np.array([0]+list(np.diff(spiketimes) > 0.001)))
    if len(spiketimes>0) and refr_period:
        spiketimes = np.hstack(
            (
                spiketimes[0], spiketimes[np.where(
                    np.diff(spiketimes) > (
                        np.random.rand() / 1000. + 0.001))[0]+1]))
    return spiketimes


def poisson_nonstat_time_rescale(rate_signal, N=1, csteps=1000):
    '''
    Generates an ensemble of non-stationary Poisson processes with identical
    intensity.

    Parameters
    ----------
    rate_signal : neo.AnalogSignal
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    N : int
        ensemble size
    csteps : int, default csteps=1000
        spike count resolution
        (number of steps between min. and max. spike count)

    Returns
    -------
    spiketrains : list(list(float))
        list of spike trains (len(spiketrains)=N)
           spiketimes  : array(float)
                         array of spike times

    (Tetzlaff, 2009-02-09, adapted to neo format)

    '''
    if any(rate_signal < 0) or not rate_signal.size:
            raise ValueError(
                'rate must be a positive non empty signal, representing the'
                'rate at time t')
    if not (type(N) == int and N > 0):
            raise ValueError('N (=%s) must be a positive integer' % str(N))
    #rescaling the unit of the signal
    elif np.any(rate_signal > 0) and rate_signal.units == pq.Hz:
        signal_simpl = rate_signal.simplified
        t_start_simpl = rate_signal.t_start.simplified
        t_stop_simpl = rate_signal.t_stop.simplified
        sampling_period_simpl = rate_signal.sampling_period.simplified
        rate_signal = neo.AnalogSignal(
            signal=signal_simpl, t_start=t_start_simpl, t_stop=t_stop_simpl,
            sampling_period=sampling_period_simpl)
        ## rectification of intensity
        dt = rate_signal.sampling_period
        out = []

        ## compute cumulative intensity function and its inverse
        # cumulative rate function (intensity)
        crf = _cumrate(rate_signal, dt)
        # inverse of cumulative intensity
        icrf, dc, D = _invcumrate(crf, csteps)
        icrf *= dt  # convert icrf to time

        ## generate spike trains
        np.random.seed()
        for cn in range(N):
            buf = _poisson_nonstat_single(icrf, dc, D, dt)
            st = neo.SpikeTrain(
                buf, t_stop=rate_signal.t_stop - rate_signal.t_start,
                units=rate_signal.t_stop.units)
#            st = st + rate_signal.t_start
            st = shift_spiketrain(st, rate_signal.t_start)
            st.t_start = rate_signal.t_start
            st.t_stop = rate_signal.t_stop
            out.append(st)
        return out
    elif rate_signal.units == pq.Hz:
        return(
            [neo.SpikeTrain(
                [], t_stop=rate_signal.t_stop,
                units=rate_signal.t_stop.units) for i in range(N)])
    else:
        raise ValueError(
            'rate must be in Hz, representing the rate at time t')


def _mother_proc_cpp_nonstat(A, rate_signal, method='time_rescale'):
    '''
    Generate the "mother" poisson process for a non-stationary
    Compound Poisson Process (CPP) with rate profile described by the
    analog-signal rate_signal.


    Parameters
    -----
    A : array
        Amplitude distribution, representing at each j-th entry the probability
        of a synchronous event of size j.
        The sum over all entries of a must be equal to one.

    rate_signal : neo.core.AnalogSignal, units=Hz
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    method : string
        The method used to generate the non-stationary poisson process:
        *'time_rescale': method based on the time rescaling theorem
        (ref. Brwn et al. 2001)
        *'thinning': thinning method of a stationary poisson process
        (ref. Sigman Notes 2013)
        Default:'time_rescale'


    Output
    -----
    Poisson spike train representing the hidden process generating a CPP model
    with prfile rate lambda(t)=a*cos(w*2*greekpi*t+phi)+b
    '''
    # Dic of non-stat generator methods
    methods_dic = {
        'time_rescale': poisson_nonstat_time_rescale,
        'thinning': poisson_nonstat_thinning}
    if method not in methods_dic:
        raise ValueError("Unknown method selected.")
    method_use = methods_dic[method]
    N = len(A) - 1
    exp_A = np.dot(A, range(N + 1))  # expected value of a
    if exp_A == 0:
        rate_M = rate_signal * 0
    else:
        rate_M = rate_signal * N / float(exp_A)
    return method_use(
        rate_signal=rate_M)[0]


def _cpp_hom_nonstat(A, rate_signal, method='time_rescale'):
    '''
    Generation a compound poisson process (CPP) with amplitude distribution A,
    homogeneus non-stationary profile rate described by the analog-signal
    rate_signal.

    The CPP process is composed of n=length(A)-1 different parallel poissonian
    spike trains with a correlation structure determined by the amplitude
    distribution A.


    Parameters
    -----
    A : array
        Amplitude distribution, representing at each j-th entry the probability
        of a synchronous event of size j.
        The sum over all entries of a must be equal to one.

    rate_signal [neo.core.AnalogSignal, units=Hz]
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    method : string
        The method used to generate the non-stationary poisson process:
        *'time_rescale': method based on the time rescaling theorem
        (ref. Brwn et al. 2001)
        *'thinning': thinning method of a stationary poisson process
        (ref. Sigman Notes 2013)
        Default:'time_rescale'



    Output
    -----
    trains [list f spike trains]
        list of n spike trains all with same rates profile and distribuited as
        a CPP with amplitude given by A
    '''

    N = len(A) - 1  # number of trains in output
    # generation of mother process
    mother = _mother_proc_cpp_nonstat(A, rate_signal, method=method)
    # generation of labels from the amplitude
    labels = _sample_int_from_pdf(A, len(mother))
    N = len(A) - 1  # number of trains in output
    M = len(mother)  # number of spikes in the mother process

    spike_matrix = np.zeros((N, M), dtype=bool)

    for spike_id, l in enumerate(labels):  # for each spike, take its label l,
        train_ids = random.sample(xrange(N), l)  # choose l random trains
        for train_id in train_ids:  # and set the spike matrix for that train
            spike_matrix[train_id, spike_id] = True  # and spike to True
#TODO:delete previous version if the corrected pointer problem works
#    trains = [mother[row] for row in spike_matrix]
    times = [[] for i in range(N)]
    for train_id, row in enumerate(spike_matrix):
            times[train_id] = mother[row].view(pq.Quantity)
    trains = [neo.SpikeTrain(
        times=t, t_start=mother.t_start, t_stop=mother.t_stop) for t in times]
    return trains


def _cpp_het_nonstat(A, signals, method='time_rescale'):
    '''
    Generate a compound poisson process (CPP) with amplitude distribution A,
    heterogeneous non-stationary profiles rate described by a list of
    analog-signals.

    The CPP process is composed of n=length(A)-1 different parallel poissonian
    spike trains with a correlation structure determined by the amplitude
    distribution A.


    Parameters
    -----
    A : array
        Amplitude distribution, representing at each j-th entry the probability
        of a synchronous event of size j.
        The sum over all entries of a must be equal to one.

    signals : list
        The list conteining the analog signals representing the discretization
        on the time axis of the rate profile function of each spike trains to
        generate, in the i-th psition of the list is stored the rate profile
        of the i-th train. All the N signals must have same t_start and t_stop
        and sampling_rate

    method : string
        The method used to generate the non-stationary poisson process:
        *'time_rescale': method based on the time rescaling theorem
        (ref. Brwn et al. 2001)
        *'thinning': thinning method of a stationary poisson process
        (ref. Sigman Notes 2013)
        Default:'time_rescale'


    Output
    -----
    trains [list f spike trains]
        list of n spike trains all with same rates profile and distribuited as
        a CPP with amplitude given by A
    '''
    # Dic of non-stat generator methods
    methods_dic = {
        'time_rescale': poisson_nonstat_time_rescale,
        'thinning': poisson_nonstat_thinning}
    if method not in methods_dic:
        raise ValueError("Unknown method selected.")
    method_use = methods_dic[method]

    #number of neurons
    N = len(signals)

    #signals' parameters
    sampling_period = signals[0].sampling_period
    unit = signals[0].units
    t_start = signals[0].t_start

    # expectation of A
    exp_A = np.dot(A, range(N + 1))

    #analog signal of the minimum signals per bin
    sign_min = neo.AnalogSignal(
        np.min(signals, axis=0), units=unit, sampling_period=sampling_period,
        t_start=t_start)

    #total rates of each  neuron
    r = np.sum(np.array(signals) * unit * sampling_period, axis=1)

    #rate of the mother process
    r_mother = np.sum(r) / float(exp_A)
    r_min = np.sum(sign_min * sampling_period)

    #rate of the mother process of the independent population
    r1 = np.sum(r - r_min)

    #rate of the mother process of the correlated population
    r2 = r_mother - r1

    #analytical constrain
    if A[1] < (r1 / r_mother).rescale(pq.dimensionless).magnitude:
        raise ValueError('A[1] too small / A[i], i>1 too high')

    #new amplitude of the correlated population
    a = [(r_mother * i) / float(r2) for i in A]
    a[1] = a[1] - r1 / float(r2)

    #generation of the correlated population
    cpp = _cpp_hom_nonstat(A=a, rate_signal=sign_min, method=method)

    #generation of the independent population
    signals_indip = neo.AnalogSignal(
        np.array(signals) - np.array([sign_min] * N), units=unit,
        sampling_period=sampling_period, t_start=t_start)
    poiss = [
        method_use(rate)[0] for rate in signals_indip]

    #pool of the two population
    out = [_pool_two_spiketrains(cpp[i], poiss[i]) for i in range(N)]
    return out


def cpp_nonstat(A, rate, method='time_rescale'):
    '''
    Generate a Compound Poisson Process (CPP) with a given amplitude
    distribution A and non-stationary rate profiles rate.

    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of len(A)-1
    spike trains with a correlation structure determined by the amplitude
    distribution A: A[j] is the probability that a spike occurs synchronously
    in any j spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to j of the output spike trains with
    probability A[j].

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    rate : list of neo.AnalogSignal
        Average time-depending rate of each spike train generated. Can be:
        * single-AnalogSignal: if so, all spike trains will have same rate
        profile
        * a sequence of AnalogSignal (of length len(A)-1), each indicating the
          firing rate profile of one process in output
    method : string
        The method used to generate the non-stationary poisson process:
        *'time_rescale': method based on the time rescaling theorem
        (ref. Brwn et al. 2001)
        *'thinning': thinning method of a stationary poisson process
        (ref. Sigman Notes 2013)
        Default:'time_rescale'

    Returns
    -------
    List of SpikeTrain
        SpikeTrains with specified firing rates forming the CPP with amplitude
        distribution A.
    '''
    if abs(sum(A)-1) > np.finfo('float').eps:
        raise ValueError(
            'A must be a probability vector, sum(A)= %i !=1' % int(sum(A)))
    if any([a < 0 for a in A]):
        raise ValueError(
            'A must be a probability vector, all the elements of must be >0')
    #TODO: decide unit (problem Hz or rate.t_start)
    if type(rate) == neo.AnalogSignal:
        return _cpp_hom_nonstat(A=A, rate_signal=rate, method=method)
    elif len(rate) == 1:
        return _cpp_hom_nonstat(A=A, rate_signal=rate[0], method=method)
    else:
        return _cpp_het_nonstat(A=A, signals=rate, method=method)


def cpp_corrcoeff(ro, xi, t_stop, rate, N, t_start=0 * pq.s):
    '''
    Generation a compound poisson process (CPP) with a prescribed pairwise
    correlation coefficient ro.

    The CPP process is composed of N different parallel poissonian
    spike trains with a correlation structure determined by the correlation
    coefficient ro and maximum order of correlation xi.


    Parameters
    ----------
    ro : float
        Pairwise correlation coefficient of the population, $0 <= ro <= 1$.
    xi : int
        Maximum order of correlation of the neuron population, $1 <= xi <= N$.
    t_stop : Quantity
        The stopping time of the output spike trains.
    rate : Quantity
        Average rate of each spike train generated expressed in units of
        frequency.
    N : int
        Number of parallel spike trains to create.
    t_start : Quantity (optional)
        The starting time of the output spike trains.

    Returns
    -------
    list of SpikeTrain
        list of N spike trains all with same rate and distributed as a CPP with
        correlation cefficient ro and maximum order of crrelation xi.
    '''

    if xi > N or xi < 1:
        raise ValueError('xi must be an integer such as 1 <= xi <= N.')
    if (xi ** 2 - xi - (N - 1) * ro * xi) / float(
            xi ** 2 - xi - (N - 1) * ro * xi + (N - 1) * ro) < 0 or (
            xi ** 2 - xi - (N - 1) * ro * xi) / float(
            xi ** 2 - xi - (N - 1) * ro * xi + (N - 1) * ro) > 1:
        raise ValueError(
            'Analytical check failed: ro= %f too big with xi= %d' % (ro, xi))
    # Computation of the pick amplitude for xi=1
    nu = (xi ** 2 - xi - (N - 1) * ro * xi) / float(
        xi ** 2 - xi - (N - 1) * ro * xi + (N - 1) * ro)

    # Amplitude vector in the form A=[0,nu,0...0,1-nu,0...0]
    A = [0] + [nu] + [0] * (xi - 2) + [1 - nu] + [0] * (N - xi)
    return cpp(A=A, t_stop=t_stop, rate=rate, t_start=t_start)


# theoretical correlation coefficient
def corrcoef_CPP(A, N):
    """
    Compute the pairwise correlation coefficient c of a CPP with N neurons and
    amplitude distribution A. The coefficient is computed according to the
    following equaion:
    c = (E[A^2]/E[A] - 1)/N-1

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    N : int
        number of neurons in the population


    Reference:
     Staude et al. (2010)
    """
    exp_xi = 0.
    exp_xi = np.sum(A*np.arange(len(A)))
    exp_xi2 = np.sum(A*np.arange(len(A))**2)
    return ((exp_xi2/exp_xi) - 1)/(N - 1.)


def synfirechain(
    t_stop, rate_tot, rate_sf, l, w, d, tj=0 * pq.s, n_ind=0,
    rate_ind=0 * pq.Hz, t_start=0 * pq.ms, rnd_link_times=True,
    return_sf_spikes=False, return_sf_starts=False, return_params=False):

    """
    Generate spike trains containing synfire chain (SFC) activity,
    where synchronous spiking propagate across links of neurons.

    Parameters:
    -----------
    t_stop : Quantity
        end of the simulation time. The start is 0 s by default.
        All generated spike trains are in the range [t_start, t_stop]
    rate_tot : Quantity
        total firing rate (deterministic) of each neuron in the SFC, i.e.
        number of spikes per time unit.
    rate_sf : Quantity
        rate of SFC activation (deterministic)
    l : int
        number of layers composing the synfire chain
    w : int
        number of neurons per layer
    d : float
        time delay from one link to the next (in ms)
    tj : float, optional
        temporal jitter of the spikes composing the activation of each link
        in the SFC, around the common activation time of that link. Those
        spikes are jittered around the original time randomly, up to +/- tj.
        Default: 0 ms
    n_ind : int, optional
        number of independent spike trains, simulated in addition to the
        spike trains composing the SFC activity.
        Default: 0
    rate_ind : float, optional
        rate of the uncorrelated neurons (Hz)
        Default: 0 Hz
    rnd_link_times : bool, optional
        if True, the spike times per sf run are randomized. if False, the
        synfire chain spikes repeat identical in all of its occurrences.
        Default: True
    return_sf_spikes : bool, optional
        whether to return the spikes composing the SFC activity as an
        additional optional output argument
    return_sf_times : bool, optional
        whether to return the activation (i.e. start) times of the SFC runs
        as an additional optional output argument
    return_params : bool, optional
        whether to return the parameters used for the simulation as an
        additional optional output argument

    Returns:
    --------
    sts : list
        List of SpikeTrain. The first l*w elements are the spike trains
        involved in the SFC activity, the others are additional independent
        spike trains.
        Each SpikeTrain has an annotation dictionary with the following keys:
        * 'spiketrain_id': a unique integer ID per spike train, starting at 0
        * 'link_id': the ID of the SFC link the spike train belongs to (int
          from 0 to l-1, or None for the independent trains)
        * 'unit_id_in_link': the neuron ID the spike train represents within
          its link (int from 0 to w-1, or None for the independent trains)
        * 'in_sfc': bool, stating whether the spike train belongs to the SFC
          activity (True) or not (False)
        * 'indep': bool, stating whether the spike train belongs to the
          group of independent spike trains (True) or not (False)

    Optional output arguments:

    sf_spikes : list of SpikeTrain
        (returned if return_sf_spikes = True)
        list of l * w spike trains, containing the spikes forming the SFC
        activity only. The spike trains have the same annotations as sts
    sf_starts : Quantity array
        (returned if return_sf_times = True)
        start times of each SFC activation in the simulation
    params : dict
        (returned if return_params = True)
        dictionary of parameters used for the simulation
    """

    # Return the independent spike trains only, if l==0 or w==0 or rate_sf=0
    if l == 0 or w == 0 or rate_sf == 0 * pq.Hz:
        sf_spikes = []
        t_sf = [] * t_stop.units
        sts = [] if n_ind == 0 else poisson(
            rate=rate_ind, t_stop=t_stop, t_start=t_start, n=n_ind)

        # Define and return the output
        output = sts
        only_sts_in_output = True

        if return_sf_spikes is True:
            if only_sts_in_output:
                output = [sts, sf_spikes]
                only_sts_in_output = False
            else:
                output.append(sf_spikes)

        if return_sf_starts is True:
            if only_sts_in_output:
                output = [sts, t_sf]
                only_sts_in_output = False
            else:
                output.append(t_sf)

        if return_params is True:
            params = {
                't_stop': t_stop, 'rate_sf': rate_sf, 'l': l, 'w': w, 'd': d,
                'tj': tj, 'rate_tot': rate_tot, 'n_ind': n_ind,
                'rate_ind': rate_ind, 'rnd_link_times': rnd_link_times}
            if only_sts_in_output:
                output = [sts, params]
                only_sts_in_output = False
            else:
                output.append(params)

        return output

    # Define the background rate for SF spike trains
    rate_bg = rate_tot - rate_sf
    if rate_bg < 0 * pq.Hz:
        raise ValueError(
            'rate_tot (%s) cannot be lower that rate_sf (%s)' % \
            (rate_tot, rate_sf))

    dT = t_stop - t_start  # Total simulation time

   # Define a time unit to use implicitely in each ndarray below
    time_unit = t_stop.units

    # Generate the (deterministic) nr. of SFC runs and their starting times:
    # TODO: make the number of SFC runs stochastic?
    n_sf = int((rate_sf * dT).rescale(pq.dimensionless))
    t_sf = t_start + tj + np.sort(np.random.random(n_sf)) * (
        dT - d * l - 2 * tj)
    t_sf = t_sf.rescale(time_unit)

    # Generate times of spikes involved in SFC runs, as a matrix of shape
    # n_sf x l x w. st[i,j,k]: i-th run, j-th link, k-th neuron in the link
    st_sf = np.array([[(np.array([ii.magnitude] * w) * ii.units + jj * d
        ).rescale(time_unit).magnitude for jj in xrange(l)] for ii in t_sf])

    # Randomize the spike times of the synfire chain:
    if rnd_link_times is False:
        st_sf[:] += (np.random.random((l, w)) * (2 * tj) - tj).rescale(
            time_unit).magnitude
    elif rnd_link_times is True:
        st_sf += (np.random.random((n_sf, l, w)) * (2 * tj) - tj).rescale(
            time_unit).magnitude

    # Compute the matrix n_sf x l x w of neuron ids associated to the
    # spikes in the synfire chain:
    ids_sf = np.ones((n_sf, l, w))
    ids_sf[:] += np.array([[jj * w + kk for kk in xrange(w)]
        for jj in xrange(l)])

    # Create the gdf-format array of spike ids and spike times
    # (sorted by increasing spike times)
    gdf_sf = np.array([np.array([ids_sf[ii, jj, kk], st_sf[ii, jj, kk]])
        for kk in xrange(w) for jj in xrange(l) for ii in xrange(n_sf)])
    gdf_sf = gdf_sf[np.argsort(gdf_sf[:, 1])]

    # Generate times of background spikes (same amount for all neurons!)
    # TODO: make this number stochastic!
    n_b = int((rate_bg * dT).rescale(pq.dimensionless))
    st_b = (np.sort(np.random.random((n_b, l, w)) * dT, axis=0) + \
        t_start).rescale(time_unit).magnitude

    # Compute the matrix n_sf x l x w of neuron ids associated to the
    # background spikes (same as ids_sf):
    ids_b = np.ones((n_b, l, w))
    ids_b[:] += np.array([[jj * w + kk for kk in xrange(w)]
        for jj in xrange(l)])

    # Create the gdf-format array of spike ids and spike times
    # (sorted along increasing spike times):
    if rate_bg == 0:
        gdf_b = np.transpose([[], []])
    else:
        gdf_b = np.array([np.array([ids_b[ii, jj, kk], st_b[ii, jj, kk]])
            for kk in xrange(w) for jj in xrange(l) for ii in xrange(n_b)])
        gdf_b = gdf_b[np.argsort(gdf_b[:, 1])]

    # Create the spike times for independent neurons
    # TODO: generate as list of spike trains using stocmod.poisson()
    # TODO: make the number of spikes stochastic!
    if rate_ind <= 0 * pq.Hz or n_ind <= 0:
        gdf_ind = np.transpose([[], []])
    else:
        nr_ind_spikes = int((rate_ind * dT).rescale(pq.dimensionless))
        st_ind = (np.sort(np.random.random((nr_ind_spikes, n_ind)) * dT,
            axis=0) + t_start).rescale(time_unit).magnitude
        ids_ind = np.array([range(1, n_ind + 1) for ii in
            xrange(nr_ind_spikes)]) + w * l
        gdf_ind = np.array([np.array([ids_ind[ii, jj], st_ind[ii, jj]])
            for jj in xrange(n_ind) for ii in xrange(nr_ind_spikes)])
        gdf_ind = gdf_ind[np.argsort(gdf_ind[:, 1])]

    # Merge all gdf's into a single array, and sort by increasing spike times
    gdf_tot = np.concatenate((gdf_sf, gdf_b, gdf_ind), axis=0)
    gdf_tot = gdf_tot[np.argsort(gdf_tot[:, 1])]

    # Convert gdf_tot to spike trains, and place the neuron id as annotation
    sts = []
    neur_ids = np.unique(gdf_tot[:, 0])
    sfcneur_ids = np.unique(gdf_sf[:, 0])
    for n_id in neur_ids:
        st = neo.SpikeTrain(
            times=gdf_tot[gdf_tot[:, 0] == n_id, 1] * time_unit,
            t_stop=t_stop, t_start=t_start)
        st.annotations['spiketrain_id'] = n_id
        st.annotations['link_id'] = n_id // w if n_id < l * w else None
        st.annotations['unit_id_in_link'] = n_id % w if n_id < l * w else None
        st.annotations['in_sfc'] = True if n_id in sfcneur_ids else False
        st.annotations['indep'] = False if n_id in sfcneur_ids else True
        sts.append(st)

    # Convert gdf_sf to spike trains, and place the neuron id as annotation
    sf_spikes = []
    for n_id in sfcneur_ids:
        st = neo.SpikeTrain(
            times=gdf_sf[gdf_sf[:, 0] == n_id, 1] * time_unit,
            t_stop=t_stop, t_start=t_start)
        st.annotations['id'] = n_id
        st.annotations['link_id'] = n_id // w if n_id < l * w else None
        st.annotations['unit_id_in_link'] = n_id % w if n_id < l * w else None
        st.annotations['in_sfc'] = True if n_id in sfcneur_ids else False
        st.annotations['indep'] = False if n_id in sfcneur_ids else True
        sf_spikes.append(st)

    # Define and return the output
    output = sts
    only_sts_in_output = True

    if return_sf_spikes is True:
        if only_sts_in_output:
            output = [sts, sf_spikes]
            only_sts_in_output = False
        else:
            output.append(sf_spikes)

    if return_sf_starts is True:
        if only_sts_in_output:
            output = [sts, t_sf]
            only_sts_in_output = False
        else:
            output.append(t_sf)

    if return_params is True:
        params = {
            't_stop': t_stop, 'rate_sf': rate_sf, 'l': l, 'w': w, 'd': d,
            'tj': tj, 'rate_tot': rate_tot, 'n_ind': n_ind,
            'rate_ind': rate_ind, 'rnd_link_times': rnd_link_times}
        if only_sts_in_output:
            output = [sts, params]
            only_sts_in_output = False
        else:
            output.append(params)

    return output


# TODO:
# function to calculate the rate of mother process
# from the rate of child processes
# E[lambda_bg] = (lambda_mp/N) * \sum_{xi=1}^{xi=N}xi*f_A(xi)

def gamma_thinning(t_stop, shape, rate, N=None, t_start=0*pq.s):
    '''
    Generate a Renewal process with Gamma isi distribution of parameter
    (shape, rate). In paticular the output spiketrain will have a mean firing
    rate equal to the parameter rate=scale*shape where scale and shape are the
    parameter of a gamma distribution (e.g. in numpy.random.gamma())

    The process is generated via thinning of a Poisson process keeping only one
    event each shape (int parameter of the Gamma)


    Parameters
    -----
    t_stop : Quantity
        The stopping time of the spike train in output

    shape : int
        The parameter of the Gamma distribution

    rate : Quantity
        The meaning firing rate of the train in output

     N : int or None (optional)
        If rate is a single Quantity value, n specifies the number of
        SpikeTrains to be generated. If rate is an array, n is ignored and the
        number of SpikeTrains is equal to len(rate).
        Default: None

    t_start: Quantity (Optional)
        The starting time of the spike train
        Default: 0*pq.s
    Output
    -----
    spiketrain : list
        list neo.SpikeTrain with ISI distribution gamma(1/rate,shape)
    '''
    if type(shape) == int:
        if t_start < t_stop:
            #Poisson process to be thinned
            poiss = poisson(
                rate=shape*(rate), t_stop=t_stop, t_start=t_start -
                10. / np.max(rate),
                n=N)

        #    #Thinning
            spiketrains = [st[0::shape] for st in poiss]
            spiketrains = [st[st > t_start] for st in spiketrains]
            for st in spiketrains:
                st.t_start = t_start
            return spiketrains
        else:
            raise ValueError('t_start has to be lower than t_stop')
    else:
        raise ValueError(
            'The shape parameter of the gamma distribution has to be an'
            'integer')


def gamma_nonstat_rate(rate_signal, shape, N=1):
    '''
    Generate a non-stationary Gamma process with rate profile sampled from
    the analog-signal rate_signal.
    The process is obtained via thinning of a non-stationary poisson process
    with rate profile rate_signal*shape, where shape is the integer parameter
    od the Gamma


    Parameters
    -----
    rate_signal : neo.core.AnalogSignal, units=Hz
        The analog signal containing the discretization on the time axis of the
        rate profile function of the spike trains to generate

    shape : int
        The parameter of the Gamma distribution

    N : int
        The number of utput processes.
        Default:1

    Output
    -----
    List of N non-stationary neo.SpikeTrain with profile rate
    lambda(t)= rate_signal and Gamma distribuited ISI
    '''
    if type(shape) == int:
        if type(rate_signal) == neo.AnalogSignal:
            #adjustment of the signal to avoid bias given by the first spike
            rate_adj = neo.AnalogSignal(
                np.hstack(
                    (
                        rate_signal[0].magnitude*np.array(10*[1]),
                        rate_signal.magnitude.reshape(len(rate_signal)))),
                units=rate_signal.units,
                sampling_period=rate_signal.sampling_period,
                t_start=rate_signal.t_start -
                10 * rate_signal.sampling_period) * shape
            t_start = [rate_signal.t_start] * N
#        elif all([type(r) == neo.AnalogSignal for r in rate_signal]):
        else:
            rate_adj = [
                neo.AnalogSignal(
                    np.hstack(
                        (
                            r[0].magnitude*np.array(10*[1]), r.magnitude.reshape(len(r)))),
                    units=r.units, sampling_period=r.sampling_period,
                    t_start=r.t_start - 10 * r.sampling_period) * shape
                for r in rate_signal]
            t_start = []
            for r in rate_signal:
                t_start = t_start + [r.t_start] * N
#        else:
#            raise ValueError(
#                'rate_signal has to be an neo.AnalogSiagnal or a list of '
#                'neo.AnalogSiagnal instead is %s' % type(rate_signal))
        #Poisson non-stationary process to be thinned
        poiss = poisson_nonstat(rate_signal=rate_adj, N=N)

        #
        #    #Thinning

        spiketrains = [st[0::shape] for st in poiss]
        spiketrains = [
            st[st > t_start[i]] for i, st in enumerate(spiketrains)]
        for i, st in enumerate(spiketrains):
            st.t_start = t_start[i].rescale(st.t_stop.units)
        return spiketrains

    else:
        raise ValueError(
            'The shape parameter of the gamma distribution has to be an'
            'integer')


def cgp(A, t_stop, shape, rate, t_start=0 * pq.s):
    '''
    Generate a Compound Gamma Process (CGP) with a given amplitude
    distribution A and stationary marginal rates rate.
    The idea is that the independent portion of spikes is generated as
    independent gamma processes, then overlapped with CPPs that inject the
    correlated amount of spikes.

    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of len(A)-1
    spike trains with a correlation structure determined by the amplitude
    distribution A: A[j] is the probability that a spike occurs synchronously
    in any j spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to j of the output spike trains with
    probability A[j].

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    t_stop : Quantity (time)
        The end time of the output spike trains
    shape: positive integer
        It's the positive integer parameter of the gamma distribution of the
        ISIs of independent spikes
    rate : Quantity (1/time)
        Average rate of each spike train generated. Can be:
        * single-valued: if so, all spike trains will have same rate rate
        * a sequence of values (of length len(A)-1), each indicating the
          firing rate of one process in output
    t_start : Quantity (time). Optional, default to 0 s
        The t_start time of the output spike trains

    Returns
    -------
    List of SpikeTrain
        SpikeTrains with specified firing rates forming the CPP with amplitude
        distribution A.
    '''
    if sum(A) != 1 or any([a < 0 for a in A]):
        raise ValueError(
            'A must be a probability vector, sum(A)= %f !=1' % sum(A))
    if A[1] == 0:
        raise ValueError(
            'A_1 must be positive (otherwise use cpp() function)')
    #number of neurons
    N = len(A) - 1

    #expectations of the amplitude
    exp_A = float(np.dot(A, xrange(N + 1)))
    A = np.array(A)

    #rate and amplitude of the constructive N processes (gamma + N-1 CPP)
    rate_marg = A * np.arange(0, N + 1, 1) * (rate / float(exp_A))
    index = np.where(A > 0)[0]
    amplitudes = np.zeros((len(index), N + 1))
    proc_marg = []
    #base gamma independent processes
    proc_marg.append(gamma_thinning(
        t_stop=t_stop, shape=shape, rate=rate_marg[1], N=N, t_start=t_start))
    #CPPs
    for i, j in enumerate(index[1:]):
        amplitudes[i][j] = 1
        proc_marg.append(
            cpp(
                A=amplitudes[i], t_stop=t_stop, rate=rate_marg[j],
                t_start=t_start))
    #pooling of the N processes
    proc_marg = np.array(proc_marg).T
    spiketrains = [_pool_spiketrains(p) for p in proc_marg]
    return spiketrains


def cgp_nonstat(A, shape, rate):
    '''
    Generate a Compound Gamma Process (CGP) with a given amplitude
    distribution A and non-stationary marginal rate profile rate.
    The idea is that the independent portion of spikes is generated as
    independent gamma processes, then overlapped with CPPs that inject the
    correlated amount of spikes.


    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of len(A)-1
    spike trains with a correlation structure determined by the amplitude
    distribution A: A[j] is the probability that a spike occurs synchronously
    in any j spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to j of the output spike trains with
    probability A[j].

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    shape: positive integer
        It's the positive integer parameter of the gamma distribution of the
        ISIs of independent spikes
    rate : neo.AnalogSignal
        Average rate profile of each spike train generated. Can be:

    Returns
    -------
    List of SpikeTrain
        SpikeTrains with specified firing rates forming the CGP with amplitude
        distribution A.
    '''
    if sum(A) != 1 or any([a < 0 for a in A]):
        raise ValueError(
            'A must be a probability vector, sum(A)= %f !=1' % sum(A))
    if A[1] == 0:
        raise ValueError(
            'A_1 must be positive (otherwise use cpp() function)')
    #Number of neurons
    N = len(A) - 1

    #expected value of the amplitude
    exp_A = float(np.dot(A, xrange(N + 1)))
    A = np.array(A)

    #rate and amplitude of the constructive N multiple processes
    #(gamma + N-1 CPP)
    sampling_period = rate.sampling_period
    unit = rate.units
    t_start = rate.t_start
    rate_marg = [
        rate.magnitude * ((a * i) / float(exp_A)) for i, a in enumerate(
            A)]
    rate_marg = [
        neo.AnalogSignal(
            signal=r, sampling_period=sampling_period, units=unit,
            t_start=t_start) for r in rate_marg]
    index = np.where(A > 0)[0]
    amplitudes = np.zeros((len(index), N + 1))
    proc_marg = []

    #Independent gamma processes
    proc_marg.append(
        gamma_nonstat_rate(rate_signal=rate_marg[1], shape=shape, N=N))

    #CPPs
    for i, j in enumerate(index[1:]):
        amplitudes[i][j] = 1
        proc_marg.append(
            _cpp_hom_nonstat(
                A=amplitudes[i], rate_signal=rate_marg[j]))

    #Pool of the N different multile processes
    proc_marg = np.array(proc_marg).T
    spiketrains = [_pool_spiketrains(p) for p in proc_marg]
    return spiketrains
