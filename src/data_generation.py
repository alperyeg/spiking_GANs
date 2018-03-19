import numpy as np
import neo

from scipy.misc import factorial
from quantities import Hz, s, ms
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.conversion import BinnedSpikeTrain
from stocmod import poisson_nonstat
from STP_generation import generate_sts

np.random.seed(123)


class DataDistribution(object):

    def __init__(self):
        """
       Real data distribution, a simple Gaussian with mean 4 and standard
       deviation of 0.5. It has a sample function that returns a given
       number of samples (sorted by value) from the distribution.
        """
        # Normal sample parameter
        self.mu = 4
        self.sigma = 0.5
        # Poisson sample parameter
        self.lam = 10
        # TODO set variable parameter
        self.lambdas = np.linspace(0, 20, 100)

    def normal_sample(self, n):
        """

        :param n: int, size of sample
        :return: samples from the normal distribution
        """
        samples = np.random.normal(loc=self.mu, scale=self.sigma, size=n)
        samples.sort()
        return samples

    def poisson_pmf(self, n):
        """

        :param n: size of the array for the poisson distribution, int
        :return: a poisson distribution realized by a random lambda
        """
        def _poisson_sample(x, lam):
            pmf = np.exp(-lam) * lam ** x * factorial(x) ** - 1
            pmf[pmf < 0] = 0
            return pmf
        xs = np.arange(n)
        rand_lam = self.lambdas[np.random.randint(100)]
        p = _poisson_sample(xs, rand_lam)
        # TODO: Sort for stratified sampling? for that set fixed lambda and make array variable
        return p

    def poisson_sample(self, n):
        """

        :param n: int, size of samples
        :return: samples from the parametrized Poisson distribution
        """
        samples = np.random.poisson(self.lam, size=n)
        samples.sort()
        return samples

    @staticmethod
    def poisson_step_rate(t_start=0 * s, t_stop=100 * s, rate1=5 * Hz,
                          rate2=10 * Hz, n_samples=1000, min_len=1000):
        """
        Returns a poisson process with step rate given by `rate1` and `rate2`
        """
        # TODO: give different start stop for 2nd spiketrain
        minimum = np.inf
        sts = []
        for i in range(n_samples):
            spikes1 = homogeneous_poisson_process(rate1, t_start, t_stop)
            s1_start, s1_stop = spikes1.t_start, spikes1.t_stop
            spikes2 = homogeneous_poisson_process(rate2, t_start=spikes1[-1],
                                                  t_stop=2 * s1_stop,
                                                  as_array=True)
            s_concat = np.concatenate((spikes1.magnitude, spikes2))
            sts.append(s_concat)
            if len(s_concat) < minimum:
                minimum = len(s_concat)
        sts = np.array([st[:minimum] for st in sts])
        return sts

    @staticmethod
    def poisson_nonstat_sample(rate1=5 * Hz, rate2=10 * Hz, dt=1 * ms,
                               t_stop=1000 * ms, binned=True, num_bins=100, num_sts=1):
        """
        Returns a non-stationary poisson process with step rate given by
        `rate1` and `rate2`

        :param rate1: pq.Quantity First rate, e.g. in Hz
        :param rate2: pq.Quantity Second rate, e.g. in Hz
        :param dt: pq.Quantity Sampling period
        :param t_stop: pq.Quantity End time of the first spike train
        :param binned: bool If the spike trains should be binned
        :param num_bins: int Number of bins
        :param num_sts: int Number of spike trains
        :return: if `binned` is **True** returns binned spiketrains,
            corresponding spikes, and the rate signal, if `binned` is
            **False** returns only spikes
        """
        t1 = 2 * t_stop
        rate_profile = [rate1 for _ in range(int(t_stop / dt))] + [rate2 for _ in range(
            int((t1 - t_stop) / dt))]
        rate_signal = neo.AnalogSignal(signal=rate_profile, units=Hz,
                                       sampling_period=dt)
        spikes = poisson_nonstat(rate_signal, N=num_sts)
        if binned:
            binned_sts = BinnedSpikeTrain(spikes, num_bins=num_bins)
            return binned_sts, spikes, rate_signal
        return spikes

    @staticmethod
    def gen_nonstat_sample(data_type=6, t=10000 * ms, sample_period=10 * ms,
                           num_bins=100, num_sts=1, binned=True):
        """
        :param data_type: int, An integer specifying the type of
            background activity
        :param t: quantity.Quantity, Simulation time. Default is 1000 * pq.ms
        :param sample_period: quantity.Quantity, Sampling period of the
            rate profile. Default is 10 * pq.ms
        :param num_bins: int, Number of bins
        :param num_sts: int, Number of spike trains
        :param binned: bool, If the spike trains should be binned
        :return: binned spiketrains and corresponding spiketrains
        """
        sts = generate_sts(data_type, T=t, sampl_period=sample_period,
                           N=num_sts)[0]
        if binned:
            binned_sts = BinnedSpikeTrain(sts, num_bins=num_bins)
            return binned_sts, sts
        return sts


class GeneratorDistribution(object):

    def __init__(self, lower, upper):
        """
         Generator input noise distribution (with a similar sample function).
         A stratified sampling approach for the generator input noise - the
         samples are first generated uniformly over a specified range,
         and then randomly perturbed.
        """
        self.lower_range = lower
        self.upper_range = upper

    def sample(self, n):
        return np.linspace(self.lower_range, self.upper_range,
                           n) + np.random.random(n) * 0.01

    def sample_int(self, n):
        return np.linspace(self.lower_range, self.upper_range, n) + \
               np.random.randint(-self.upper_range/2, self.upper_range/2, n)

    def binned_samples(self, shape):
        def samples(shape):
            return np.linspace(self.lower_range, self.upper_range,
                               shape[0]) + \
                   np.random.randint(self.lower_range,
                                     self.upper_range / 2, shape[1])

        data = np.zeros(shape)
        for i in range(shape[0]):
            data[i] = np.histogram(samples(shape), bins=shape[1])[0]
        return data
