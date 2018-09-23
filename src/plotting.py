import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import quantities as pq
import torch

from matplotlib import animation
from utils import convert_to_spiketrains as cts
from wgangp.models.wgan import GoodGenerator

sns.set(color_codes=True, context='paper')
cmap = sns.husl_palette(10, l=.4)


class PlotDistribution(object):
    def __init__(self, data):
        self.data = data

    def dist_plot(self):
        pass


def dist_plot(data, hist=True, kde=True, rug=False, label='spike times'):
    ax = sns.distplot(data, hist=hist, kde=kde, rug=rug, color=cmap[7],
                      label=label)
    ax.set_title('Distribution of firing rate {}/{}'.format(5, 10))
    plt.legend()
    plt.show()


def plot_distributions(session, save=False, **kwargs):
    low_range = kwargs['lower_range']
    up_range = kwargs['upper_range']
    db, p_d, p_g = samples(session, num_points=10000, num_bins=1000, **kwargs)
    db_x = np.linspace(low_range, up_range, len(db))
    # p_x = np.linspace(low_range, up_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    # plt.plot(p_x, pd, label='real data')
    sns.distplot(p_d, label='real data', hist=True, kde=True,
                 rug=False)
    sns.distplot(p_g, bins=20, label='generated data', hist=True, kde=True,
                 rug=False)
    # plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('count')
    plt.ylabel('Non Stationary Poisson Distribution')
    plt.legend()
    if save:
        plt.savefig('fig1.png', format='png')
    plt.show()


def samples(session, num_points=10000, num_bins=100, **kwargs):
    """
    Return a tuple (db, d, g), where db is the current decision
    boundary, d is a sample from the data distribution,
    and g is a list of generated samples.
    """
    low_range = kwargs['lower_range']
    up_range = kwargs['upper_range']
    batch_size = kwargs['batch_size']
    D1 = kwargs['D1']
    G = kwargs['G']
    x = kwargs['x']
    z = kwargs['z']
    data = kwargs['data']
    xs = np.linspace(low_range, up_range, num_points)
    # bins = np.linspace(low_range, up_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(D1, {
            x: np.reshape(
                xs[batch_size * i:batch_size * (i + 1)],
                (batch_size, 1)
            )
        })

    # data distribution
    idx = np.random.randint(low=low_range, high=len(data))
    d = data[idx]
    # d = np.mean(data, axis=0)
    # pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(low_range, up_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(G, {
            z: np.reshape(
                zs[batch_size * i:batch_size * (i + 1)],
                (batch_size, 1)
            )
        })
    # pg, _ = np.histogram(g, bins=bins, density=True)
    return db, d, g


def plot_training_loss(loss_d, loss_g):
    """
    Plots the training loss
    :param loss_d:
    :param loss_g:
    :return:
    """
    plt.plot(1 - np.array(loss_d), label='loss_d')
    plt.plot(loss_g, label='loss_g')
    plt.legend()
    plt.xlabel("iteration step")
    plt.ylabel("loss")
    plt.show()


def save_animation(path, anim_frames, **kwargs):
    lower_range = kwargs['lower_range']
    upper_range = kwargs['upper_range']
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(lower_range, upper_range, len(db))
    p_x = np.linspace(lower_range, upper_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return line_db, line_pd, line_pg, frame_number

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames)))
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return line_db, line_pd, line_pg, frame_number

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, bitrate=1800)
    anim.save(path + 'anim.mp4', writer=writer)


def plot_mean_activity(data_scheme='binned_data',
                       epoch=-1, real_data_num=(-1, -1),
                       sample_num=(-1, -1, -1, -1), path='results.npy',
                       save=False, figname='mean_activity.pdf'):
    """
    Mean activity of the results

    :param data_scheme: string, `binned_data` or `encoded_data`
    :param epoch: int, epoch number to be considered
    :param real_data_num: tuple or list of tuples
        1-dim: sample
        2-dim: Channel (here only one) with binned data of shape 64x64
    :param sample_num: tuple or list of tuples
        1-dim: list containing lists of integer and tuple, integer indicates the
            epoch, the tuple contains the step index and the output data,
            [int, tuple]
        2-dim: tuple of integer and data as torch.FloatTensor, the integer
            indicates the step index of the corresponding batch in the loop
            (int, FloatTensor)
        3-dim: FloatTensor of shape 64x1x64x64:
        4-dim: 64 samples x
            Channel number (here always only 1) x
            64x64 normalized binned data
    :param save, bool, if the results should be saved
    :param path: Path and filename of the results
    :param figname: string, name of the figure to be saved
    """
    if data_scheme == 'binned_data':
        fig, ax, lrs = _plot_mean_activity_binned(epoch, real_data_num,
                                                  sample_num, path)
    elif data_scheme == 'encoded_data':
        fig, ax, lrs = _plot_mean_activity_encoded(epoch, real_data_num,
                                                   sample_num, path)
    else:
        raise ValueError('Unknown data scheme')
    ax.set_xlim(0, lrs)
    ax.set_xlabel('Bins', fontsize=14.)
    ax.set_ylabel('Counts', fontsize=14.)
    ax.set_xticks(range(0, lrs, 5))
    ax.set_title('Averaged activity epoch {}'.format(epoch), fontsize=14.)
    axs = plt.gca()
    max_xtick = max(axs.get_xlim()) - axs.get_xticks()[-1] + axs.get_xticks()[
        -2]
    max_ytick = max(axs.get_ylim())
    fig.text(x=max_xtick, y=max_ytick,
             s="real sample: {0} \n fake sample: {1}".format(real_data_num,
                                                             sample_num),
             ha='left')
    ax.set_ylim(0, 0.5)
    ax.legend()
    if save:
        fig.savefig(figname)
        fig.clf()
    fig.show()


def _plot_mean_activity_binned(epoch=-1, real_data_num=(-1, -1),
                               sample_num=(-1, -1, -1, -1),
                               path='results.npy'):
    if not isinstance(epoch, range):
        epoch = [epoch]
    if not isinstance(sample_num, list):
        sample_num = [sample_num]
    # load all data
    data = np.load(path).item()
    # load real data
    reals = data['binned_data']
    fakes = data['fake_data']
    real_sample = reals[real_data_num[0]][real_data_num[1]]
    real_sample /= real_sample.max()
    fig, ax = plt.subplots()
    for ep in epoch:
        for sn in sample_num:
            # load fake sample
            fake_sample = fakes[ep][sn[0]][sn[1]][sn[2]][sn[3]].numpy()
            ax.plot(fake_sample.mean(axis=0))  # , label='ep{}'.format(ep))
    ax.plot(real_sample.mean(axis=0), label='real{}'.format(real_data_num[0]),
            color='red', lw=3)
    return fig, ax, len(real_sample)


def _plot_mean_activity_encoded(epoch=-1, real_data_num=(-1, -1),
                                sample_num=(-1, -1, -1, -1),
                                path='results.npy'):
    if not isinstance(epoch, range):
        epoch = [epoch]
    if not isinstance(sample_num, list):
        sample_num = [sample_num]
    # load all data
    data = np.load(path).item()
    # load real data
    reals = data['encoded_data']
    fakes = data['fake_data']
    real_sample = reals[real_data_num[0]][real_data_num[1]]
    fig, ax = plt.subplots()
    for ep in epoch:
        for sn in sample_num:
            # load fake sample
            fake_sample = fakes[ep][sn[0]][sn[1]][sn[2]][sn[3]].numpy()
            ax.plot(fake_sample.mean(axis=0))
    ax.plot(real_sample.mean(axis=0), label='real{}'.format(real_data_num[0]),
            color='red', lw=3)
    return fig, ax, len(real_sample)


def plot_mean_histogram(data_scheme='binned_data',
                        epoch=-1, real_data_num=(-1, -1),
                        sample_num=(-1, -1, -1, -1), bins=32, rho=6.0,
                        path='results.npy',
                        hist_kwgs=None, save=False,
                        figname='mean_histogram.pdf',
                        **kwargs):
    """
    Histogram of the averaged results

    :param data_scheme: string, `binned_data` or `encoded_data`
    :param epoch: int, epoch number to be considered
    :param real_data_num: tuple
        1-dim: sample
        2-dim: Channel (here only one) with binned data of shape 64x64
    :param sample_num: tuple
        1-dim: list containing lists of integer and tuple, integer indicates the
            epoch, the tuple contains the step index and the output data,
            [int, tuple]
        2-dim: tuple of integer and data as torch.FloatTensor, the integer
            indicates the step index of the corresponding batch in the loop
            (int, FloatTensor)
        3-dim: FloatTensor of shape 64x1x64x64:
        4-dim: 64 samples x
            Channel number (here always only 1) x
            64x64 normalized binned data
    :param bins: int, number of bins for the histogram
    :param path: Path and filename of the results
    :param figname: string, name of the figure to be saved
    :param hist_kwgs: histogram keywords which are passed on to the histogram
        function used by seaborn's `distplot` (dictionary, optional)
    :param kwargs: key word arguments for seaborn's `distplot`
        (dictionary, optional)
    :param save, bool, if the results should be saved
    """
    if not isinstance(sample_num, list):
        sample_num = [sample_num]
    if data_scheme == 'binned_data':
        distp = _plot_mean_histogram_binned(epoch, real_data_num, sample_num,
                                            bins, path, hist_kwgs, **kwargs)
    elif data_scheme == 'encoded_data':
        distp = _plot_mean_histogram_encoded(epoch, real_data_num[0],
                                             sample_num, bins, path, rho,
                                             hist_kwgs, **kwargs)
    else:
        raise ValueError('Unknown data scheme')
    max_xtick = max(distp.get_xlim()) - distp.get_xticks()[-1] + \
        distp.get_xticks()[-2]
    max_ytick = max(distp.get_ylim())
    plt.title('Distribution of mean activity (epoch:{})'.format(epoch),
              fontsize=14)
    plt.text(x=max_xtick, y=max_ytick,
             s="real sample: {0} \n fake sample: {1}".format(real_data_num,
                                                             sample_num),
             ha='left')
    plt.legend()
    if save:
        plt.savefig(figname)
    plt.show()


def _plot_mean_histogram_binned(epoch, real_data_num, sample_num, bins, path,
                                hist_kwgs, **kwargs):
    # load all data
    data = np.load(path).item()
    # load real data
    reals = data['binned_data']
    fakes = data['fake_data']
    real_sample = reals[real_data_num[0]][real_data_num[1]]
    real_sample /= real_sample.max()
    if epoch == -1:
        epoch = len(fakes) - 1
    for sn in sample_num:
        fake_sample = fakes[epoch][sn[0]][sn[1]][sn[2]][sn[3]].numpy()
        sns.distplot(fake_sample.mean(axis=0).ravel(), label='fake', bins=bins,
                     hist_kws=hist_kwgs, **kwargs)
    distp = sns.distplot(real_sample.mean(axis=0), label='real', bins=bins,
                         color='red', hist_kws=hist_kwgs, **kwargs)
    return distp


def _plot_mean_histogram_encoded(epoch, real_data_num, sample_num, bins, path,
                                 rho, hist_kwgs, **kwargs):
    # load all data
    data = np.load(path).item()
    # load real data
    reals = data['encoded_data']
    fakes = data['fake_data']
    real_sample = reals[real_data_num]
    if epoch == -1:
        epoch = len(fakes) - 1
    for sn in sample_num:
        fake_sample = fakes[epoch][sn[0]][sn[1]][sn[2]][sn[3]].numpy()
        sns.distplot(fake_sample.mean(axis=0).ravel() * rho, label='fake',
                     bins=bins, hist_kws=hist_kwgs, **kwargs)
    distp = sns.distplot(real_sample.mean(axis=0), label='real', bins=bins,
                         color='red', hist_kws=hist_kwgs, **kwargs)
    return distp


def plot_loss(path='results.npy', save=False, figname='loss.pdf',
              y_scale='linear', **kwargs):
    """
    Plot the loss of the discriminator and generator

    :param save: bool, To save a picture. Default is False
    :param figname: string, Figurename to save with extension
    :param y_scale: string, Scale of the y-axis. Default is linear
    :param path: string, Path to the results file
    :param kwargs: dictionary, additionally plot parameter
    """
    # load data
    data = np.load(path).item()
    # get all errors
    err_d = data['errD']
    err_g = data['errG']
    mean_d = []
    mean_g = []
    std_d = []
    std_g = []
    # loop over epochs
    for ep_d, ep_g in zip(err_d, err_g):
        d = np.array(ep_d[1:])[1:, 1]
        g = np.array(ep_g[1:])[1:, 1]
        mean_d.append(d.mean())
        mean_g.append(g.mean())
        std_d.append(d.std())
        std_g.append(g.std())
    p1 = plt.plot(range(len(err_d)), mean_d, label='Discriminator', **kwargs)
    p2 = plt.plot(range(len(err_g)), mean_g, label='Generator', **kwargs)
    mean_d = np.array(mean_d)
    mean_g = np.array(mean_g)
    std_d = np.array(std_d)
    std_g = np.array(std_g)
    plt.fill_between(range(len(err_d)), mean_d + std_d, mean_d - std_d,
                     alpha=0.2, color=p1[0].get_color())
    plt.fill_between(range(len(err_d)), mean_g + std_g, mean_g - std_g,
                     alpha=0.2, color=p2[0].get_color())
    plt.legend()
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.yscale(y_scale)
    plt.xticks(range(len(err_g)))
    plt.xlim(xmax=(len(err_g) - 1))
    if save:
        plt.savefig(figname)
    plt.show()


def plot_all_loss(path='results.npy', save=False, figname='loss.pdf',
                  y_scale='linear', **kwargs):
    """
    Plot the total loss of the discriminator and generator

    :param save: bool, To save a picture. Default is False
    :param figname: string, Figurename to save with extension
    :param y_scale: string, Scale of the y-axis. Default is linear
    :param path: string, Path to the results file
    :param kwargs: dictionary, additionally plot parameter
    """

    # load data
    data = np.load(path).item()
    # get all errors
    err_d = data['errD']
    err_g = data['errG']
    # to plot the mean activity
    mean_d = []
    mean_g = []
    idxs = []
    idx = 0
    # loop over epochs
    for ep_d, ep_g in zip(err_d, err_g):
        d = np.array(ep_d[1:])[1:, 1]
        g = np.array(ep_g[1:])[1:, 1]
        plt.plot(range(idx, idx + len(d)), d, 'r',
                 label='Discriminator' if idx == 0 else "", **kwargs)
        plt.plot(range(idx, idx + len(g)), g, 'b',
                 label='Generator' if idx == 0 else "", **kwargs)
        mean_g.append(g.mean())
        mean_d.append(d.mean())
        idxs.append((idx + len(d) - idx) / 2 + idx)
        idx += len(d)
    plt.plot(idxs, mean_d, 'k')
    plt.plot(idxs, mean_g, 'k')
    plt.yscale(y_scale)
    # set epochs as labels
    plt.xticks(range(0, idx, len(err_d[0])), range(len(err_d)))
    plt.legend()
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    # plt.xlim(xmax=(len(err_g) - 1))
    if save:
        plt.savefig(figname)

    plt.show()


def plot_isi_distribution(data_scheme='encoded_data',
                          epoch=-1, real_data_num=-1,
                          sample_num=(-1, -1, -1, -1), bins=32, rho=6.,
                          path='results.npy', unique=False, hist_kwgs=None,
                          save=False, figname='isi_distribution.pdf', **kwargs):
    """
    Plots the interspike interval distribution

    :param data_scheme: string, `binned_data` or `encoded_data`
    :param epoch: epoch number to be considered
    :param real_data_num:
    :param sample_num: sample_num: tuple or list of tuples
        1-dim: list containing lists of integer and tuple, integer indicates the
            epoch, the tuple contains the step index and the output data,
            [int, tuple]
        2-dim: tuple of integer and data as torch.FloatTensor, the integer
            indicates the step index of the corresponding batch in the loop
            (int, FloatTensor)
        3-dim: FloatTensor of shape 64x1x64x64:
        4-dim: 64 samples x
            Channel number (here always only 1) x
            64x64 normalized binned data
    :param bins: int, number of bins for the histogram
    :param rho: float, Upscaling parameter, Default is 6.0
    :param unique: bool,
    :param path: Path and filename of the results
    :param hist_kwgs: histogram keywords which are passed on to the histogram
          function used by seaborn's `distplot` (dictionary, optional)
    :param save: bool, To save a picture. Default is False
    :param figname: string, Figurename to save with extension
    :param kwargs: dictionary, additionally plot parameter for seaborn's
            `distplot` function (dictionary, optional)
    """
    if not isinstance(sample_num, list):
        sample_num = [sample_num]
    if data_scheme == 'binned_data':
        distp = None
    elif data_scheme == 'encoded_data':
        distp = _plot_isi_distr_encoded(epoch, real_data_num, sample_num, rho,
                                        bins, path, unique,
                                        hist_kwgs, **kwargs)
    else:
        raise ValueError('Unknown data scheme')
    max_xtick = max(distp.get_xlim()) - distp.get_xticks()[-1] + \
        distp.get_xticks()[-2]
    max_ytick = max(distp.get_ylim())
    plt.title('ISI Distribution (epoch:{})'.format(epoch),
              fontsize=14)
    plt.text(x=max_xtick, y=max_ytick,
             s="real sample: {0} \n fake sample: {1}".format(real_data_num,
                                                             sample_num),
             ha='left')
    plt.xlabel('Interspike interval (s)')
    plt.ylabel('Number of intervals per bin')
    plt.legend()
    if save:
        plt.savefig(figname)
    plt.show()


def _plot_isi_distr_encoded(epoch, real_data_num, sample_num, rho,
                            bins, path, unique, hist_kwgs, **kwargs):
    # load all data
    data = np.load(path).item()
    # load real data
    reals = data['encoded_data']
    fakes = data['fake_data']
    real_sample = np.diff(reals[real_data_num].flatten())
    if epoch == -1:
        epoch = len(fakes) - 1

    def uniq(x): return np.unique(np.abs(np.diff(x))) if unique else (
        np.abs(np.diff(x)))

    for sn in sample_num:
        fake_sample = uniq(
            (fakes[epoch][sn[0]][sn[1]][sn[2]][sn[3]].numpy() * rho).flatten())
        sns.distplot(fake_sample, label='fake',
                     bins=bins, hist_kws=hist_kwgs, **kwargs)
    distp = sns.distplot(uniq(real_sample), label='real', bins=bins,
                         color='red', hist_kws=hist_kwgs, **kwargs)
    return distp


def plot_generated_dot_display_joint(fname='', sample_num=(22, 1, 1, 40, 0),
                                     encoding=False, rate=10, rho=6.,
                                     save=False, figname=''):
    """
    Plots the dot display as a scatter plot, the histogram (over the x-axis)
    and the summed activty for each Neuron (on the y-axis)

    :param fname: string, data file name to load
    :param sample_num: tuple
        1-dim: list containing lists of integer and tuple, integer indicates the
            epoch, the tuple contains the step index and the output data,
            [int, tuple]
        2-dim: tuple of integer and data as torch.FloatTensor, the integer
            indicates the step index of the corresponding batch in the loop
            (int, FloatTensor)
        3-dim: FloatTensor of shape 64x1x64x64:
        4-dim: 64 samples x
            Channel number (here always only 1) x
            64x64 normalized data
    :param encoding: bool, specifies which kind of data should be loaded,
        Default is False
    :param rate: int, specifies the rate parameter, used only for the figure
        title, Default is 10
    :param rho: float, Upscaling parameter, Default is 6.0
    :param save: bool, if the figure should be saved
    :param figname: string, name of the figure to save
    """
    print('loading data')
    data = np.load(fname).item()
    print('done loading')
    fakes = data['fake_data']
    sn = sample_num
    if encoding:
        fake = fakes[sn[0]][sn[1]][sn[2]][sn[3]][sn[4]].numpy() * rho
        # fake = fake[:, :40]
        x = []
        for j, i in enumerate(fake):
            st = np.abs(np.unique(i))
            for s in st:
                x.append((s, j))
        df = pd.DataFrame(x, columns=['time [s]', 'Neuron ID'])
        g = sns.JointGrid(x=df['time [s]'], y=df['Neuron ID'])
        g = g.plot_joint(plt.scatter, marker="|")
        # g = g.plot_marginals(sns.distplot)
        # mx = np.mean(fake, axis=0)
        my = np.sum(fake, axis=1) / rho
        # g.ax_marg_x.step(x=np.linspace(0, 6, len(mx)), y=mx)
        g.ax_marg_y.step(my, y=range(len(my)), where='pre', color=cmap[5])
        g.ax_marg_x.hist(df['time [s]'], bins=64,
                         histtype='step', color=cmap[5], lw=1.5)
        g.ax_marg_x.set_title('counts')
        g.ax_marg_y.set_title('rate [Hz]')
        # g.ax_marg_y.barh(range(len(my)), width=my)
        # g.ax_marg_x.fill_between(np.linspace(0, 6, len(mx)), mx, step='pre')
        # g.ax_marg_y.fill_between(y1=range(0, 64), x=my, step='pre')
        g.fig.suptitle('Generated spikes, [5/{}] Hz'.format(rate))
        plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)
        plt.setp(g.ax_marg_y.get_xticklabels(), visible=True)
    else:
        # TODO binsize need to be loaded from the real data,
        # but for speed and memory reasons omitted here
        binsize = 312.5 * pq.ms
        generated = fakes[sn[0]][sn[1]][sn[2]][sn[3]][sn[4]]
        print('Converting')
        # rho needs to be extracted from the binned_data by getting the
        # maximum of counts of the set or the average
        # rho e.g. 16
        sts = cts(generated, binsize, rho=rho)
        [plt.plot(i.magnitude, np.ones_like(i) * j, '.k') for j, i in
         enumerate(sts, 1)]
        plt.xlabel('ms')
    if save:
        plt.savefig(figname)
    plt.show()


def plot_generated_dot_display(fname='', sample_num=(22, 1, 1, 40, 0),
                               encoding=False, rate=10, rho=6.,
                               save=False, figname=''):
    """
    Plots the dot display

    :param fname: string, data file name to load
    :param sample_num: tuple
        1-dim: list containing lists of integer and tuple, integer indicates the
            epoch, the tuple contains the step index and the output data,
            [int, tuple]
        2-dim: tuple of integer and data as torch.FloatTensor, the integer
            indicates the step index of the corresponding batch in the loop
            (int, FloatTensor)
        3-dim: FloatTensor of shape 64x1x64x64:
        4-dim: 64 samples x
            Channel number (here always only 1) x
            64x64 normalized data
    :param encoding: bool, specifies which kind of data should be loaded,
        Default is False
    :param rate: int, specifies the rate parameter, used only for the figure
        title, Default is 10
    :param rho: float, Upscaling parameter, Default is 6.0
    :param save: bool, if the figure should be saved
    :param figname: string, name of the figure to save

    """
    print('loading data')
    data = np.load(fname).item()
    print('done loading')
    fakes = data['fake_data']
    sn = sample_num
    fake = fakes[sn[0]][sn[1]][sn[2]][sn[3]][sn[4]].numpy() * rho
    if encoding:
        for j, i in enumerate(fake):
            y = np.ones_like(np.abs(np.unique(i))) * j
            plt.plot(np.abs(np.unique(i)), y, 'k.')
            plt.xlabel('time [s]')
    else:
        # binsize need to be loaded from the real data, but for speed and
        # memory reasons omitted
        binsize = 312.5 * pq.ms
        generated = fakes[sn[0]][sn[1]][sn[2]][sn[3]][sn[4]]
        print('Converting')
        # rho needs to be extracted from the binned_data by getting the
        # maximum of counts of the set or the average
        sts = cts(generated, binsize, rho=rho)
        [plt.plot(i.magnitude, np.ones_like(i) * j, '.k') for j, i in
         enumerate(sts, 1)]
        plt.xlabel('ms')
    plt.ylabel('Neuron ID')
    plt.title('Generated spikes, [5/{}] Hz'.format(rate))
    if save:
        plt.savefig(figname)
    plt.show()


class GeneratorPlotter(object):
    """
    Reads in a pre-saved generator with file ending '.pt'. Generates data from
    random noise. Provides plotting functionalities.
    """
    def __init__(self, generator_path, map_location="cpu"):
        """

        :param generator_path: string, path to pre-saved generator file
        """
        self.aG = GoodGenerator()
        self.aG.load_state_dict(torch.load(generator_path,
                                           map_location=map_location))
        # alternatively give location
        # self.aG = torch.load(generator_path, map_location=map_location)

    def plot_dot_display_joint(self, shapes=(32, 128),
                               reshapes=(32, 1, 32, 32),
                               sample_num=(-1, 0),
                               rho=6.,
                               save=False, figname=""):
        sn = sample_num
        fakes = self.aG(torch.randn(shapes))
        fakes = fakes.reshape(reshapes)
        fake_data = fakes[sn[0], sn[1]].detach().numpy() * rho
        x = []
        for j, i in enumerate(fake_data):
            for s in i:
                x.append((s, j))
        df = pd.DataFrame(x, columns=['time [s]', 'Neuron ID'])
        g = sns.JointGrid(x=df['time [s]'], y=df['Neuron ID'])
        g = g.plot_joint(plt.scatter, marker="|")
        # g = g.plot_marginals(sns.distplot)
        # mx = np.mean(fake, axis=0)
        my = np.sum(fake_data, axis=1) / rho
        # g.ax_marg_x.step(x=np.linspace(0, 6, len(mx)), y=mx)
        g.ax_marg_y.step(my, y=range(len(my)), where='pre', color=cmap[5])
        g.ax_marg_x.hist(df['time [s]'], bins=32,
                         histtype='step', color=cmap[5], lw=1.5)
        g.ax_marg_x.set_title('counts')
        g.ax_marg_y.set_title('rate [Hz]')
        # g.ax_marg_y.barh(range(len(my)), width=my)
        # g.ax_marg_x.fill_between(np.linspace(0, 6, len(mx)), mx, step='pre')
        # g.ax_marg_y.fill_between(y1=range(0, 64), x=my, step='pre')
        g.fig.suptitle('Generated spikes')
        plt.setp(g.ax_marg_x.get_yticklabels(), visible=True)
        plt.setp(g.ax_marg_y.get_xticklabels(), visible=True)
        if save:
            plt.savefig(figname)
        plt.show()
