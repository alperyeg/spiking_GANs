import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import animation

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


def plot_mean_activity(epoch=-1, real_data_num=(-1, -1),
                       sample_num=(-1, -1, -1, -1), path='results.npy',
                       save=False, figname='mean_activity.pdf'):
    """
    Mean activity of the results

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

    :param path: Path and filename of the results
    """
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
    for ep in epoch:
        for sn in sample_num:
            # load fake sample
            fake_sample = fakes[ep][sn[0]][sn[1]][sn[2]][sn[3]].numpy()
            plt.plot(fake_sample.mean(axis=0))#, label='ep{}'.format(ep))
    plt.plot(real_sample.mean(axis=0), label='real{}'.format(real_data_num[0]), color='red',
             lw=3)
    plt.xlim(0, len(real_sample))
    plt.xlabel('Bins', fontsize=14.)
    plt.ylabel('Counts', fontsize=14.)
    plt.xticks(range(0, len(real_sample), 5))
    plt.title('Averaged activity epoch {}'.format(ep), fontsize=14.)
    ax = plt.gca()
    max_xtick = max(ax.get_xlim()) - ax.get_xticks()[-1] + ax.get_xticks()[-2]
    max_ytick = max(ax.get_ylim())
    plt.text(x=max_xtick, y=max_ytick,
             s="real sample: {0} \n fake sample: {1}".format(real_data_num,
                                                             sample_num),
             ha='left')
    plt.ylim(0, 0.5)
    plt.legend()
    if save:
        plt.savefig(figname)
        plt.close()
    plt.show()


def plot_mean_histogram(epoch=-1, real_data_num=(-1, -1),
                        sample_num=(-1, -1, -1, -1), bins=32,
                        path='results.npy',
                        hist_kwgs=None, save=False,
                        figname='mean_histogram.pdf',
                        **kwargs):
    """
    Histogram of the averaged results


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
    :param hist_kwgs: histogram keywords which are passed on to the histogram
        function used by s seaborns `distplot` (dictionary, optional)
    :param kwargs: key word arguments for seaborns `distplot`
        (dictionary, optional)
    """
    if not isinstance(sample_num, list):
        sample_num = [sample_num]
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
