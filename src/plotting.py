import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import data_generation as dgen

from matplotlib import animation


sns.set(color_codes=True)
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
