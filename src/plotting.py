import matplotlib.pyplot as plt
import seaborn as sns

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
