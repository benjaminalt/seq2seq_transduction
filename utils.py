import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
default_backend = plt.get_backend()

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def plot_loss_history(points, filepath=None, labels=None):
    labels = labels if labels is not None else ["Train"]
    plt.switch_backend('agg')
    num_series = 1 if type(points[0]) == float else len(points[0])
    assert num_series == len(labels)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    if num_series == 1:
        plt.plot(points)
    else:
        for i in range(num_series):
            ax.plot([point[i] for point in points])
    ax.legend(labels)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)


def plot_waves(params, data, reference_data=None, ax=None):
    show = False
    if ax is None:
        show = True
        plt.switch_backend(default_backend)
        fig, ax = plt.subplots()
    for i in range(len(params)):
        param_set = params[i]
        y = data[i]
        ax.plot(np.arange(y.shape[0]), y, label="A: {:.2f}, f: {:.2f}, phi: {:.2f}".format(*param_set))
        if reference_data is not None:
            y_ref = reference_data[i]
            ax.plot(np.arange(y_ref.shape[0]), y_ref, linestyle=":")
    ax.legend()
    if show:
        plt.show()
