import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, dx, A=1.0, x0=0.0, offset=0.0):
    """Gaussian function with standard deviation dx."""
    return A * np.exp(-0.5 * np.power((x-x0)/dx, 2.)) + offset


def gradient_between(x, y, dy, ax=None, **kwargs):
    """Similar to fill_between but with a gradient."""
    if ax is None:
        fig, ax = plt.subplots()

    # Gradiated fill.
    fc = kwargs.get('gradcolor', 'dodgerblue')
    am = kwargs.get('alphamax', .7)
    ns = kwargs.get('nsigma', 3)
    fn = kwargs.get('nfills', 15)
    fy = np.linspace(0., ns, fn)

    # Calculate the appropriate alpha for the layer.
    alphacum = 0.0
    for n in fy[::-1]:
        alpha = np.mean(gaussian(n, ns / 3., am)) - alphacum
        ax.fill_between(x, y-n*dy, y+n*dy, facecolor=fc, lw=0, alpha=alpha)
        alphacum += alpha

    # Properties for the edges. These are able to be interated over.
    ed = np.array([kwargs.get('edges', [1, 3])]).flatten()
    ea = np.array([kwargs.get('edgealpha', [0.5, 0.25])]).flatten()
    es = np.array([kwargs.get('edgestyle', ':')]).flatten()
    ec = np.array([kwargs.get('edgecolor', 'k')]).flatten()
    for e, edge in enumerate(ed):
        ax.plot(x, y-edge*dy, alpha=ea[e % len(ea)], color=ec[e % len(ec)],
                linestyle=es[e % len(es)])
        ax.plot(x, y+edge*dy, alpha=ea[e % len(ea)], color=ec[e % len(ec)],
                linestyle=es[e % len(es)])

    # Mean value including the label. Note that we do not call the legend here
    # so extra legend kwargs can be used if necessary.
    lc = kwargs.get('linecolor', 'k')
    ax.plot(x, y, color=lc, label=kwargs.get('label', None))

    return ax


def gradient_fill(x, y, dy, region='below', ax=None, **kwargs):
    """Fill above or below a line with a gradiated fill."""
    if ax is None:
        fig, ax = plt.subplots()
    if region not in ['above', 'below']:
        raise ValueError("Must set 'region' to 'above' or 'below'.")
    ax = gradient_between(x, y, dy, ax=ax, **kwargs)
    fc = kwargs.get('facecolor', ax.get_facecolor())
    if region == 'above':
        ax.fill_between(x, y, ax.get_ylim()[1], facecolor=fc, lw=0)
    else:
        ax.fill_between(x, ax.get_ylim()[1], y, facecolor=fc, lw=0)
    lc = kwargs.get('linecolor', 'k')
    ax.plot(x, y, color=lc)
    return ax


def running_mean(arr, ncells=2):
    """Returns the running mean of 'arr' over 'ncells' number of cells."""
    if type(arr) != np.ndarray:
        arr = np.array(arr)
    cum_sum = np.insert(np.insert(arr, 0, arr[0]), -1, arr[-1])
    cum_sum = np.cumsum(cum_sum)
    return (cum_sum[ncells:] - cum_sum[:-ncells]) / ncells
