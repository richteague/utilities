import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def gaussian(x, dx, A=1.0, x0=0.0, offset=0.0):
    """Gaussian function with standard deviation dx."""
    return A * np.exp(-0.5 * np.power((x-x0)/dx, 2.)) + offset


def gradient_between(x, y, dy, ax=None, **kwargs):
    """Similar to fill_between but with a gradient."""

    if ax is None:
        fig, ax = plt.subplots()

    # If dy has a shape (x.size, 2) or (2, x.size) then we assume that these
    # are percentiles and y is the median. If so, we can convert these to
    # uncertainties for the plotting. Otherwise we just double up the
    # uncertainties.

    try:
        ndim = dy.ndim
    except AttributeError:
        dy = np.squeeze(dy)
        ndim = dy.ndim

    if ndim == 1:
        dy = np.array([dy, dy])
    elif ndim == 2:
        if dy.shape[1] == 2:
            dy = dy.T
        if kwargs.get('percentiles', False):
            dy = np.squeeze([y - dy[0], dy[1] - y])
    else:
        raise TypeError("'dy' must be only 1D or 2D.")
    if not (dy.shape[0] == 2 and dy.shape[1] == x.size):
        raise ValueError()

    # Populate the colours and styles of the plotting.
    # Colors, can set each individually but 'color' will override all.

    color = kwargs.pop('color', None)
    if color is not None:
        fc = color
        lc = color
        ec = [color]
    else:
        fc = kwargs.get('gradcolor', 'dodgerblue')
        lc = kwargs.get('linecolor', 'dodgerblue')
        ec = np.array([kwargs.get('edgecolor', 'k')]).flatten()

    # Controlls the gradient fill. The gradient will run from an alpha of
    # alphamax at the median to ~0 at y +/- nsigma * dy. This will be built
    # from nfill fill_between calls.

    am = kwargs.get('alphamax', .7)
    ns = kwargs.get('nsigma', 1)
    fn = kwargs.get('nfills', kwargs.get('nfill', 35))

    # Styles for the percentiles. Will cycle through the values if given in a
    # list. Should not complain if the lists are different lenghts.

    lw = kwargs.get('linewidth', 1.25)
    ms = kwargs.get('markersize', 3)
    ed = np.array([kwargs.get('edges', [1])]).flatten()
    ea = np.array([kwargs.get('edgealpha', [0.5, 0.25])]).flatten()
    es = np.array([kwargs.get('edgestyle', ':')]).flatten()

    # Incrementally calculate the alpha for a given layer and plot it.

    alphacum = 0.0
    fy = np.linspace(0., ns, fn)
    for n in fy[::-1]:
        alpha = np.mean(gaussian(n, ns / 3., am)) - alphacum
        ax.fill_between(x, y-n*dy[0], y+n*dy[1], facecolor=fc, lw=0,
                        alpha=alpha)
        alphacum += alpha

    # Properties for the edges. These are able to be interated over.

    for e, edge in enumerate(ed):
        ax.plot(x, y-edge*dy[0], alpha=ea[e % len(ea)], color=ec[e % len(ec)],
                linestyle=es[e % len(es)])
        ax.plot(x, y+edge*dy[1], alpha=ea[e % len(ea)], color=ec[e % len(ec)],
                linestyle=es[e % len(es)])

    # Mean value including the label. Note that we do not call the legend here
    # so extra legend kwargs can be used if necessary. If 'outline' is true,
    # include a slightly thicker line in black.

    if kwargs.get('outline', True):
        ax.errorbar(x, y, color='k', fmt=kwargs.get('fmt', '-o'),
                    ms=ms, mew=1, lw=lw*2, zorder=5)

    """
        ax.errorbar(x, y, color=lc, fmt=kwargs.get('fmt', '-o'), zorder=5,
                    ms=ms, mew=0, lw=lw, label=kwargs.get('label', None),
                    path_effects=[pe.Stroke(linewidth=lw*2, foreground='k'),
                    pe.Normal()])
    """

    ax.errorbar(x, y, color=lc, fmt=kwargs.get('fmt', '-o'),
                ms=ms, mew=0, lw=lw, label=kwargs.get('label', None), zorder=5)

    return ax


def gradient_fill(x, y, dy, region='below', ax=None, **kwargs):
    """Fill above or below a line with a gradiated fill."""
    if ax is None:
        fig, ax = plt.subplots()
    if region == 'below':
        ax = gradient_between(x, y, [dy, np.zeros(x.size)], ax=ax, **kwargs)
    elif region == 'above':
        ax = gradient_between(x, y, [np.zeros(x.size), dy], ax=ax, **kwargs)
    else:
        raise ValueError("Must set 'region' to 'above' or 'below'.")
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


def plotbeam(bmaj, bmin=None, bpa=0.0, ax=None, **kwargs):
    """Plot a beam. Input must be same units as axes. PA in degrees E of N."""
    if ax is None:
        fig, ax = plt.subplots()
    if bmin is None:
        bmin = bmaj
    if bmin > bmaj:
        temp = bmin
        bmin = bmaj
        bmaj = temp
    offset = kwargs.get('offset', 0.125)
    ax.add_patch(Ellipse(ax.transLimits.inverted().transform((offset, offset)),
                         width=bmin, height=bmaj, angle=bpa,
                         fill=False, hatch=kwargs.get('hatch', '////////'),
                         lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                         color=kwargs.get('color', kwargs.get('c', 'k'))))
    return
