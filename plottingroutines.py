from matplotlib import rcParams
import numpy as np
import scipy.constants as sc
import seaborn as sns


'''
rcParams for consisten plots.
'''


sns.set_style('ticks')
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 6
rcParams['axes.linewidth'] = 1.25
rcParams['xtick.major.size'] = 2.5
rcParams['xtick.minor.size'] = 1.5
rcParams['xtick.major.width'] = 1.25
rcParams['xtick.minor.width'] = 1.25
rcParams['ytick.major.size'] = 2.5
rcParams['ytick.minor.size'] = 1.5
rcParams['ytick.major.width'] = 1.25
rcParams['ytick.minor.width'] = 1.25
rcParams['text.usetex'] = True
rcParams['xtick.major.pad'] = 6
rcParams['ytick.major.pad'] = 6
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['figure.figsize'] = 3.5, 3./sc.golden
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


'''
Colour maps and palettes with nicer colours.
'''


snscols = sns.xkcd_palette(["windows blue", "amber", "faded green",
                            "greyish", "dusty purple", "pale red"])
rainbows = ['#e61b23', '#f8a20e', '#00a650', '#1671c2', '#282b80', '#000000']
mutedrwb = ['#1e1a31', '#6c2963', '#b54b76', '#e5697a', '#f98b74', '#ffb568',
            '#ffe293', '#ffffff', '#b8f7f7', '#70d3e4', '#5eadcd', '#548abe',
            '#4f63b1', '#4d4690', '#383353']
whblbk = sns.cubehelix_palette(light=1., dark=0., start=0.1, hue=1.0,
                               rot=-0.3, as_cmap=True)
bkblwh = sns.cubehelix_palette(light=1., dark=0., start=0.1, hue=1.0,
                               rot=-0.3, as_cmap=True, reverse=True)
bkgrkh = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=1, reverse=1)
khgrbk = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=1)


'''
Commonly used functions useful for plotting.
  clip         - manipulate an array for nicer contour plots.
  running_mean - calculate the running mean of an array.
'''


def clip(arr_in, **kwargs):
    """Return an array suitable for plotting with contours."""

    arr = arr_in.copy()

    if kwargs.get('log', False):
        arr = np.log10(arr)
    elif kwargs.get('ln', False):
        arr = np.log(arr)

    minval = kwargs.get('minval', False)
    minNaN = kwargs.get('minNaN', False)
    maxval = kwargs.get('maxval', False)
    maxNaN = kwargs.get('maxNaN', False)
    fillNaN = kwargs.get('fillNaN', False)

    if minNaN and minval:
        raise ValueError('Only specify one min fill value.')
    if minNaN and minval:
        raise ValueError('Only specify one max fill value.')
    if (minNaN and fillNaN) or (maxNaN and fillNaN) or (minNaN and maxNaN):
        raise ValueError('Only specify one NaN fill value.')

    if minval:
        arr = np.where(arr >= minval, arr, minval)
    if minNaN:
        arr = np.where(arr >= minNaN, arr, minNaN)
        arr = np.where(np.isfinite(arr), arr, minNaN)
    if maxval:
        arr = np.where(arr <= maxval, arr, maxval)
    if maxNaN:
        arr = np.where(arr >= maxNaN, arr, maxNaN)
        arr = np.where(np.isfinite(arr), arr, maxNaN)
    if fillNaN:
        arr = np.where(np.isfinite(arr), arr, fillNaN)

    return arr


def running_mean(arr, ncells=2):
    """Returns the running mean of 'arr' over 'ncells' number of cells."""
    if type(arr) != np.ndarray:
        arr = np.array(arr)
    cum_sum = np.insert(np.insert(arr, 0, arr[0]), -1, arr[-1])
    cum_sum = np.cumsum(cum_sum)
    return (cum_sum[ncells:] - cum_sum[:-ncells]) / ncells


'''
Common functional forms.
'''


def powerlaw(x, a, b, p=100.):
    return a * np.power(x / p, b)


def gaussian(x, x0, dx, a):
    return a * np.exp(-0.5 * np.power((x - x0) / dx, 2.))


def norm_gaussian(x, x0, dx):
    return gaussian(x, x0, dx, 1./(dx * np.sqrt(2.*np.pi)))


def normalisedgaussian(x, center, width):
    return gaussian(x, center, width, 1./width/np.sqrt(2.*np.pi))
