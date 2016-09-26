import numpy as np
import scipy.constants as sc
import seaborn as sns
sns.set_style('ticks')

from matplotlib import rcParams
rcParams['axes.labelsize']      = 8
rcParams['xtick.labelsize']     = 7
rcParams['ytick.labelsize']     = 7
rcParams['legend.fontsize']     = 6
rcParams['axes.linewidth']      = 1.25
rcParams['xtick.major.size']    = 2.5
rcParams['xtick.minor.size']    = 1.5
rcParams['xtick.major.width']   = 1.25
rcParams['xtick.minor.width']   = 1.25
rcParams['ytick.major.size']    = 2.5
rcParams['ytick.minor.size']    = 1.5
rcParams['ytick.major.width']   = 1.25
rcParams['ytick.minor.width']   = 1.25
rcParams['text.usetex']         = True
rcParams['xtick.major.pad']     = 6
rcParams['ytick.major.pad']     = 6
rcParams['ytick.direction']     = 'in'
rcParams['xtick.direction']     = 'in'
rcParams['figure.figsize']      = 3.5, 3./sc.golden

# Colours.
snscols = sns.xkcd_palette(["windows blue", "amber", "faded green", 
                            "greyish", "dusty purple", "pale red"])

rainbow = ['#e61b23', '#f8a20e', '#00a650', '#1671c2', '#282b80', 'k']

mutedrwb = ['#1e1a31', '#6c2963', '#b54b76', '#e5697a', '#f98b74', '#ffb568',
            '#ffe293', '#ffffff', '#b8f7f7', '#70d3e4', '#5eadcd', '#548abe',
            '#4f63b1', '#4d4690', '#383353']


# Colourmaps.
whblbk = sns.cubehelix_palette(light=1., dark=0., start=0.1, hue=1.0, 
                               rot=-0.3, as_cmap=True) 
bkblwh = sns.cubehelix_palette(light=1., dark=0., start=0.1, hue=1.0, 
                               rot=-0.3, as_cmap=True, reverse=True) 
bkgrkh = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=1, reverse=1)
khgrbk = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=1)


# Clip the data for plotting.
def clip(arr, maxval=None, minval=None, maskNaN=None, log=False, 
         minNaN=None, maxNaN=None):

    if log:
	    arr = np.log10(arr)

    if minNaN is not None:
    	maskNaN = minNaN
    	minval = minNaN
    elif maxNaN is not None:
    	maskNaN = maxNaN
    	maxval = maxNaN

    if maskNaN is not None:
        clipped = np.where(np.isnan(arr), maskNaN, arr)
    else:
        clipped = arr
 
    if maxval is not None:
        clipped = np.where(clipped > maxval, maxval, clipped)
    if minval is not None:
        clipped = np.where(clipped < minval, minval, clipped)
    return clipped


# Common functional forms.
def powerLaw(x, a, b, p=100.):
    return a*np.power(x/p, b)

def gaussian(x, center, width, amp):
    return amp * np.exp(-0.5*((x - center) /width)**2.)
    
def normalisedgaussian(x, center, width):
    return gaussian(x, center, width, 1./width/np.sqrt(2.*np.pi))




