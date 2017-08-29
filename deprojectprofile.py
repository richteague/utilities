"""
Function to use the vertical asymmeteric in an image to infer the vertical
extent of the disk.

    -- Input --

posax:                  Declination axis, if dist is provided, assume in
                        [arcsec] otherwise assume in [au].
image:                  Either the zeroth moment map or a channel map. The
                        profile will be taken from the central column (or
                        central two if there is an even number of pixels).
inc:                    Inclination of the disk in [rad].
dist [optional]:        Distance to the source to convert from ["] to [au].
minfrac [optional]:     Minimum fraction of the peak value to apply the fitting
                        to. This should be around the noise level of the image.
nsamples [optional]:    Number of sampling points.
logsamples [optional]:  Whether the sampling should be logarithmic.
plot [optional]:        If true, plot the profiles and mark the regions which
                        have been masked.
getprofile [optional]:  If true, returns also the flux density profile used
                        with the mask array.

    -- Returns --

rvals:                  Radial coordinates in [au].
zvals:                  Height coordinates in [au].

"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def deprojectprofile(posax, image, inc, dist=None, minfrac=0.2, nsamples=50,
                     plot=True, getprofile=False, logsamples=False):

    # Convert to absolute units if necessary.
    if dist is not None:
        posax *= dist

    # Find the central pixel(s).
    cidx = image.shape[0]
    if cidx % 2:
        odd = True
    else:
        odd = False
    cidx = int((cidx - 1) / 2)

    # Extract the emission profile and find the masked regions.
    if odd:
        prof = image[:, cidx]
    else:
        prof = np.average(image[:, cidx:cidx+2], axis=1)
    pmask, nmask = posax > 0, posax < 0
    Tbmax = min(max(prof[pmask]), max(prof[nmask]))
    Tbmin = minfrac * Tbmax

    for i in range(posax.size-1):
        if (prof[i+1] - Tbmax) * (prof[i] - Tbmax) < 0:
            nidx = i
            break

    for i in range(posax.size):
        if (prof[-i] - Tbmax) * (prof[-i-1] - Tbmax) < 0:
            pidx = i
            break

    # Plot the profiles if necessary.
    if plot:
        fig, ax = plt.subplots()
        ax.step(posax, prof, c='orangered', lw=1.5, zorder=-10)
        ax.set_xlabel(r'${\rm Offset \quad (au)}$')
        ax.set_ylabel(r'${\rm Flux \;\; Density}$')
        ax.axhline(Tbmax, c='k', ls=':')
        ax.text(ax.get_xlim()[1]*1.05, Tbmax, r'$T_B^{\rm max}$',
                va='center', ha='left', fontsize=6)
        ax.axhline(Tbmin, c='k', ls=':')
        ax.text(ax.get_xlim()[1]*1.05, Tbmin, r'$T_B^{\rm min}$',
                va='center', ha='left', fontsize=6)
        ymax = ax.get_ylim()[1]
        ax.fill_between([posax[nidx], posax[pidx]], [0, 0], [ymax, ymax],
                        hatch='////', facecolor='none')
        if logsamples:
            ax.set_yscale('log')
            ax.set_ylim(1, ymax)
        else:
            ax.set_ylim(0.0, ymax)

    # Find the minimum maximum value and mask values above this.
    pmask, nmask = posax > 0, posax < 0
    pos = interp1d(prof[pidx:], abs(posax[pidx:]), bounds_error=False)
    neg = interp1d(prof[:nidx], abs(posax[:nidx]), bounds_error=False)

    if logsamples:
        Tbint = np.logspace(np.log10(Tbmax), np.log10(Tbmin), nsamples)
    else:
        Tbint = np.linspace(Tbmax, Tbmin, nsamples)
    rvals = np.array([rdeproject(I, pos, neg, inc) for I in Tbint])
    zvals = np.array([zdeproject(I, pos, neg, inc) for I in Tbint])

    # Return the requested values.
    if getprofile:
        return rvals, zvals, prof
    return rvals, zvals


def rdeproject(I, pos, neg, inc):
    """Returns the deprojected radial value."""
    return 0.5 * (pos(I) + neg(I)) / np.cos(inc)


def zdeproject(I, pos, neg, inc):
    """Returns the deprojected height value."""
    p, n = pos(I), neg(I)
    return (0.5 * (p + n) - min(p, n)) / np.sin(inc)
