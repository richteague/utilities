'''
Simple class for beams.
'''

import numpy as np
from astropy.io.fits import getval
from matplotlib.patches import Ellipse


class synthbeam:

    # Assume all input is in ["] and [rad]. If values are given in __init__
    # through the params list, use these first, otherwise try and find values
    # in the header. Finally, if no values are specified, just use NaNs.

    def __init__(self, path, beamparams=None, dist=1., **kwargs):
        """Initialise an instance of synthbeam."""

        self.verbose = kwargs.get('verbose', True)
        self.checkbeam = False

        if beamparams is None:
            try:
                self.min = getval(path, 'bmin', 0)
                self.maj = getval(path, 'bmaj', 0)
                self.pa = np.radians(getval(path, 'bpa', 0))
            except:
                self.min, self.maj, self.pa = np.nan, np.nan, np.nan
        else:
            self.min = beamparams[0]
            self.maj = beamparams[1]
            self.pa = -1. * np.radians(beamparams[2])
            if self.min > self.maj:
                temp = self.min
                self.min = self.maj
                self.maj = temp

        if all(np.isnan([self.min, self.maj, self.pa])):
            self.checkbeam = True
            if self.verbose:
                print('No beam parameters found. Convolution impossible.')
            return

        # Pixel scaling. If not distance is given, assume in ["].

        self.dist = dist
        self.min *= self.dist
        self.maj *= self.dist

        # Derived properties.

        self.fwhm = 2. / np.sqrt(2. * np.log(2.))
        self.dpix = abs(getval(path, 'cdelt1', 0) * 3600.) * self.dist
        self.area = np.pi * self.min * self.maj / 4. / np.log(2)
        self.eff = np.hypot(self.min, self.maj)
        self.kernel = self.kernel_2D()

        return

    def kernel_2D(self):
        """Generate a 2D beam kernel for the convolution."""
        sig_x = self.maj / self.dpix / self.fwhm
        sig_y = self.min / self.dpix / self.fwhm
        grid = np.arange(-np.round(sig_x) * 8, np.round(sig_x) * 8 + 1)
        a = np.cos(self.pa)**2 / 2. / sig_x**2
        a += np.sin(self.pa)**2 / 2. / sig_y**2
        b = np.sin(2. * self.pa) / 4. / sig_y**2
        b -= np.sin(2. * self.pa) / 4. / sig_x**2
        c = np.sin(self.pa)**2 / 2. / sig_x**2
        c += np.cos(self.pa)**2 / 2. / sig_y**2
        kernel = c * grid[:, None]**2 + a * grid[None, :]**2
        kernel += 2 * b * grid[:, None] * grid[None, :]
        return np.exp(-kernel) / 2. / np.pi / sig_x / sig_y

    def plotbeam(self, ax, x=0.1, y=0.1, **kwargs):
        if self.checkbeam:
            return
        """Plot the beam onto axis."""
        lw = kwargs.get('lw', 1.25)
        hatch = kwargs.get('hatch', '///////')
        color = kwargs.get('c', 'k')
        x_pos = x * (ax.get_xlim()[1] - ax.get_xlim()[0]) + ax.get_xlim()[0]
        y_pos = y * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
        ax.add_patch(Ellipse((x_pos, y_pos), width=self.min,
                             height=self.maj, angle=np.degrees(self.pa),
                             fill=False, hatch=hatch, lw=lw,
                             color=color, transform=ax.transData))
        return
