import numpy as np
from astropy.io import fits
import scipy.constants as sc
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Kernel
import warnings
warnings.filterwarnings("ignore")


class imagecube:

    msun = 1.989e30
    fwhm = 2. * np.sqrt(2. * np.log(2))

    def __init__(self, path, offset=[0.0, 0.0], dist=59.):
        """Read in a CASA produced image."""
        self.path = path
        self.data = np.squeeze(fits.getdata(path))
        self.header = fits.getheader(path)
        self.velax = self.readvelocityaxis(path)
        self.chan = np.mean(np.diff(self.velax))
        self.xaxis = self.readpositionaxis(path, 1)
        self.yaxis = self.readpositionaxis(path, 2)
        self.nxpix = int(self.xaxis.size)
        self.nypix = int(self.yaxis.size)
        self.dpix = np.mean([abs(np.mean(np.diff(self.xaxis))),
                             abs(np.mean(np.diff(self.yaxis)))])
        self.dist = 59.
        return

    def _spectralaxis(self, fn):
        """Returns the spectral axis in [Hz]."""
        a_len = fits.getval(fn, 'naxis3')
        a_del = fits.getval(fn, 'cdelt3')
        a_pix = fits.getval(fn, 'crpix3')
        a_ref = fits.getval(fn, 'crval3')
        return a_ref + (np.arange(a_len) - a_pix + 0.5) * a_del

    def writemask(self, name=None, **kwargs):
        """
        Write a .fits file of the mask.
        """
        mask = self._mask(**kwargs)
        kern = self._beamkernel(**kwargs)
        if kwargs.get('fast', True):
            mask = np.array([convolve_fft(c, kern) for c in mask])
        else:
            mask = np.array([convolve(c, kern) for c in mask])
        mask = np.where(mask > 1e-2, 1, 0)

        # Replace the data, swapping axes as appropriate.
        # I'm not sure why this works but it does...
        hdu = fits.open(self.path)
        hdu[0].data = np.swapaxes(mask, 1, 2)
        if name is None:
            name = self.path.replace('.fits', '.mask.fits')
        hdu.writeto(name.replace('.fits', '') + '.fits',
                    overwrite=True, output_verify='fix')
        if kwargs.get('return', False):
            return mask

    def _beamkernel(self, **kwargs):
        """
        Returns the 2D Gaussian kernel.
        """
        bmaj = self.header['bmaj'] * 3600.
        bmin = self.header['bmin'] * 3600.
        bmaj /= self.dpix * self.fwhm
        bmin /= self.dpix * self.fwhm
        bpa = np.radians(self.header['bpa'])
        if kwargs.get('nbeams', 1.0) > 1.0:
            bmin *= kwargs.get('nbeams', 1.0)
            bmaj *= kwargs.get('nbeams', 1.0)
        return Kernel(self.gaussian2D(bmin, bmaj, pa=bpa))

    def _mask(self, **kwargs):
        """
        Returns the Keplerian mask.
        """
        rout = kwargs.get('rout', 4.) * sc.au * self.dist
        inc = kwargs.get('inc', 6.)
        mstar = kwargs.get('mstar', 0.7) * self.msun
        vlsr = kwargs.get('vlsr', 2.89) * 1e3
        dV = 0.5 * (kwargs.get('dV', 300.) + self.chan)

        # Deproject the on-sky coordinates.
        rsky, tsky = self._deproject(**kwargs)
        rsky *= self.dist * sc.au
        rsky = rsky[None, :, :] * np.ones(self.data.shape)
        tsky = tsky[None, :, :] * np.ones(self.data.shape)

        # Calculate the projected Keplerian velocities.
        vkep = np.sqrt(sc.G * mstar / rsky)
        vkep *= np.sin(np.radians(inc)) * np.cos(tsky)

        vdat = self.velax - vlsr
        vdat = vdat[:, None, None] * np.ones(self.data.shape)
        vdat = np.where(rsky <= rout, vdat, 1e10)
        return np.where(abs(vkep - vdat) <= dV, 1, 0)

    def _deproject(self, **kwargs):
        """
        Returns the deprojected pixel values, (r, theta).
        """
        inc = kwargs.get('inc', 0.0)
        pa = kwargs.get('pa', 0.0)
        dx = kwargs.get('dx', 0.0)
        dy = kwargs.get('dy', 0.0)
        x_sky = self.xaxis[None, :] * np.ones(self.nypix)[:, None] - dx
        y_sky = self.yaxis[:, None] * np.ones(self.nxpix)[None, :] - dy
        x_rot, y_rot = self.rotate(x_sky, y_sky, np.radians(pa))
        x_dep, y_dep = self.incline(x_rot, y_rot, np.radians(inc))
        return np.hypot(x_dep, y_dep), np.arctan2(y_dep, x_dep)

    def _velocityaxis(self, fn):
        """
        Return velocity axis in [km/s].
        """
        a_len = fits.getval(fn, 'naxis3')
        a_del = fits.getval(fn, 'cdelt3')
        a_pix = fits.getval(fn, 'crpix3')
        a_ref = fits.getval(fn, 'crval3')
        return (a_ref + (np.arange(a_len) - a_pix + 0.5) * a_del)

    def readvelocityaxis(self, fn):
        """
        Wrapper for _velocityaxis and _spectralaxis.
        """
        if fits.getval(fn, 'ctype3').lower() == 'freq':
            specax = self._spectralaxis(fn)
            try:
                nu = fits.getval(fn, 'restfreq')
            except KeyError:
                nu = fits.getval(fn, 'restfrq')
            return (nu - specax) * sc.c / nu
        else:
            return self._velocityaxis(fn)

    def readpositionaxis(self, fn, a=1):
        """
        Returns the position axis in ["].
        """
        if a not in [1, 2]:
            raise ValueError("'a' must be in [0, 1].")
        a_len = fits.getval(fn, 'naxis%d' % a)
        a_del = fits.getval(fn, 'cdelt%d' % a)
        a_pix = fits.getval(fn, 'crpix%d' % a)
        return 3600. * ((np.arange(1, a_len+1) - a_pix + 0.5) * a_del)

    def rotate(self, x, y, t):
        '''
        Rotation by angle t [rad].
        '''
        x_rot = x * np.cos(t) + y * np.sin(t)
        y_rot = y * np.cos(t) - x * np.sin(t)
        return x_rot, y_rot

    def incline(self, x, y, i):
        '''
        Incline the image by angle i [rad].
        '''
        return x, y / np.cos(i)

    def gaussian2D(self, dx, dy, pa=0.0):
        """
        2D Gaussian kernel in pixel coordinates.
        """
        xm = np.arange(-4*max(dy, dx), 4*max(dy, dx)+1)
        x, y = np.meshgrid(xm, xm)
        x, y = self.rotate(x, y, pa)
        k = np.power(x / dx, 2) + np.power(y / dy, 2)
        k -= 2. * x * y / dx / dy
        return np.exp(-0.5 * k) / 2. / np.pi / dx / dy
