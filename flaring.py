'''
Functions to try and measure the flaring of CO emission after Rosenfeld++ 2013.

By providing the datacube with which to make a comparison, the sky coordinates,
(x_sky, y_sky) can be derived, along with the velocity axis. The inclination,
position angle, distance to the source and stellar mass must all be provided.

Several coordinate systems will be derived from these, including _dep values
which are the midplane values of the disk for each pixel in [au].

By providing 'get_model' with an opening angle and linewidth a model can be
built. This will return a cube of the same shape at the provided datacube, with
each voxel the fraction of the total integrated intensity which originates from
there.

TODO:
- Can we somehow estimate the line width from the data?
- Weighting of the near and far cones.
'''

import numpy as np
import scipy.constants as sc
from beamclass import synthbeam
from astropy.io import fits
from astropy.io.fits import getval
from scipy.interpolate import interp1d
from astropy.convolution import convolve
from astropy.convolution import convolve_fft


class conicalmodel:

    # Within this class use the convention that all distances are in [au],
    # all velocities are in [m/s] and all angles are in [rad].
    # Try and use the convention 'coord' is a single array of coordinates,
    # while 'coords' is a list of both the near and far side coordinates.

    def __init__(self, path, beamparams=None, **kwargs):
        '''Read in the datacube.'''

        # Default values.
        self.inc = kwargs.get('inc', 0.0)
        self.pa = kwargs.get('pa', 0.0)
        self.dist = kwargs.get('dist', 150.)
        self.mstar = kwargs.get('mstar', 1.0)
        self.verbose = kwargs.get('verbose', True)

        # Read in the .fits file and determine the axes.

        self.path = path
        self.data = fits.getdata(self.path, 0)
        self.bunit = getval(self.path, 'bunit', 0).lower()

        self.posax = self.read_posax()
        self.npix = getval(self.path, 'naxis2', 0)
        self.dpix = np.diff(self.posax).mean()

        self.velax = self.read_velax()
        self.nchan = getval(self.path, 'naxis3', 0)
        self.dvchan = np.diff(self.velax).mean()

        # Derived coordinates.
        # _sky - image coordinates.
        # _rot - rotated coordinates
        # _dep - midplane coordinates.

        self.x_sky = self.posax[None, :] * np.ones(self.npix)[:, None]
        self.y_sky = self.posax[:, None] * np.ones(self.npix)[None, :]
        self.x_rot, self.y_rot = rotate(self.x_sky, self.y_sky, self.pa)
        self.x_dep, self.y_dep = incline(self.x_rot, self.y_rot, self.inc)
        self.r_dep = np.hypot(self.x_dep, self.y_dep)
        self.t_dep = np.arctan2(self.y_dep, self.x_dep)

        # From the derived coordinates, can estimate a radial integrated
        # intensity profile. This consists of both an RMS estimate of the
        # background and a binning. Note that self.profile is not the total
        # flux.

        self.rms = self.estimate_rms(**kwargs)
        self.nsig = kwargs.get('nsig', 3)
        self.masked = self.mask_data(self.rms, self.nsig, **kwargs)
        self.zeroth = self.get_zeroth(self.masked, **kwargs)
        self.profile, self._profile = self.get_profile(self.zeroth, **kwargs)
        self.zeroth = self.get_zeroth(self.masked, self.velax, **kwargs)
        self.int_profile, _ = self.get_profile(self.zeroth, **kwargs)

        # Include the beamclass.

        self.beam = synthbeam(self.path, beamparams, dist=self.dist, **kwargs)

        # Dictionaries to save pre-calculated models.
        # For: _phis, keys are a (phi, linewidth) tuple.

        self._phis = {}

        # Read out information if verbose.

        if self.verbose:
            print('Successfully read in %s.' % self.path.split('/')[-1])
            print('Velocity axis has %d channels.' % self.velax.size)
            print('Systemic velocity is index %d.' % abs(self.velax).argmin())
            print('Estimated RMS noise at %.2e %s.' % (self.rms, self.bunit))
            print('Clipped data below %.f sigma.' % self.nsig)

        return

    def channel(self, cidx, phi, linewidth, FFT=None, clip=True):
        """Returns model channel emission."""
        if FFT is None:
            chan = self.get_channel(cidx, phi, linewidth)
        else:
            chan = self.get_channel_conv(cidx, phi, linewidth, FFT)
        if clip:
            chan = np.where(chan >= self.nsig * self.rms, chan, 0.0)
        return chan

    def get_channel_conv(self, cidx, phi, linewidth, FFT=True):
        """Convolve the channel with FFT-convolution."""
        chan = self.get_channel(cidx, phi, linewidth)
        if self.beam.checkbeam:
            if self.verbose:
                print('No beam specified. Ignoring convolution.')
            return chan
        if FFT:
            return convolve_fft(chan, self.beam.kernel)
        return convolve(chan, self.beam.kernel)

    def mask_data(self, rms=None, nsig=None, **kwargs):
        """Mask the data by some RMS value."""
        if rms is None:
            rms = self.rms
        if nsig is None:
            nsig = self.nsig
        maskval = kwargs.get('maskval', 0.0)
        return np.where(self.data >= nsig * rms, self.data, maskval)

    def estimate_rms(self, **kwargs):
        """Estimate the RMS of the datacube."""
        nchan = kwargs.get('nchan', 2)
        linefree = np.hstack([self.data[:nchan], self.data[-nchan:]])
        return np.nanstd(linefree)

    def get_zeroth(self, data=None, velax=None, **kwargs):
        """Return the zeroth moment of the datacube."""
        if data is None:
            data = self.data
        if velax is None:
            return np.nansum(data, axis=0)
        return np.trapz(data, velax, axis=0)

    def get_profile(self, toavg=None, **kwargs):
        """Return the radial intensity profile."""
        nbins = kwargs.get('nbins', 20)
        if toavg is None:
            toavg = self.data
        rpnts = np.linspace(self.r_dep.min(), self.r_dep.max(), nbins)
        ridxs = np.digitize(self.r_dep.ravel(), rpnts)
        ipnts = [np.percentile(toavg.ravel()[ridxs == r],
                               [14., 50., 86.])
                 for r in range(1, nbins+1)]
        ipnts = np.vstack([rpnts, np.squeeze(ipnts).T])
        interp = interp1d(rpnts, ipnts[2])
        return ipnts, interp

    def interpolate_profile(self, rvals):
        """Returns an interpolated intensity value at rvals."""
        return self._profile(rvals)

    def get_model(self, phi, linewidth):
        """Return a conical model."""
        try:
            return self._phis[phi, linewidth]
        except:
            pass

        # TODO: How to better weight this part? There is a difference between
        #       the near and front cone but currently this isn't taken into
        #       account.
        # TODO: Should this be at a higher sampling rate?
        #       If yes, make it a multiple of self.dvchan. But then how would
        #       one index this in 'intensity_fraction'?

        pos, neg = self.get_3Dcylin(phi)
        vproj = [self.proj_velocity(pos), self.proj_velocity(neg)]
        model = [normgaussian(self.velax[:, None, None],
                              vproj[0][None, :, :], linewidth),
                 normgaussian(self.velax[:, None, None],
                              vproj[1][None, :, :], linewidth)]
        model = np.amax(model, axis=0) * self.dvchan
        self._phis[phi, linewidth] = model
        return model

    def get_channel(self, cidx, phi, linewidth):
        """Returns the model intensities."""
        ifrac = self.intensity_fraction(cidx, phi, linewidth)
        emiss = self.interpolate_profile(self.r_dep) * ifrac
        return np.where(emiss >= self.nsig * self.rms, emiss, 0.0)

    def intensity_fraction(self, cidx, phi, linewidth):
        """Fraction of intensity at given (x, y, v) voxel."""
        return np.take(self.get_model(phi, linewidth), cidx, axis=0)

    def proj_velocity(self, coord, cylin=True):
        """Returns projected velocity pixel in [m/s]."""
        if cylin is False:
            coord = cart_to_cylin(coord)
        vel = np.sqrt(sc.G * self.mstar * 1.989e30 / coord[0] / sc.au)
        vel *= np.sin(self.inc) * np.cos(coord[1])
        return vel

    def get_3D(self, phi=0.0):
        """Returns the (x, y, z) intercepts at each pixel."""
        pos_t, neg_t = self.get_intercept(phi)
        pos_coords = [self.x_dep,
                      self.y_dep + pos_t * np.sin(self.inc),
                      pos_t * np.cos(self.inc)]
        neg_coords = [self.x_dep,
                      self.y_dep + neg_t * np.sin(self.inc),
                      neg_t * np.cos(self.inc)]
        return pos_coords, neg_coords

    def get_3Dcylin(self, phi=0.0):
        """Returns the (r, t, z) intercepts at each pixel."""
        pos_cart, neg_cart = self.get_3D(phi)
        return cart_to_cylin(pos_cart), cart_to_cylin(neg_cart)

    def get_intercept(self, phi=0.0):
        """Roots of Eqn. 5 from Rosenfeld++ 2013."""
        a = np.cos(2. * self.inc) + np.cos(2. * phi)
        b = -4. * np.power(np.sin(phi), 2) * np.sin(self.inc) * self.y_dep
        c = -2. * np.power(np.sin(phi), 2) * np.power(self.r_dep, 2)
        p = (-b + np.sqrt(np.power(b, 2) - 4. * a * c)) / 2. / a
        m = (-b - np.sqrt(np.power(b, 2) - 4. * a * c)) / 2. / a
        return np.amax([p, m], axis=0), np.amin([p, m], axis=0)

    def read_posax(self):
        """Returns the position axis in [au]."""
        a_len = getval(self.path, 'naxis2', 0)
        a_del = getval(self.path, 'cdelt2', 0)
        a_pix = getval(self.path, 'crpix2', 0)
        posax = np.arange(a_len) - a_pix + 1
        return self.dist * 3600. * posax * a_del

    def read_velax(self):
        """Returns the velocity axis in [m/s]."""
        a_len = getval(self.path, 'naxis3', 0)
        a_del = getval(self.path, 'cdelt3', 0)
        a_pix = getval(self.path, 'crpix3', 0)
        return (np.arange(1, a_len+1) - a_pix) * a_del


# Static methods:
# rotate - anticlockwise rotation.
# incline - deproction.
# gaussian - Gaussian function.
# normgaussian - normalised Gaussian function.
# cart_to_cylin - convert (x, y, x) -> (r, t, z)
# cylin_to_cart - convert (r, t, z) -> (x, y, z)


def rotate(x_coords, y_coords, angle):
    '''Anticlockwise rotation.'''
    x_rot = x_coords * np.cos(angle) + y_coords * np.sin(angle)
    y_rot = y_coords * np.cos(angle) - x_coords * np.sin(angle)
    return x_rot, y_rot


def incline(x_coords, y_coords, angle):
    '''Incline the image.'''
    x_dep = x_coords
    y_dep = y_coords / np.cos(angle)
    return x_dep, y_dep


def gaussian(x, x0, dx):
    '''Gaussian.'''
    return np.exp(-0.5 * np.power((x - x0) / dx, 2))


def normgaussian(x, x0, dx):
    '''Normalised Gaussian.'''
    return gaussian(x, x0, dx) / dx / np.sqrt(2. * np.pi)


def cart_to_cylin(cart):
    '''Converts cartestian coords to cylindrical coords.'''
    return np.array([np.hypot(cart[1], cart[0]),
                     np.arctan2(cart[1], cart[0]),
                     cart[2]])


def cylin_to_cart(cylin):
    '''Converts cylindrical coords to cartestian coords.'''
    return np.array([cylin[0] * np.cos(cylin[1]),
                     cylin[0] * np.sin(cylin[1]),
                     cylin[2]])
