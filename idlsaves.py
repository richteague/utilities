from scipy.io import readsav
import numpy as np
import scipy.constants as sc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import warnings


class idlmodel:
    """Class to read in a IDL model structure from Roy."""

    def __init__(self, path, model=0, verbose=False, **kwargs):
        """Read in the model and parse the various arrays."""

        recarr = readsav(path, python_dict=True, verbose=verbose)
        self.filename = path.split('/')[-1]
        self.recarr = recarr[recarr.keys()[0]][int(model)]
        if verbose:
            print self.recarr.dtype
        else:
            warnings.filterwarnings('ignore')

        # Coordinates.
        self.rvals = self.recarr['r']['mean'][0] / 1e2 / sc.au
        self.tvals = np.radians(self.recarr['theta']['mean'][0])
        self.xvals = self.rvals[None, :] * np.cos(self.tvals[:, None])
        self.yvals = self.rvals[None, :] * np.sin(self.tvals[:, None])

        # Physical properties. Note that `abun` is the grainsize bin abundance.
        self.temperature = self.recarr['t']
        self.gas = self.recarr['rho_gas']
        self.dust = self.recarr['rho_dust']
        self.abun = self.recarr['abun']

        # Gas-to-dust ratios.
        self.totalg2d = self.recarr['pars']['dust'][0]['gas2dust'][0]
        self.g2d = self.gas / self.dust
        self.g2d = np.where(np.isfinite(self.g2d), self.g2d, np.nan)
        self.scaleheight = self.recarr['hp'] / 1e2 / sc.au

        # Model specific values.
        self.mdust = self.recarr['mdust']
        self.mgas = self.recarr['mgas']
        self.logalpha = float('%.2f' % np.log10(self.recarr['alpha']))
        self.rin = self.recarr['pars']['sigma'][0]['rin'][0]
        self.rout = self.recarr['pars']['sigma'][0]['rout'][0]

        # Stellar and grain roperties.
        self.star = star(self.recarr)
        self.grains = grains(self.recarr, kwargs.get('ignore_bins', 1))

        # Dictionary for precalculated ALCHEMIC structures.
        self._equalgrids = {}
        self._unequalgrids = {}
        return

    def cart_axes(self, nr=100, nz=100, log=False):
        """Return the axes onto which to grid."""
        if log:
            assert self.yvals.min() > 0
            rgrid = np.logspace(np.log10(self.xvals.min()),
                                np.log10(self.xvals.max()),
                                nr)
            zgrid = np.logspace(np.log10(self.yvals.min()),
                                np.log10(self.yvals.max()),
                                nz)
        else:
            rgrid = np.linspace(self.xvals.min(), self.xvals.max(), nr)
            zgrid = np.linspace(self.yvals.min(), self.yvals.max(), nz)
        return rgrid, zgrid

    def cart_grid(self, pvals, nr=100, nz=100, log=False):
        """Converts pvals from polar to cartestian gridding."""
        rgrid, zgrid = self.cart_axes(nr, nz, log)
        pgrid = griddata((self.xvals.ravel(), self.yvals.ravel()),
                         np.log10(pvals).ravel(),
                         (rgrid[None, :], zgrid[:, None]),
                         method='linear')
        return np.power(10., pgrid)

    def unequal_grid(self, mindens=1e3, nr=100, nz=100, log=False):
        """Return 1+1D grid with unequal number of z points at each r."""
        try:
            return self._unequalgrids[mindens, nr, nz, log]
        except:
            pass
        rpts, zpts = self.cart_axes(nr, nz, log)
        rval = (rpts[None, :] * np.ones(nz)[:, None]).ravel()
        zval = (np.ones(nr)[None, :] * zpts[:, None]).ravel()
        temp = self.cart_grid(self.temperature, nr, nz, log)
        dens = self.cart_grid(self.gas, nr, nz, log)
        size = self.cart_grid(self.grains.effsize, nr, nz, log)
        g2dr = self.cart_grid(self.g2d, nr, nz, log)

        # Sort the data into increasing radius. Remove values is a value for
        # mindens has been supplied.

        idx = np.argsort(rval)
        arr = np.squeeze([np.take(rval, idx), np.take(zval, idx),
                          np.take(temp, idx), np.take(dens, idx),
                          np.take(size, idx), np.take(g2dr, idx)])

        if mindens is not None:
            arr = np.delete(arr, self.density_mask(arr[3], mindens), axis=1)
        self._unequalgrids[mindens, nr, nz, log] = np.nan_to_num(arr)
        return arr

    def equal_grid(self, mindens=1e3, nr=100, nz=100, log=False):
        """Return 1+1D grid with equal number of z points at each r."""
        try:
            return self._equalgrids[mindens, nr, nz, log]
        except:
            pass

        high_res = self.unequal_grid(mindens, nr, nz*100, log)
        rvals = np.unique(high_res[0])
        grid = [[], [], [], [], [], []]

        # Isolate the vertical column to interpolate from.

        for r in rvals:
            idx = [i for i in range(high_res[0].size) if high_res[0][i] == r]
            column = np.take(high_res, idx, axis=1)

            # Interpolate the new values. For some reason, np.interp() doesn't
            # work as well as scipy.interpolate.interp1d. Interpolate in
            # log-space for a smoother result.

            rvals = np.ones(nz) * r
            zvals = np.linspace(column[1].min(), column[1].max(), nz)
            grid[0] = np.concatenate([grid[0], rvals], axis=0)
            grid[1] = np.concatenate([grid[1], zvals], axis=0)
            for i in range(2, len(grid)):
                interp = interp1d(column[1], np.log10(column[i]))
                insert = np.power(10, interp(zvals))
                grid[i] = np.concatenate([grid[i], insert], axis=0)

        self._equalgrids[mindens, nr, nz, log] = np.squeeze(grid)
        return self._equalgrids[mindens, nr, nz, log]

    def toALCHEMIC(self, fileout, equal=True, **kwargs):
        """Save model for ALCHEMIC."""
        nr = kwargs.get('nr', 100)
        nz = kwargs.get('nz', 100)
        log = kwargs.get('log', False)
        mindens = kwargs.get('mindens', 1e3)

        if equal:
            arr = self.equal_grid(mindens, nr, nz, log)
        else:
            arr = self.unequal_grid(mindens, nr, nz, log)

        print arr.shape
        header = 'r [au], z [au], T [K], rho [g/ccm], a [um], g2d'
        np.savetxt(fileout, arr.T, fmt='%.5', header=header)
        print('Successfully saved to %s.' % fileout)
        return

    def density_mask(self, dens_grid, mindens=1e3):
        """Select points in disk by their gas density."""
        mask = dens_grid >= mindens * sc.m_p * 2.
        mask = mask.ravel()
        return [i for i in range(len(mask)) if not mask[i]]


class grains:
    """Grains subclass."""
    def __init__(self, recarr, ignore=1):
        """Ignore the final `ignore` populations."""

        if ignore == 0:
            raise NotImplementedError()

        self.abundance = recarr['abun'][:-ignore]
        self.modelpnts = self.abundance[0].shape
        self.grainprops = recarr['kappa'][0]['pars']
        self.minsizes = self.grainprops['amin'][:-ignore]
        self.maxsizes = self.grainprops['amax'][:-ignore]
        self.binrange = self.maxsizes - self.minsizes
        self.exponent = self.grainprops['alpha'][:-ignore]
        self.nbins = self.minsizes.size

        assert self.maxsizes.size == self.nbins
        assert self.exponent.size == self.nbins

        self.avgsizebin = self.calcAverageSizeBin()
        self.avgsizecell = self.calcAverageSizeCell()
        self.effsize = self.calcEffectiveSize()
        return

    def calcAverageSizeBin(self):
        """Return average grain size per bin."""
        avg = np.zeros(self.nbins)
        for i in range(self.nbins):
            size = np.linspace(self.minsizes[i], self.maxsizes[i], 1e5)
            freq = np.power(size, self.exponent[i])
            avg[i] = np.average(size, weights=freq)
        return avg

    def calcAverageSurfaceBin(self):
        """Return average grain surface area per bin."""
        avg = np.zeros(self.nbins)
        for i in range(self.nbins):
            size = np.linspace(self.minsizes[i], self.maxsizes[i], 1e5)
            freq = np.power(size**2, self.exponent[i])
            avg[i] = np.average(size, weights=freq)
        return avg

    def calcAverageSizeCell(self):
        """Return the average grain size per cell."""
        avg3D = self.avgsizebin[:, None, None] * np.ones(self.abundance.shape)
        return np.average(avg3D, weights=self.abundance, axis=0)

    def calcEffectiveSize(self):
        """Return the effect grain size from Eqn 9, Vasyunin++ 2011."""
        avg3D = self.avgsizebin[:, None, None] * np.ones(self.abundance.shape)
        return np.average(avg3D, weights=self.abundance * avg3D**2, axis=0)


class star:
    """Star subclass."""
    def __init__(self, recarr):
        self.starprops = recarr['pars']['star'][0]
        self.mass = self.starprops['mstar'][0]
        self.luminosity = self.starprops['lstar'][0]
        self.distance = self.starprops['dpc'][0]
        return
