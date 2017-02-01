from scipy.io import readsav
import numpy as np
import scipy.constants as sc
from scipy.interpolate import griddata


class idlmodel:
    """Class to read in a IDL model structure from Roy."""

    def __init__(self, path, model, verbose=False, **kwargs):
        """Read in the model and parse the various arrays."""
        recarr = readsav(path, python_dict=True, verbose=verbose)
        self.filename = path.split('/')[-1]
        self.recarr = recarr[recarr.keys()[0]][int(model)]
        if verbose:
            print self.recarr.dtype

        # Coordinates.
        self.rvals = self.recarr['r']['mean'][0] / 1e2 / sc.au
        self.tvals = np.radians(self.recarr['theta']['mean'][0])
        self.xvals = self.rvals[None, :] * np.cos(self.tvals[:, None])
        self.yvals = self.rvals[None, :] * np.sin(self.tvals[:, None])

        # Physical properties.
        self.temperature = self.recarr['t']
        self.gas = self.recarr['rho_gas']
        self.dust = self.recarr['rho_dust']
        self.g2d = self.gas / self.dust
        self.g2d = np.where(np.isfinite(self.g2d), self.g2d, np.nan)
        self.abun = self.recarr['abun']
        self.scaleheight = self.recarr['hp'] / 1e2 / sc.au

        # Model specific values.
        self.mdust = self.recarr['mdust']
        self.mgas = self.recarr['mgas']
        self.alpha = self.recarr['alpha']
        self.totalg2d = self.recarr['pars']['dust'][0]['gas2dust'][0]
        self.rin = self.recarr['pars']['sigma'][0]['rin'][0]
        self.rout = self.recarr['pars']['sigma'][0]['rout'][0]

        # Stellar variables.
        self.star = star(self.recarr)

        # Grain properties.
        self.grains = grains(self.recarr, kwargs.get('ignore_bins', 1))
        return

    def writeALCHEMIC(self, mindens=1e3, fileout=None, **kwargs):
        """Write model for ALCHEMIC."""

        temp_grid = self.grid_data(self.xvals, self.yvals,
                                   self.temperature,
                                   **kwargs).ravel()

        dens_grid = self.grid_data(self.xvals, self.yvals,
                                   np.log10(self.gas),
                                   **kwargs).ravel()
        dens_grid = np.power(10., dens_grid)

        size_grid = self.grid_data(self.xvals, self.yvals,
                                   np.log10(self.grains.effsizecell),
                                   **kwargs).ravel()
        size_grid = np.power(10., size_grid)

        g2d_grid = self.grid_data(self.xvals, self.yvals,
                                  self.g2d, **kwargs).ravel()

        rgrid, zgrid = self.estimate_grids(self.xvals, self.yvals, **kwargs)
        r_grid = (rgrid[None, :] * np.ones(zgrid.size)[:, None]).ravel()
        z_grid = (np.ones(rgrid.size)[None, :] * zgrid[:, None]).ravel()

        # Sort the data and then mask out values.

        idx = np.argsort(r_grid)
        tosave = np.squeeze([np.take(r_grid, idx),
                             np.take(z_grid, idx),
                             np.take(temp_grid, idx),
                             np.take(dens_grid, idx),
                             np.take(size_grid, idx),
                             np.take(g2d_grid, idx)])
        mask = self.density_mask(tosave[3], mindens)
        tosave = np.delete(tosave, mask, axis=1)

        # Save to the file. If not filename is given, use the input model file.
        if fileout is None:
            fileout = self.filename.split('/')[-1]
            fileout = '.'.join(fileout.split('.')[:-1])+'.dat'
        header = 'r [au], z [au], T [K], rho [g/ccm], a [um], g2d'
        np.savetxt(fileout, tosave.T, fmt='%.5e', header=header)
        print('Successfully saved to', fileout)
        return

    def grid_data(self, xvals, yvals, pvals, **kwargs):
        """Uses griddata to grid the values."""
        rgrid, zgrid = self.estimate_grids(xvals, yvals, **kwargs)
        return griddata((xvals.ravel(), yvals.ravel()), pvals.ravel(),
                        (rgrid[None, :], zgrid[:, None]), method='linear')

    def estimate_grids(self, xpnts, ypnts, **kwargs):
        """Return the axes onto which to grid."""
        nr = kwargs.get('nr', 100)
        nz = kwargs.get('nz', 100)
        if kwargs.get('log', False):
            assert ypnts.min() > 0
            rgrid = np.logspace(np.log10(xpnts.min()),
                                np.log10(xpnts.max()),
                                nr)
            zgrid = np.logspace(np.log10(ypnts.min()),
                                np.log10(ypnts.max()),
                                nz)
        else:
            rgrid = np.linspace(xpnts.min(), xpnts.max(), nr)
            zgrid = np.linspace(ypnts.min(), ypnts.max(), nz)
        return rgrid, zgrid

    def points_to_grid(self, mindens):
        """Returns the points to grid."""
        mask = self.density_mask(mindens)
        xpnt = self.xvals[mask]
        ypnt = self.yvals[mask]
        temp = self.temperature[mask]
        dens = self.gas[mask]
        size = self.grains.effsizecell[mask]
        g2dr = self.g2d[mask]
        return np.array([xpnt, ypnt, temp, dens, g2dr, size])

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
