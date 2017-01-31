from scipy.io import readsav
import numpy as np
import scipy.constants as sc


class idlmodel:
    """Class to read in a IDL model structure from Roy."""

    def __init__(self, path, model, verbose=False, **kwargs):
        """Read in the model and parse the various arrays."""
        recarr = readsav(path, python_dict=True, verbose=verbose)
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
        self.dust = self.recarr['rho_dust']
        self.abun = self.recarr['abun']
        self.scaleheight = self.recarr['hp'] / 1e2 / sc.au

        # Model specific values.
        self.mdust = self.recarr['mdust']
        self.mgas = self.recarr['mgas']
        self.alpha = self.recarr['alpha']
        self.gastodust = self.recarr['pars']['dust'][0]['gas2dust'][0]
        self.rin = self.recarr['pars']['sigma'][0]['rin'][0]
        self.rout = self.recarr['pars']['sigma'][0]['rout'][0]

        # Stellar variables.
        self.star = star(self.recarr)

        # Grain properties.
        self.grains = grains(self.recarr, kwargs.get('ignore_bins', 1))
        return


class grains:
    """Grains subclass."""
    def __init__(self, recarr, ignore=1):
        """Ignore the final `ignore` populations."""

        if ignore == 0:
            raise NotImplementedError()

        self.abundance = recarr['abun'][:-ignore]
        self.grainprops = recarr['kappa'][0]['pars']
        self.minsizes = self.grainprops['amin'][:-ignore]
        self.maxsizes = self.grainprops['amax'][:-ignore]
        self.exponent = self.grainprops['alpha'][:-ignore]
        self.nbins = self.minsizes.size

        assert self.maxsizes.size == self.nbins
        assert self.exponent.size == self.nbins

        self.avgsizebin = self.calcAverageSizeBin()
        self.avgsizecell = self.calcAverageSizeCell()
        return

    def calcAverageSizeBin(self):
        """Return average grain size per bin."""
        avg = np.zeros(self.nbins)
        for i in range(self.nbins):
            size = np.linspace(self.minsizes[i], self.maxsizes[i], 1e5)
            freq = np.power(size, self.exponent[i])
            avg[i] = np.average(size, weights=freq)
        return avg

    def calcAverageSizeCell(self):
        """Return the average grain size per cell."""
        avg3D = self.avgsizebin[:, None, None] * np.ones(self.abundance.shape)
        return np.average(avg3D, weights=self.abundance, axis=0)


class star:
    """Star subclass."""
    def __init__(self, recarr):
        self.starprops = recarr['pars']['star'][0]
        self.mass = self.starprops['mstar'][0]
        self.luminosity = self.starprops['lstar'][0]
        self.distance = self.starprops['dpc'][0]
        return
