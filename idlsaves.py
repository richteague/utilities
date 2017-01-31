from scipy.io import readsav
import numpy as np
import scipy.constants as sc


class idlmodel:
    """Class to read in a IDL model structure from Roy."""

    def __init__(self, path, model, verbose=False):
        """Read in the model and parse the various arrays."""
        recarr = readsav(path, python_dict=True, verbose=verbose)
        self.recarr = recarr[recarr.keys()[0]][int(model)]
        if verbose:
            print self.recarr.dtype

        # Coordinates. Can plot directly with contourf, however
        # also project onto cartestian.
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

        # Stellar variables.
        self.mstar = self.recarr['pars']['star'][0]['mstar'][0]
        self.lstar = self.recarr['pars']['star'][0]['lstar'][0]
        self.distance = self.recarr['pars']['star'][0]['dpc'][0]

        # Surface density variables.
        self.rin = self.recarr['pars']['sigma'][0]['rin'][0]
        self.rout = self.recarr['pars']['sigma'][0]['rout'][0]

        # Dust properties.
        self.dustprops = self.recarr['pars']['dust'][0]
        self.gastodust = self.dustprops['gas2dust'][0]
        self.asmalldust = [self.dustprops['aminsmall'],
                           self.dustprops['amaxsmall']]
        self.alargedust = [self.dustprops['aminlarge'],
                           self.dustprops['amaxlarge']]
        self.msmalldust = 10**self.dustprops['logmdustsmall'][0]
        self.mlargedust = 10**self.dustprops['logmdustlarge'][0]

        return
