 # Functions for the CS dip in TW Hya paper.

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.constants as sc
from astropy.io.fits import getval
from matplotlib.ticker import MultipleLocator, LogLocator
from plottingRoutines import plottingRoutines as pr


#-------- Chemical Model Functions --------#


# Read in the appropriate files.
def getChemicalModels(mol, folder='ChemicalModels/models_080816/',
                      returndata=True):
    files = np.array([folder+fn for fn in os.listdir(folder) 
                      if (parseChemicalModel(fn)[1] == mol)])
    names = np.array([parseChemicalModel(fn)[0] for fn in files])
    files = np.squeeze([files[names == name] for name in sorted(names)])
    names = sorted(names)
    if returndata:
        datas = [clipModel(np.loadtxt(fn, skiprows=3).T) for fn in files]
        return files, names, datas
    else:
        return files, names

# Parse the filename.
def parseChemicalModel(filename):
    fn = filename.split('/')[-1]
    i = 0
    while fn[:i] != 'TW_Hya_U.Gorti_':
        i += 1
    j = i
    while fn[i] != '.':
        i += 1
    model = fn[j:i].replace('_', ' ')
    if 'dust' in model:
        dust = True
    else:
        dust = False
    while fn[j:] != '.time_1Myr.out':
        j +=1
    i = j
    while fn[i] != '_':
        i -= 1
    molecule = fn[i+1:j]
    if 'standard' == model:
        model = 'Model A'
    elif '2' in model:
        model = 'Model C'
    else:
        model = 'Model B'
    if dust:
        model += 'd'
    return model, molecule
    
# Clip the chemical model.
def clipModel(model, rin=10., rout=180.):
    ncol = model.shape[0]
    model = np.array([model[i][model[3] > 0] for i in range(ncol)])
    model = np.array([model[i][model[0] >= rin] for i in range(ncol)])
    model = np.array([model[i][model[0] < rout] for i in range(ncol)])
    return model


# Get the surface density.
def getSurfaceDensity(model, mu=2.34):
    rvals = np.unique(model[0])
    sigma = np.array([2.*np.trapz(model[2][model[0] == r] / sc.m_p / mu / 1e3, 
                                  x=abs(model[1][model[0] == r]*sc.au*100.)) 
                      for r in rvals])
    return rvals, sigma


# Get the molecule column density.
def getColumnDensity(model):
    rvals = np.unique(model[0])
    sigma = np.array([2.*np.trapz(model[7][model[0] == r], 
                                  x=abs(model[1][model[0] == r]*sc.au*100.)) 
                      for r in rvals])
    return rvals, np.where(getSurfaceDensity(model)[1] > 0, sigma, 0)
    
    
# Get the relative abundance of the molecule.
def getRelativeAbundance(model, r, mu=2.34):
    xmol = model[7] / (model[2] / sc.m_p / 1e3 / mu)
    xmol = xmol[model[0] == r]
    return np.where(np.isfinite(xmol), xmol, 0.0)


# Get the abundance per cell.
def getCellAbundance(model, r):
    zvals = model[1][model[0] == r] * sc.au * 1e2
    cells = np.average([zvals[1:], zvals[:-1]], axis=0)
    cells = np.insert(cells, -1, cells[-1])
    nmols = model[7][model[0] == r] * cells
    if np.nanmean(nmols) == 0.:
        nmols = np.ones(nmols.size)
    return np.where(np.isfinite(nmols), nmols, 0.0)


# Get the abundance weighted relative abundance.
def getAverageRelAbund_weighted(model):
    rvals = np.unique(model[0])
    xmol = np.array([np.average(getRelativeAbundance(model, r), 
                                weights=getCellAbundance(model, r))
                    for r in rvals])
    dxmol = np.array([np.average((getRelativeAbundance(model, r)-xmol[ridx])**2, 
                                  weights=getCellAbundance(model, r))
                    for ridx, r in enumerate(rvals)])**0.5
    xmol = np.where(getSurfaceDensity(model)[1] != 0., xmol, 1e-20)
    return rvals, xmol, dxmol


def getRelativeAbund(model):
    rvals, nmol = getColumnDensity(model)
    sigma = getSurfaceDensity(model)[1]
    xmol = np.where(sigma != 0., nmol/sigma, 0.)
    return rvals, xmol, np.zeros(rvals.size)
    
def getAverageTemp_weighted(model):
    rvals = np.unique(model[0])
    temp = np.array([np.average(model[3][model[0] == r], 
                                weights=getCellAbundance(model, r))
                    for r in rvals])
    dtemp = np.array([np.average((model[3][model[0] == r]-temp[ridx])**2, 
                                 weights=getCellAbundance(model, r))
                    for ridx, r in enumerate(rvals)])**0.5
    temp = np.where(getSurfaceDensity(model)[1] > 0, temp, 0)
    dtemp = np.where(getSurfaceDensity(model)[1] > 0, dtemp, 0)
    return rvals, temp, dtemp
    
    
def getAverageHeight_weighted(model, unit='H'):
    rvals = np.unique(model[0])
    temp = np.array([np.average(model[1][model[0] == r], 
                                weights=getCellAbundance(model, r))
                    for r in rvals])
    dtemp = np.array([np.average((model[1][model[0] == r]-temp[ridx])**2, 
                                 weights=getCellAbundance(model, r))
                    for ridx, r in enumerate(rvals)])**0.5
    temp = np.where(getSurfaceDensity(model)[1] > 0, temp, 0)
    dtemp = np.where(getSurfaceDensity(model)[1] > 0, dtemp, 0)
    
    if unit == 'H':
        Hgas = getScaleHeight(model)[1]
        temp /= Hgas
        dtemp /= Hgas
    elif unit != 'au':
        raise NotImplementedError
    
    return rvals, temp, dtemp

# Return the pressure scale height of the model.
def getScaleHeight(model):
    rvals = np.unique(model[0])
    tmid = np.array([model[3][model[0] == r][model[1][model[0] == r].argmin()] 
                    for r in rvals])
    hgas = sc.k * tmid * (rvals*sc.au)**3. / 2.34 / sc.m_p / sc.G / 1.2e30
    hgas **= 0.5
    hgas /= sc.au   
    return rvals, hgas
    
def getAverageDensity_weighted(model, mu=2.34):
    rvals = np.unique(model[0])
    temp = np.array([np.average(model[2][model[0] == r], 
                                weights=getCellAbundance(model, r))
                    for r in rvals])
    dtemp = np.array([np.average((model[2][model[0] == r]-temp[ridx])**2, 
                                 weights=getCellAbundance(model, r))
                    for ridx, r in enumerate(rvals)])**0.5
    temp = np.where(getSurfaceDensity(model)[1] > 0, 
                    temp / sc.m_p / mu / 1e3,
                    1e-30)
    dtemp = np.where(getSurfaceDensity(model)[1] > 0, 
                     dtemp, 
                     0)
    return rvals, temp, dtemp    

#------ Plotting Routines ------#
#
#   #-- Basic Functions --#
#
#   plotColumnDensity()
#   plotRelativeColumnDensity()
#   plotRelativeAbundance()
#   plotAveragedRelativeAbundance()
#   plotAveragedDensity()
#   plotAveragedHeight()
#   plotAverageTemperature()
#
#   #-- Helper Functions --#
#
#   plotLegend()
#   getPlottingStyles()
#
#-------------------------------#

    
# Plot the column density.
def plotColumnDensity(modeldatas, modelnames, ax=None, legend=True, c_in=None, 
                      ls_in=None, lw_in=1, legend_cols=1, legend_loc=1, 
                      bap=1.25, fs=6, markdip=True):
    
    
    # Change to a list if only a single entry.
    if type(modeldatas) != list:
        modeldatas = [modeldatas]
        modelnames = [modelnames]
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
    # Loop through all the values and plot.
    # Assume we are plotting in the modle name order.
    lines = []
    for d, data in enumerate(modeldatas):
        
        name = modelnames[d]
        rvals, sigma = getColumnDensity(data)
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(name)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
            
        # Plot the figure.
        l,  = ax.semilogy(rvals, sigma, color=c, linestyle=ls,
                        label=texify(name), lw=lw_in, zorder=zo)
        lines.append(l)
    
    # Legend.
    if type(legend) is str:
        plotLegend(legend, lines, ax, bap=bap, fs=fs)
    elif legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=legend_loc, markerfirst=False, handlelength=1.5, 
                  ncol=legend_cols)

    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r'${\rm Radius \quad (au)}$', fontsize=7)
    
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    return

# Plot the realtive column densities.
def plotRelativeColumnDensity(modeldatas, modelnames, ax=None, legend=True, 
                              c_in=None, ls_in=None, lw_in=1, markdip=True, 
                              bap=1.25, fs=6, legend_loc=1):
    
    # Assume all on same radial grid.
    rvals = getColumnDensity(modeldatas[0])[0]
    sigmas = np.array([getColumnDensity(data)[1] for data in modeldatas]) 
    
    # New axis if necessary.
    if ax is None:
        fig, ax = plt.subplots()
    
    for s, sigma in enumerate(sigmas[1:]): 
        
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(modelnames[1+s])
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
        
        # Plot the lines
        ax.plot(rvals, 100.*(sigma-sigmas[0])/sigmas[0], color=c, linestyle=ls,
                label=texify(modelnames[1+s]), lw=lw_in, zorder=zo)

    # Default value.
    ax.axhline(0., c='k', lw=lw_in)
    
    # Perturbation position.
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    # Legend.
    if (legend_loc == 2 or legend_loc == 3):
        mf = True
    else:
        mf = False
    if legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5,
                  loc=legend_loc, markerfirst=mf, handlelength=1.5)
    
    # Gentrification.
    ax.xaxis.set_major_locator(MultipleLocator(30))  
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylabel(r'$\delta N \quad (\%)$')
    ax.set_xlabel(r'${\rm Radius \quad (au)}$')
    
    return

# Plot the relative abundance.
def plotRelativeAbundance(modeldatas, modelnames, ax=None, legend=True, 
                          mu=2.34, c_in=None, ls_in=None, lw_in=1, 
                          legend_cols=1, legend_loc=1, bap=1.25, fs=6, 
                          markdip=True):
    
    
    # Change to a list if only a single entry.
    if type(modeldatas) != list:
        modeldatas = [modeldatas]
        modelnames = [modelnames]
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
    # Loop through all the values and plot.
    # Assume we are plotting in the modle name order.
    lines = []
    for d, data in enumerate(modeldatas):
        
        name = modelnames[d]
        rvals, sigma = getColumnDensity(data)
        sigma /= getSurfaceDensity(data, mu=mu)[1]
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(name)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
            
        # Plot the figure.
        l,  = ax.semilogy(rvals, sigma, color=c, linestyle=ls,
                        label=texify(name), lw=lw_in, zorder=zo)
        lines.append(l)
    
    # Legend.
    if type(legend) is str:
        plotLegend(legend, lines, ax, bap=bap, fs=fs)
    elif legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=legend_loc, markerfirst=False, handlelength=1.5, 
                  ncol=legend_cols)

    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r'${\rm Radius \quad (au)}$', fontsize=7)
    
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    return
    
# Plot the abundance weighted relative abundance.
def plotAveragedRelativeAbundance(modeldatas, modelnames, ax=None, 
                                  legend=True, c_in=None, ls_in=None, lw_in=1,
                                  legend_cols=1, legend_loc=1, bap=1.25, fs=6,
                                  markdip=True, rightticks=False):

    # Change to a list if only a single entry.
    if type(modeldatas) != list:
        modeldatas = [modeldatas]
        modelnames = [modelnames]
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
    # Loop through the values.
    lines = []
    for d, data in enumerate(modeldatas):
        
        name = modelnames[d]
        x, y, dy = getAverageRelAbund_weighted(data)
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(name)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
            
        # Plot the figure.
        l,  = ax.semilogy(x, y, color=c, linestyle=ls,
                        label=texify(name), lw=lw_in, zorder=zo)
        lines.append(l)
    
    
    # Legend.
    if type(legend) is str:
        plotLegend(legend, lines, ax, bap=bap, fs=fs)
    elif legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=legend_loc, markerfirst=False, handlelength=1.5, 
                  ncol=legend_cols)
    
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r'${\rm Radius \quad (au)}$', fontsize=7)
    
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    if rightticks:
        makeTicksRight(ax)
    
    return
    
# Plot the abundance weighted average temperature.
def plotAveragedTemperature(modeldatas, modelnames, ax=None, legend=True, 
                            c_in=None, ls_in=None, lw_in=1, legend_cols=1, 
                            legend_loc=1, bap=1.25, fs=6, markdip=True, 
                            rightticks=False):

    # Change to a list if only a single entry.
    if type(modeldatas) != list:
        modeldatas = [modeldatas]
        modelnames = [modelnames]
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
    # Loop through the values.
    lines = []
    for d, data in enumerate(modeldatas):
        
        name = modelnames[d]
        x, y, dy = getAverageTemp_weighted(data)
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(name)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
            
        # Plot the figure.
        l,  = ax.plot(x, y, color=c, linestyle=ls,
                      label=texify(name), lw=lw_in, zorder=zo)
        lines.append(l)
    
    # Legend.
    if type(legend) is str:
        plotLegend(legend, lines, ax, bap=bap, fs=fs)
    elif legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=legend_loc, markerfirst=False, handlelength=1.5, 
                  ncol=legend_cols)
    
    # Gentrification.
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r'${\rm Radius \quad (au)}$', fontsize=7)
    
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    if rightticks:
        makeTicksRight(ax)
    
    return
    
# Plot the abundance weighted height.
def plotAveragedHeight(modeldatas, modelnames, ax=None, unit='au', 
                       legend=True, c_in=None, ls_in=None, lw_in=1, 
                       legend_cols=1, legend_loc=1, bap=1.25, fs=6, 
                       markdip=True, rightticks=False):

    # Change to a list if only a single entry.
    if type(modeldatas) != list:
        modeldatas = [modeldatas]
        modelnames = [modelnames]
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
    # Loop through the values.
    lines = []
    for d, data in enumerate(modeldatas):
        
        name = modelnames[d]
        x, y, dy = getAverageHeight_weighted(data, unit=unit)
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(name)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
            
        # Plot the figure.
        l,  = ax.plot(x, y, color=c, linestyle=ls,
                      label=texify(name), lw=lw_in, zorder=zo)
        lines.append(l)
    
    # Legend.
    if type(legend) is str:
        plotLegend(legend, lines, ax, bap=bap, fs=fs)
    elif legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=legend_loc, markerfirst=False, handlelength=1.5, 
                  ncol=legend_cols)
    
    # Gentrification.
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r'${\rm Radius \quad (au)}$', fontsize=7)
    
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    if rightticks:
        makeTicksRight(ax)
    
    return
    
def plotAveragedDensity(modeldatas, modelnames, ax=None, mu=2.34, 
                        legend=True, c_in=None, ls_in=None, lw_in=1, 
                        legend_cols=1, legend_loc=1, bap=1.25, fs=6, 
                        markdip=True, rightticks=False):

    # Change to a list if only a single entry.
    if type(modeldatas) != list:
        modeldatas = [modeldatas]
        modelnames = [modelnames]
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
    # Loop through the values.
    lines = []
    for d, data in enumerate(modeldatas):
        
        name = modelnames[d]
        x, y, dy = getAverageDensity_weighted(data, mu=mu)
        
        # Get common plotting styles.
        c, ls, zo = getPlottingStyles(name)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        if lw_in is not None:
            lw = lw_in
            
        # Plot the figure.
        l,  = ax.semilogy(x, y, color=c, linestyle=ls,
                          label=texify(name), lw=lw_in, zorder=zo)
        lines.append(l)
    
    # Legend.
    if type(legend) is str:
        plotLegend(legend, lines, ax, bap=bap, fs=fs)
    elif legend:
        ax.legend(fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=legend_loc, markerfirst=False, handlelength=1.5, 
                  ncol=legend_cols)
    
    # Gentrification.
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel(r'${\rm Radius \quad (au)}$', fontsize=7)
    
    if markdip:
        ax.axvline(80., c='gray', ls=':', lw=0.5, zorder=-5)
    
    if rightticks:
        makeTicksRight(ax)
    
    return
    
# Types of legend for plotting. Assumes 5 lines in name order.
def plotLegend(ltype, lines, ax, bap=1.25, fs=6):

    # Check input type.
    if type(ltype) != str:
        raise TypeError("ltype must be a string!")

    # Include the dummy line.
    l, = ax.plot(0, 0, color='none')
    lines.append(l)
    
    # Bottom right.
    if ltype == 'br':
        ax.legend((lines[5], lines[0], lines[1], 
                lines[2], lines[3], lines[4]),
               (r'', 
                r'${\rm Model \,\, A}$', 
                r'${\rm Model \,\, B}$', 
                r'${\rm Model \,\, Bd}$',
                r'${\rm Model \,\, C}$', 
                r'${\rm Model \,\, Cd}$'),
               fontsize=fs, borderaxespad=bap, labelspacing=.5, loc=4, 
               ncol=3, markerfirst=False, handlelength=1.5)
                   
    # Top right.
    elif ltype == 'tr':
        ax.legend((lines[0], lines[5], lines[1], 
                   lines[2], lines[3], lines[4]),
                  (r'${\rm Model \,\, A}$',
                   r'',  
                   r'${\rm Model \,\, B}$', 
                   r'${\rm Model \,\, Bd}$',
                   r'${\rm Model \,\, C}$', 
                   r'${\rm Model \,\, Cd}$'),
                  fontsize=fs, borderaxespad=bap, labelspacing=.5, loc=1, 
                  ncol=3, markerfirst=False, handlelength=1.5)
                   
    # Bottom left.
    elif ltype == 'bl':
        ax.legend((lines[3], lines[4], lines[1], 
                   lines[2], lines[5], lines[0]),
                  (r'${\rm Model \,\, C}$', 
                   r'${\rm Model \,\, Cd}$', 
                   r'${\rm Model \,\, B}$', 
                   r'${\rm Model \,\, Bd}$', 
                   r'', 
                   r'${\rm Model \,\, A}$'),
                  fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=3, ncol=3, handlelength=1.5)
    
    # Top left.
    elif ltype == 'tl':
        ax.legend((lines[3], lines[4], lines[1], 
                   lines[2], lines[0], lines[5]),
                  (r'${\rm Model \,\, C}$', 
                   r'${\rm Model \,\, Cd}$', 
                   r'${\rm Model \,\, B}$', 
                   r'${\rm Model \,\, Bd}$', 
                   r'${\rm Model \,\, A}$'
                   r'', ),
                  fontsize=fs, borderaxespad=bap, labelspacing=.5, 
                  loc=2, ncol=3, handlelength=1.5)
        
    else:
        raise ValueError("ltype must be: 'tr', 'br', 'tl' or 'bl'.")
            
    return

# Return the plotting style for the lines based on the model name.
def getPlottingStyles(info):
    if len(info) > 2:
        info = [info]
    if info[0][-1] == 'd':
        ls = ':'
    else:
        ls = '-'
    if info[0] == 'Model A':
        c = 'k'
        zorder = -1
    elif 'Model B' in info[0]:
        c = pr.rainbow[-2]
        zorder = -2
    elif 'Model C' in info[0]:
        c = pr.rainbow[0]
        zorder = -2
    else:
        raise ValueError("Can't read model name.")
    return c, ls, zorder
    
# Move the y-axis ticks and labels to the righthand side.
def makeTicksRight(ax, ylabel=None):
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=270., labelpad=20)
    return
        



#-------- Fits File Functions --------#

# parseFitsName()
# getFiles()
# getProfile() << UPDATES NEEDED.
# calcProfile()

# Parse the .fits file name.
def parseFitsName(filename):
    fn = filename.split('/')[-1]
    fn = fn.split('_')
    model = 'Model %s' % fn[1][0].upper()
    if len(fn[1]) == 2:
        model += fn[1][1]
    mol = fn[2].upper()
    inc = float(fn[3])
    pa = float(fn[4])
    i = 0
    while fn[5][i] != '.':
        i += 1
        if i == len(fn[5]):
            break
    trans = int(fn[5][:i])
    if len(fn) > 6:
        beam = float(fn[6][:-6])
        inttime = float(fn[7][:-4])
    else:
        beam = '%.2f' % (3600. * abs(fits.getval(filename, 'cdelt2', 0)))
        inttime = 0.
    return model, mol, inc, pa, trans, beam, inttime


# Return the filenames and information for .fits files.
def getFiles(molecule=None, beam=None, models=None, inttime=360, mtype='CASA'):

    if mtype == 'CASA':
        directory = 'FitsFiles/CASAOutput/' 
    elif mtype == 'LIME':
        directory = 'FitsFiles/LIMEOutput/'
    else:
        raise ValueError("mtype must be either 'LIME' or 'CASA'.")
        
    # Read in the files.
    files = np.array([directory+fn for fn in os.listdir(directory)])
    infos = np.array([parseFitsName(fn) for fn in files]).T

    # Choose appropriate molecules:
    if molecule is not None:
        files = np.array(files[np.where(infos[1] == molecule)])
        infos = np.array([infos[i][np.where(infos[1] == molecule)] 
                              for i in range(infos.shape[0])])        
    
    # Choose appropriate integration times.
    if (inttime is not None and mtype is not 'LIME'):
        files = np.array(files[np.where(infos[-1] == str(inttime))])
        infos = np.array([infos[i][np.where(infos[-1] == str(inttime))] 
                              for i in range(infos.shape[0])])
        
    # Choose appropriate beam sizes.
    if (beam is not None and mtype is not 'LIME'):
        files = np.array(files[np.where(infos[5] == str(beam))])
        infos = np.array([infos[i][np.where(infos[5] == str(beam))] 
                              for i in range(infos.shape[0])])  
        
    # Choose appropriate models.
    if models is not None:
        if type(models) is tuple:
            models = [model for model in models]
        if type(models) is not list:
            models = [models]
        idxs = np.squeeze([np.where(model in models, True, False) 
                           for model in infos[0]])    
        files = files[idxs]
        infos = np.array([infos[i][idxs] for i in range(infos.shape[0])])
            
    
    return files, infos.T
    

# Return the profile of a zeroth moment.
def getProfile(fn, bins=None, lowchan=0, highchan=-1,
               clip=False, removeCont=1, verbose=False):
    inc = parseFitsName(fn)[2]
    zeroth = getZeroth(fn, lowchan=lowchan, highchan=highchan, clip=clip, 
                       removeCont=removeCont, verbose=verbose)
    axes = getPositionAxes(fn)
    rvals = np.hypot(axes[None,:], axes[:,None]/np.cos(inc))
    return calcProfile(zeroth, rvals, bins=bins)


# Calculate a radial profile of a zeroth moment.
def calcProfile(zeroth, rvals, bins=None):
    zeroth = zeroth.ravel()
    rvals = rvals.ravel()
    if bins is None:
        bins = np.linspace(rvals.min(), rvals.max()/1.41, 50)
    ridxs = np.digitize(rvals, bins)
    avg = np.array([np.nanmean(zeroth[ridxs == r]) 
                    for r in range(1, bins.size)])
    std = np.array([np.nanstd(zeroth[ridxs == r]) 
                    for r in range(1, bins.size)])
    rad = np.average([bins[1:], bins[:-1]], axis=0)
    return rad, avg, std
    
    
# Return the zeroth moment of datacube.
# removeCont is the number of (first) channels to model the continuum with.
def getZeroth(filename, lowchan=0, highchan=-1, clip=False, removeCont=0, 
              getNoise=False, verbose=False, bunit=None, vunit='kms'):
              
    # Read in the data and remove empty axes and channels.
    
    data = np.squeeze(fits.getdata(filename, 0)) 
    data = emptychan(data, verbose=verbose)
    
    # toKelvin or toJansky(area).
    
    if bunit is None:
        bunit = fits.getval(filename, 'bunit', 0)
    bunit = bunit.lower()
    if bunit == 'k':
        unit, area = 'K', None
        data *= toKelvin(filename, verbose=verbose)
    else:
        unit, area = bunit.replace('per', '/').split('/')
        data *= toJansky(filename, outarea=area, verbose=verbose)
        if unit[0] == 'm':
            data *= 1e3
        
    # Remove continuum modelling it as int(removeCont) channels.
    
    if removeCont > 0:
        data = removeContinuum(data, nchan=removeCont)
        
    # Calculate the velocity axis (no offset) in [km/s].
    
    vunit = vunit.replace('/', '').replace('per', '')
    velo = getVelocityAxes(filename)
    if vunit == 'kms':
        velo /= 1e3

    
    # Clip the data with specified channels.
    
    data = data[lowchan:highchan]
    velo = velo[lowchan:highchan]
    
    # Find the noise in the channel with the outside 5 pixels.
    
    rms_chan = np.std([data[:,:5,:5], data[:,-5:,-5:]])
    rms = np.std([np.trapz(data, dx=abs(np.diff(velo)[0]), axis=0)[:5,:5], 
                  np.trapz(data, dx=abs(np.diff(velo)[0]), axis=0)[-5:,-5:]])
                  
     
    if verbose:
        if area is None:
            print 'Channel RMS = %.3e K' % rms_chan 
            print 'Zeroth-moment RMS = %.3e K km/s' % rms
        else:
            print 'Channel RMS = %.3e %s / %s' % (rms_chan, 
                                                  unit.replace('jy', 'Jy'), 
                                                  area)
            print 'Zeroth RMS = %.3e %s %s' % (rms,
                                               unit.replace('jy', 'Jy'),
                                               vunit[:-1]+'/'+vunit[-1])
           
    if clip:
        data = np.where(data < 3.*rms_chan, 0., data)
    zeroth = np.trapz(data, dx=abs(np.diff(velo)[0]), axis=0)
    
    if getNoise:
        return zeroth, rms
    else:
        return zeroth

# Remove empty channels from a datacube.
def emptychan(data, verbose=False):
    i = 0
    j = 0
    if data[0].sum() == 0:
        data = data[1:]
        i += 1
    if data[-1].sum() == 0:
        data = data[:-1]
        j += 1
    if verbose:
        print 'Removed %d channels.' % (i+j)
    return data

    
# Calculate the pixels per beam. If no beam specified, return 1.
def getPixperBeam(filename):
    try:
        getval(filename, 'bmaj')
    except:
        return 1.
    pixarea = abs(np.radians(getval(filename, 'cdelt1', 0))) 
    pixarea *= abs(np.radians(getval(filename, 'cdelt2', 0)))
    beamarea = np.pi * np.radians(getval(filename, 'bmaj', 0))
    beamarea *= np.radians(getval(filename, 'bmin', 0))
    beamarea /= 4. * np.log(2.)
    return pixarea / beamarea
    
    
# Remove the continuum from a dataset.
def removeContinuum(data, nchan=5):
    cont = np.average(data[:nchan], axis=0)
    return np.array([data[i]-cont for i in range(data.shape[0])])
    

# Convert brightness unit to Jy / pix.
def toJansky(filename, outarea='pixel', verbose=False):

    # Define the input and output brightness units and areas.
    
    if outarea == 'pix':
        outarea = 'pixel'
    outunit = 'jy/' + outarea
    if not (outarea == 'pixel' or outarea == 'beam'):
        raise NotImplementedError("outarea must be either 'pixel' or 'beam'.")
    inunit = fits.getval(filename, 'bunit').lower()
    if not inunit in ['k', 'jy/pixel', 'jy/beam']:
        raise ValueError("Unknown input brightness unit.")
    if len(inunit) > 1:
        inarea = inunit[3:]
    else:
        inarea = None

    # If the same, continue as usual.
    if inunit == outunit:
        if verbose:
            print 'No conversion needed.'
        return 1
    if verbose:
        print 'Converting from [%s] to [%s].' % (inunit.replace('jy', 'Jy'),
                                                 outunit.replace('jy', 'Jy'))
    
    # Calculate pixel and beam areas.
    
    pixarea = np.radians(fits.getval(filename, 'cdelt2'))**2
    beamarea = np.pi * np.radians(getval(filename, 'bmaj', 0))
    beamarea *= np.radians(getval(filename, 'bmin', 0))
    beamarea /= 4. * np.log(2.)
    
    # Convert from [K] -> [Jy/area]. 
    # Convert from [Jy/beam] -> [Jy/pixel]
    # Convert from [Jy/pixel] -> [Jy/beam]
    
    if inarea is None:
        scale = 2.266 * 10**26 * sc.k
        if outarea == 'beam':
            scale *= beamarea
        else:
            scale *= pixarea
        try:
            scale *= fits.getval(filename, 'restfreq')**2. / sc.c**2.
        except:
            scale *= fits.getval(filename, 'restfrq')**2. / sc.c**2.
        return scale        
    elif inarea == 'beam':
        return pixarea / beamarea        
    elif inarea == 'pixel':
        return beamarea / pixarea
    else:
        raise ValueError("Unknown inarea value.")

    return
    
def toKelvin(filename, verbose=False):
    raise NotImplementedError("No toKelvin conversion yet.")
    return


# Calculate the integrated flux in Jy/km/s.
def getIntegratedFlux(filename, dx=4., lowchan=0, highchan=-1, clip=True, 
                      removeCont=0):
    zeroth= getZeroth(filename, lowchan, highchan, clip=clip, 
                      removeCont=removeCont)   
    axes = getPositionAxes(filename)  
    imin, imax = abs(axes+dx).argmin(), abs(axes-dx).argmin()
    region = zeroth[imin:imax,imin:imax]
    totalint = np.sum(region) / 1e3
    return totalint


# Return the velocity axes (assuming no offset).
def getVelocityAxes(filename):
    a_len = getval(filename, 'naxis3', 0)
    a_del = getval(filename, 'cdelt3', 0)
    a_pix = getval(filename, 'crpix3', 0)    
    a_ref = getval(filename, 'crval3', 0)
    velax = ((np.arange(1, a_len+1) - a_pix) * a_del)
    return velax
        

# Return the position axes (assuming no offset).
def getPositionAxes(filename, dist=None):
    if dist is None:
        dist = 1.
    a_len = getval(filename, 'naxis2', 0)
    a_del = getval(filename, 'cdelt2', 0) * dist
    a_pix = getval(filename, 'crpix2', 0)    
    a_ref = getval(filename, 'crval2', 0) * 0.0    
    return 3600.*(((np.arange(1, a_len+1) - a_pix) * a_del) + a_ref)


#------ Plotting Routines ------#
#
#   #-- Basic Functions --#
#
#   plotRadialProfiles()
#
#   #-- Helper Functions --#
#
#   texify()
#   sortProfsInfos()
#   addBeam()
#
#-------------------------------#   


# Plot the intensity radial profiles.
def plotRadialProfiles(profs, infos, ax=None, legend=1, c_in=None, ls_in=None,
                         lw_in=1, rescale=None, sort=None, offset=0):
                  
    if profs.ndim == 2:
        profs = np.array([profs])
        infos = np.array([infos])

    if sort is not None:
        profs, infos = sortProfsInfos(profs, infos, sort)
            
    if profs.ndim == 2:
        profs = np.array([profs])
        infos = np.array([infos])
            
    
    # Create a figure if necessary.
    if ax is None:
        fig, ax = plt.subplots()
        
        
    for p, prof in enumerate(profs):
        
        info = infos[p]
        
        # Get the defined styles.
        c, ls, zo = getPlottingStyles(info)
        if c_in is not None:
            c = c_in
        if ls_in is not None:
            ls = ls_in
        
        # Rescale the total by some value.
        if rescale is None:
            rescale = 1    
        
        if sort == 5:
            label = r'%.2f^{\prime\prime}' % float(info[sort])
        else:
            label = info[sort]
        
        ax.step(prof[0], prof[1]*rescale+offset, color=c, linestyle=ls,
                label=texify(label), lw=lw_in, zorder=zo)
    
    if legend:
        ax.legend(fontsize=6, borderaxespad=1.25, labelspacing=.5, 
                  markerfirst=False, handlelength=1.5)

    return
    
# Make strings into LaTeX strings for labels.
def texify(labin):
    lab = r'${\rm %s}$' % labin
    lab = lab.replace(' ', '\,\,')
    return lab

def sortProfsInfos(profs, infos, sort):
    if type(sort) is not int:
        raise TypeError("'sort' must be an integer.")
    else:
        profs = np.squeeze([profs[infos.T[sort] == sortval]
                 for sortval in sorted(np.unique(infos.T[sort]))])
        infos = np.squeeze([infos[infos.T[sort] == sortval]
                 for sortval in sorted(np.unique(infos.T[sort]))])
    return profs, infos


def addBeam(ax, filename, dist=None, verbose=False):
    if dist is None:
        scale = 1.
    else:
        scale = dist 
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((-180*scale/54., -180*scale/54.), 
                         width=getval(filename, 'bmin', 0) * 3600. * scale,
                         height=getval(filename, 'bmaj', 0) * 3600. * scale, 
                         angle=getval(filename, 'bpa', 0),
                         fill=False, hatch='/////////', lw=1., color='w'))
    if verbose:
        print 'bmin = ', getval(filename, 'bmin', 0) * 3600. 
        print 'bmaj = ', getval(filename, 'bmaj', 0) * 3600. 
        print 'bpa = ', getval(filename, 'bpa', 0)
    return
    
