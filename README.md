# utilities
Tools and functions for projects.


## keplerianmask.py

Use this to build a mask for cleaning in CASA based on the expected Keplerian
rotation of a disk. Basic usage would be,

```python
from utilities.keplerianmask import imagecube
img = imagecube('imagecube.fits')
imagecube.write(inc=10., pa=150., mstar=0.5, vlsr=2.5, rout=4.0)
```

where the inclination and position angle are given in degrees, the stellar mass
is in Msun, the systemic velocity in km/s and rout in arcseconds. The default
output name is `imagecube.mask.fits`, however this can be changed with the
`name` keyword.

To be able to execute this from within CASA, first make sure there is a
pointer to it,

```bash
export CASAPYUTILS="~/path/to/utils/"
```

then include

```python
ip.ex("sys.path.append(os.environ['CASAPYUTILS'])")
```

into the `ipy_user_conf.py` which can be found through,

```python
os.environ.get('CASAPATH').split()[0]
```

in CASA.
