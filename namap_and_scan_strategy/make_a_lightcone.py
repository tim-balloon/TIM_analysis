import os
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as cst
from IPython import embed
# Set plot parameters
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'xtick.direction':'in'})
matplotlib.rcParams.update({'ytick.direction':'in'})
matplotlib.rcParams.update({'xtick.top':True})
matplotlib.rcParams.update({'ytick.right':True})
matplotlib.rcParams.update({'legend.frameon':False})
matplotlib.rcParams.update({'lines.dashed_pattern':[5,3]})
import pickle
from astropy.cosmology import FlatLambdaCDM
# Set cosmology to match Bolshoi-Planck simulation
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307, Ob0=0.048, Tcmb0=2.7255, Neff=3.04)

from simim.lightcone import LCMaker
from simim.lightcone import LCHandler
import simim.instrument as inst
from simim.map import Gridder
from astropy.wcs import WCS


#Rerun simim_tutorial_for_tim.ipynb 
#--- Make a detector
f0 = 1e12              # Hz
channel_width = f0/250 # Hz (4GHz)
spectral_unit = 'Hz'
dish_diameter = 1.9    # m
fwhm = 1.22 * cst.c / (f0*dish_diameter) #rad
spatial_unit = 'rad'
nei = 1e7 # Jy/str/sqrt(s)
nefd = nei * (2*np.pi*(fwhm/2.355)**2)
# Box parameters
df = channel_width #/ 7
da = fwhm / 2

simple_detector = inst.Detector(spatial_response='spec-gauss',
                                spatial_kwargs={'fwhmx':fwhm,'fwhmy':fwhm,'freq0':f0},
                                spatial_unit=spatial_unit,
                                spectral_response='gauss',
                                spectral_kwargs={'freq0':f0,'fwhm':channel_width},
                                spectral_unit=spectral_unit,
                                noise_function='white',
                                noise_kwargs={'rms':nefd})

res = (150 * u.arcsec).to(u.rad).value
# We can now visualize this detector's response functions:
simple_detector.plot_detector_response(fmin=.95e12,fmax=1.05e12,fspatial=1e12,
                              xmin=-res,xmax=res,ymin=-res,ymax=res,)

#---
# Intensity
icii = 10 ** (26 + np.log10(lc1.return_property('lcii_dl', use_all_inds=False)) + np.log10(3.828) + 26 - np.log10(4*np.pi) - 2*np.log10(3.0857) - 2*22 - 2*np.log10(dl) - np.log10(df*da**2)) # Intensity of cell, Jy/str
fcii = 10 ** (26 + np.log10(lc1.return_property('lcii_dl', use_all_inds=False)) + np.log10(3.828) + 26 - np.log10(4*np.pi) - 2*np.log10(3.0857) - 2*22 - 2*np.log10(dl) - np.log10(df)) # Specific flux, Jy (???)

grid = Gridder(pos, fcii, 
               center_point=[0,0,f0], 
               side_length=[lightcone_openangle*np.pi/180,
                        lightcone_openangle*np.pi/180,
                        10*channel_width], 
               pixel_size=[da,da,df],
               axunits = ['rad','rad','Hz'], 
               gridunits=['Jy'])

grid.visualize(plotkws={'vmax':.01})
simple_detector.add_field(grid, 'field0')
simple_detector.map_fields('field0', kernel_size=(200,200),pad=0,spatial_response_norm='peak')

simple_detector.maps['field0'].center_point,simple_detector.maps['field0'].side_length,simple_detector.maps['field0'].pixel_size


simple_detector.maps['field0'].visualize(plotkws={'vmax':.01})
plt.ylim(-0.011,0.011);plt.xlim(-0.011,0.011)

x_series,y_series = az_scan(0.017,0.5) 
position_series = np.stack((x_series,y_series)).T

ts = simple_detector.sample_fields(position_series,'field0',sample_noise=False)
#tsnoise = simple_detector.sample_fields(position_series,'field0',sample_noise=True)
fig, ax = plt.subplots(figsize=(8,4),sharex=True)
fig.subplots_adjust(wspace=.3)
ax.set(xlabel='Sample',ylabel='Flux [Jy]')
ax.plot(ts[:,0],color='k')
plt.show()

### Recreate sample_field 

detector_groups = simple_detector._find_pointing_clones()
map_f = simple_detector.maps['field0']
map_samples = [0 for d in simple_detector.detector_names]
for group in detector_groups:
    idxs = np.sort(np.array([simple_detector.detector_names.index(d) for d in group]))
    offset = simple_detector.detectors[group[0]].pointing_offsets
    #data = map_f.sample(position_series + offset, properties=idxs)
    positions = position_series.copy()
    positions -= map_f.center_point#.reshape(1,map_f.n_dimensions)
    positions += map_f.side_length/2#.reshape(1,map_f.n_dimensions)/2
    positions /= map_f.pixel_size#.reshape(1,map_f.n_dimensions)
    positions = np.floor(positions).astype('int')

    # Find points outside the grid area, store them, and for now set their pixels to 0
    invalid_positions = np.nonzero(np.any(positions<0,axis=1) | np.any(positions/map_f.n_pixels.reshape(1,map_f.n_dimensions)>=1,axis=1))
    positions[invalid_positions] = 0
    properties = np.arange(map_f.n_properties,dtype=int)

    positions_tuple = tuple([tuple(line) for line in positions.T])
    samples = map_f.grid[positions_tuple]
    samples[invalid_positions] = np.nan
    samples = samples[:,properties]
###
regrid = Gridder(position_series, np.array([samples.flatten(),np.ones(len(samples))]).T, 
            pixel_size=[fwhm/1.5,fwhm/1.5],
            axunits = ['rad','rad'], gridunits=['Jy'])
regrid.grid[:,:,0] /= regrid.grid[:,:,1]
regrid.visualize(property=0,axkws={'aspect':'equal'},plotkws={'vmax':.01})
plt.ylim(-0.011,0.011);plt.xlim(-0.011,0.011)


