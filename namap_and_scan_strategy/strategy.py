import numpy as np
import scipy as sp
import matplotlib.colors as colors
from matplotlib.pyplot import cm

from scipy.integrate import simps
from time import time 
import matplotlib.animation as animation
from scan_fcts import * 
from TIM_scan_strategy import *
import importlib
from astropy.coordinates import SkyCoord
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import os
import astropy.units as u
import matplotlib
import scipy.constants as cst
from IPython import embed
# Set plot parameters

import pygetdata as gd
import numpy as np
from IPython import embed
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import h5py
import os
import tracemalloc
import time
from progress.bar import Bar

import argparse

'''
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'xtick.direction':'in'})
matplotlib.rcParams.update({'ytick.direction':'in'})
matplotlib.rcParams.update({'xtick.top':True})
matplotlib.rcParams.update({'ytick.right':True})
matplotlib.rcParams.update({'legend.frameon':False})
matplotlib.rcParams.update({'lines.dashed_pattern':[5,3]})
'''
import pickle
from astropy.cosmology import FlatLambdaCDM
# Set cosmology to match Bolshoi-Planck simulation
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307, Ob0=0.048, Tcmb0=2.7255, Neff=3.04)
from astropy.wcs import WCS
from astropy.coordinates import Angle

from simim.lightcone import LCMaker
from simim.lightcone import LCHandler
import simim.instrument as inst
from simim.map import Gridder
from astropy. io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_params(path, force_pysides_path = ''):
    '''
    #Load the parameter file as a dictionnary
    '''
    file = open(path)

    params = {}
    for line in file:
        line = line.strip()
        if not line.startswith("#"):
            no_comment = line.split('#')[0]
            key_value = no_comment.split("=")
            if len(key_value) == 2:
                params[key_value[0].strip()] = key_value[1].strip()

    for key in params.keys():
        params[key] = eval(params[key])

    return params

if __name__ == "__main__":
    '''
    To run: python strat.py params_strategy.par
    Left to be done:
    Add the second array of detectors
    Produce spectral TOD
    Compute mapping efficiency
    Get the right acquisition rate
    Make cst elevation scan turn instead of vertical ascenscion 
    set properly the spf 
    add noise in TOD
    make the simIM version
    '''
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options

    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()

    P = load_params(args.params)

    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = c.ra.value
    dec = c.dec.value

    contours = P['contours']
    x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])
    lat = P['latittude']

    f_range = P['f_range']

    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value

    #Scan parameters
    stripe_size = P['stripe_size']
    dy = P['dy']
    nb_rows = P['nb_rows']
    T_duration = P['T_duration'] * 3600 #second angle
    dt = P['dt']*np.pi/3.14 #second angle, such as dt is not rational. 

    theta = np.radians(P['theta'])

    pixel_offset_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW']) 
    pixel_offset_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW']) 
    pixel_offset = pixel_offset_HW

    #----------------------------------------
    T = np.arange(0,T_duration,dt)
    HA = np.arange(-T_duration/2,T_duration/2,dt) / 3600 # hours angle
    az, alt = az_scan_custom(np.radians(stripe_size),np.radians(dy),nb_rows)
    az = np.degrees(az); alt=np.degrees(alt)
    scan_path, _  = genScanPath(T, alt, az, np.zeros(len(alt)))   
    scan_path_sky = genPointingPath(T, scan_path, HA, lat, dec)
    pixel_paths  = genPixelPath(scan_path, pixel_offset, theta)
    pointing_paths = [genPointingPath(T, pixel_path, HA, lat, dec) for pixel_path in pixel_paths]
    xedges,yedges,hit_map = binMap(pointing_paths,res=res,f_range=f_range,dec=dec,)
    #----------------------------------------
    #scan route
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
    fig, axs = plt.subplots(2,2,figsize=(10,10), dpi=160,)# sharey=True, sharex=True)
    axradec, ax, axr, axp = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'cyan')
    if(contours is not None): axradec.plot(contours[:, 1]-ra, contours[:, 0]-dec, c='g' )
    axradec.set_xlabel('Az [deg]')
    axradec.set_ylabel('El [deg]')
    axradec.set_aspect(aspect=1)
    if(contours is not None): heatmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0]+contours[:, 1].mean(), scan_path_sky[:,1], bins=int(2/res))
    else: heatmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(heatmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    if(contours is not None): ax.plot(contours[:, 1], contours[:, 0], c='g' )

    img = axr.imshow((hit_map), extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], 
                    interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hit_map), cmap='binary' )
    if(contours is not None): axr.plot(contours[:, 1], contours[:, 0], c= 'r')
    fig.colorbar(img, ax=axr, label='Counts')
    CS = axr.contour(np.roll(np.sqrt(hit_map*np.roll(hit_map, 4, axis=1)),-2,axis=1), levels=[0.5*np.max(hit_map)], lw=0.5, origin='lower', linewidths=0.6,
                extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], colors='magenta')
    # CS = ax1.contour(hit_map, levels=[0.5*np.max(hit_map)], lw=0.5, origin='lower', linewidths=0.6,\
    #             extent=[x_cen-maps, x_cen+maps,y_cen-maps, y_cen+maps,], colors='w')
    ax.clabel(CS, inline=1, fontsize=8, )
    #pixel_xy = np.array([-1*pixel_offset*np.sin(theta)-0.036*np.cos(theta), pixel_offset*np.cos(theta)-0.036*np.sin(theta)])
    pixel_xy = np.array([-1*pixel_offset*np.sin(theta)+0.036*np.cos(theta), pixel_offset*np.cos(theta)+0.072*np.sin(theta)])
    axr.scatter(pixel_xy[0]+x_cen, pixel_xy[1]+y_cen, s=2, marker='+', c='white',label='SW Spatial Pixels')
    patchs = []
    axp.set_title('Pointing Path per pixel')
    idx = np.arange(len(pixel_paths))[::10]
    n = 11
    for i,c in zip(idx, cm.rainbow(np.linspace(0.,1,len(idx)))):
        axp.scatter(pointing_paths[i][::n,0]+ra, pointing_paths[i][::n,1], s=0.1,c=c)
        patch = mpatches.Patch(color=c, label= 'pixel %d'%i); patchs.append(patch); 
    axp.set_aspect(aspect=1)
    axp.legend(handles=patchs,frameon=False, bbox_to_anchor=(1,1))
    fig.tight_layout()
    plt.savefig("scan_route.png")
    #----------------------------------------
    cube = fits.getdata(P['path']+P['file'])
    hdr  = fits.getheader(P['path']+P['file'])
    hdr['CRVAL1']=ra
    hdr['CRVAL2']=dec
    xbins = np.arange(-0.5, hdr['NAXIS1']+0.5, 1)
    ybins = np.arange(-0.5, hdr['NAXIS2']+0.5, 1)
    map = cube[0,:,:]
    #----------------------------------------
    wcs = WCS(hdr, naxis=2) 

    positions_x = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    positions_y = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    samples = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))

    for detector, offset in enumerate(pixel_offset):
        #data = map_f.sample(position_series + offset, properties=idxs)
        y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_paths[detector][:,0]+hdr['CRVAL1'], pointing_paths[detector][:,1])    
        # Round the positions and convert to integer indices
        x_pixel_coords_rounded = np.round(x_pixel_coords).astype(int)
        y_pixel_coords_rounded = np.round(y_pixel_coords).astype(int)
        # Create a mask for positions within bounds
        valid_mask = (
            (x_pixel_coords_rounded >= 0) & (x_pixel_coords_rounded < hdr['NAXIS1'] - 1) &  # x within bounds
            (y_pixel_coords_rounded >= 0) & (y_pixel_coords_rounded < hdr['NAXIS2'] - 1) )   # y within bounds
        # Initialize the output array with zeros
        values = np.zeros_like(x_pixel_coords_rounded, dtype=float)
        # Assign values from the map for valid positions
        values[valid_mask] = map[x_pixel_coords_rounded[valid_mask], y_pixel_coords_rounded[valid_mask]]
        samples[detector,:] = np.asarray(values.astype(float))
        positions_x[detector,:] = x_pixel_coords
        positions_y[detector,:] = y_pixel_coords

    norm, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins),  )
    hist, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()),  bins=(xbins,ybins), weights=samples.ravel())
    hist /= norm

    fig, axs = plt.subplots(1,3, figsize=(12,4), dpi = 200,subplot_kw={'projection': wcs}, sharex=True, sharey=True )
    imgdec = axs[0].imshow(hist, interpolation='nearest', origin='lower', vmin=map.min(), vmax=map.max(), cmap='cividis' )
    img = axs[1].imshow(map, #extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], 
                        interpolation='nearest', origin='lower', vmin=map.min(), vmax=map.max(), cmap='cividis' )
    count = axs[2].imshow(norm, #extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], 
                        interpolation='nearest', origin='lower', cmap='binary' )

    for ax in (axs[0], axs[1], axs[2]):
        lon = ax.coords[0]
        lat = ax.coords[1]
        lat.set_major_formatter('d.d')
        lon.set_major_formatter('d.d')
        lon.set_axislabel('RA')
        lat.set_axislabel('Dec')
        if(ax is not axs[0]): ax.tick_params(axis='y', labelleft=False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    #----------------------------------------
    #TOD generator in hdf5

    H = h5py.File(P['path']+'TOD_'+P['file'][:-5]+'.hdf5', "w")

    spf = 3 # How to set properly ?? 

    for i, (name, coord) in enumerate(zip(('RA', 'DEC'), (scan_path_sky[:,0],scan_path_sky[:,1]))):
        grp = H.create_group(name)
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)

    for detector, offset in enumerate(pixel_offset):
        grp = H.create_group(f'kid{detector}_roachN')
        grp.create_dataset('data', data=samples[detector,:], compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('pixel_offset', data=pixel_offset)

    grp = H.create_group('time')
    grp.create_dataset('data', data=T, compression='gzip', compression_opts=9)
    grp.create_dataset('spf', data=spf)

    H.close()