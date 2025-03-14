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
# Set plot parameters

import pygetdata as gd
import numpy as np
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
import pandas as pd

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

def load_params(path):
    """
    Return as a dictionary the parameters stores in a .par file
    
    Parameters
    ----------
    path: string
        name of the .par file       
    Returns
    -------
    params: dictionary
        dictionary containing the loaded parameters
    """    
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

def gen_tod_one_array(simu_sky_path, pixel_offset,pointing_paths, scan_path_sky, ra, dec, spf, tod_file, P,T, plot=True):

    """
    Generate the tod for one array of TIM detectors and save it in the .hdf5 format. 

    Parameters
    ----------
    simu_sky_path: string
        path to the sky simulation out of which to generate the timestreams
    pixel_offset: array
        pixel position on the array wrt the central pixel, in degrees
    pointing_paths: 2d array
        coordinates of the scan path, for the central pixe, in degrees
    ra: float
        the ra coordinate of the field center
    dec: float
        the dec coordinate of the field center

    Returns
    -------
    H: hdf5
        save the output hdf5 file.
    """ 

    #Generate the TOD given a simulated sky
    cube = fits.getdata(simu_sky_path)
    cubemean = np.mean(cube, axis=(1,2))
    cube -= cubemean[:, None, None]
    hdr  = fits.getheader(simu_sky_path)
    hdr['CRVAL1'] = - (hdr['NAXIS1'] / 2) * hdr['CDELT1']  # Shift RA reference '''to center
    hdr['CRVAL2'] =  dec - (hdr['NAXIS2'] / 2) * hdr['CDELT2']  # Shift DEC reference
    hdr['CRPIX1'] = 0
    hdr['CRPIX2'] = 0

    xbins = np.arange(-0.5, hdr['NAXIS1']+0.5, 1)
    ybins = np.arange(-0.5, hdr['NAXIS2']+0.5, 1)
    map = cube[0,:,:]

    wcs = WCS(hdr, naxis=2) 
    
    positions_x = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    positions_y = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    samples = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))

    for detector, offset in enumerate(pixel_offset):
        
        #data = map_f.sample(position_series + offset, properties=idxs)
        y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_paths[detector][:,0], pointing_paths[detector][:,1])    
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
    img = axs[1].imshow(map, interpolation='nearest', origin='lower', vmin=map.min(), vmax=map.max(), cmap='cividis' )
    count = axs[2].imshow(norm, interpolation='nearest', origin='lower', cmap='binary' )
    
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
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    det_names = det_names_dict['Name']
    H = h5py.File(tod_file, "w")

    for i, (name, coord) in enumerate(zip(('RA', 'DEC'), (scan_path_sky[:,0],scan_path_sky[:,1]))):
        grp = H.create_group(name)
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)

    for detector, (offset, name, pointing) in enumerate(zip(pixel_offset, det_names, pointing_paths)):
        grp = H.create_group(f'kid{name}_roach')
        grp.create_dataset('data', data=samples[detector,:], compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('pixel_offset', data=pixel_offset)

        grp = H.create_group(f'kid{name}_RA')    
        grp.create_dataset('data', data=pointing[:,0])

        grp = H.create_group(f'kid{name}_DEC')    
        grp.create_dataset('data', data=pointing[:,1])

        for comp in ('I', 'Q'): 
            grp = H.create_group(f'{comp}_kid{name}_roach') #conversion factor = 1 Jy/Hz
            grp.create_dataset('data', data=samples[detector,:] / np.sqrt(2), compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=spf)
            grp.create_dataset('pixel_offset', data=pixel_offset)
        
    grp = H.create_group('time')
    grp.create_dataset('data', data=T, compression='gzip', compression_opts=9)
    grp.create_dataset('spf', data=spf)
    H.close()

    d = {'wcs':wcs,
         'xbins':xbins,
         'ybins':ybins}
    pickle.dump(d, open('wcs.p', 'wb'))

    '''
    embed()
    
    H = h5py.File(tod_file, "a")
    positions_x = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    positions_y = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    samples = np.zeros((len(pixel_offset), len(pointing_paths[0][:,0])))
    num_frames = 3500
    first_frame=0
    for detector, (offset, name, pointing) in enumerate(zip(pixel_offset, det_names, pointing_paths)):
        #data = map_f.sample(position_series + offset, properties=idxs)
        pointing_paths_X = H['kid'+name+'_RA']['data'][()]
        pointing_paths_Y = H['kid'+name+'_DEC']['data'][()]
        # Round the positions and convert to integer 
        values = H[f'kid{name}_roach']['data'][()]
        y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(pointing_paths_X[first_frame*spf:(first_frame+num_frames)*spf], pointing_paths_Y[first_frame*spf:(first_frame+num_frames)*spf])    

        # Assign values from the map for valid positions
        samples[detector,:] = np.asarray(values.astype(float))[first_frame*spf:(first_frame+num_frames)*spf]
        positions_x[detector,:] = x_pixel_coords
        positions_y[detector,:] = y_pixel_coords
    H.close()


    if(plot):
        norm, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins),  )
        hist, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()),  bins=(xbins,ybins), weights=samples.ravel())
        hist /= norm

        fig, axs = plt.subplots(1,3, figsize=(12,4), dpi = 200,subplot_kw={'projection': wcs}, sharex=True, sharey=True )
        imgdec = axs[0].imshow(hist, interpolation='nearest', origin='lower', vmin=map.min(), vmax=map.max(), cmap='cividis' )
        img = axs[1].imshow(map, interpolation='nearest', origin='lower', vmin=map.min(), vmax=map.max(), cmap='cividis' )
        count = axs[2].imshow(norm, interpolation='nearest', origin='lower', cmap='binary' )

        for ax in (axs[0], axs[1], axs[2]):
            lon = ax.coords[0]
            lat = ax.coords[1]
            lat.set_major_formatter('d.d')
            lon.set_major_formatter('d.d')
            lon.set_axislabel('RA')
            lat.set_axislabel('Dec')
            if(ax is not axs[0]): ax.tick_params(axis='y', labelleft=False)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.title('try')
        plt.show()
    '''

    return positions_y, positions_x, samples


    """
    Generate the tod for one array of TIM detectors and save it in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file     
    scan_path_sky: nd array
        coordinates of the pointing path of each detector
    samples: 2d array
        TOD amplitudes of each detector
    pixel_offset: array
        position of each pixel on the array. 
    T: array
        hour angle in degree

    Returns
    -------
    H: hdf5
        save the output hdf5 file.
    """ 


if __name__ == "__main__":
    '''
    To run: python strategy.py params_strategy.par
    Left to be done:
        Add the second array of detectors
        Produce spectral TOD (One frequency so far)
        Compute mapping efficiency
        add noise and atmosphere in TOD
        make the simIM version
        LST LAT timestreams
        AZ and EL coord timestreams, and test the namap related steps
        implement noise peak / cosmic ray makerSo far in namap, noise is loaded. Should be computed from tod for inverse varriance weighting
    '''

    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()
    P = load_params(args.params)
    #------------------------------------------------------------------------------------------
    #Initiate the parameters
    
    #The coordinates of the field
    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = c.ra.value
    dec = c.dec.value
    #The contour of the field
    contours = P['contours']
    x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])
    lat = P['latittude']
    f_range = P['f_range']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value

    #Pixel offsets from the center of the field of view in degree. 
    pixel_offset_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW']) 
    pixel_offset_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW']) 
    pixel_offset = pixel_offset_HW

    theta = np.radians(P['theta'])
    
    #----------------------------------------
    #Generate the sacan path
    T_duration = P['T_duration'] 
    dt = P['dt']*np.pi/3.14 
    T = np.arange(0,T_duration,dt) *3600
    HA = 15*np.arange(-T_duration/2,T_duration/2,dt)  # hours angle

    if(P['scan']=='loop'): 
        az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=P['dt_scan'])
        scan_path, scan_flag = genScanPath(T, alt, az, flag)
        scan_path = scan_path[scan_flag==1] 
        T_trim = T[scan_flag==1]
        HA_trim = HA[scan_flag==1]
    elif(P['scan']=='raster'):
        az, alt = az_scan_custom(np.radians(P['stripe_size']),np.radians(P['dy']),P['nb_rows'])
        az = np.degrees(az); alt=np.degrees(alt)
        scan_path, scan_flag = genScanPath(T, alt, az, np.ones(len(alt)))
        T_trim = T
        HA_trim = HA

    scan_path_sky = genPointingPath(T_trim, scan_path, HA_trim, lat, dec) #az, alt
    pixel_paths  = genPixelPath(scan_path, pixel_offset, theta)
    pointing_paths = [genPointingPath(T_trim, pixel_path, HA_trim, lat, dec) for pixel_path in pixel_paths]
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
    #Generate the TOds and save it in hdf5
    spf = int(1/(dt*3600)) #Hz
    simu_sky_path = P['path']+P['file']
    tod_file=P['path']+'TOD_'+P['file'][:-5]+'.hdf5'

    positions_y, positions_x, samples = gen_tod_one_array(simu_sky_path, pixel_offset,pointing_paths, scan_path_sky, ra, dec, spf, tod_file, P,T_trim)

    #----------------------------------------
