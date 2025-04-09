import numpy as np
import scipy as sp
import matplotlib.colors as colors
from matplotlib.pyplot import cm

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import time
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
import sim_tools_tod
import sim_tools_flatsky
import sys

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
from progress.bar import Bar

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

def gen_tod(wcs, Map, ybins, xbins, pixel_offset, pointing_paths):

    """
    Generate the tod for one array of TIM detectors. 

    Parameters
    ----------
    wcs: astropy.wcs.wcs.WCS
        The wcs used to generate the TODs
    Map: 2d array
        the 2d angular map of the sky in a frequency channel
    ybins: array
        edges of the pixels
    xbins: array
        edges of the pixels
    pixel_offset: array
        pixel position on the array wrt the central pixel, in degrees
    pointing_paths: list of 2d array
        coordinates of the sky scan path of each pixel.
    Returns
    -------
    wcs: astropy.wcs.wcs.WCS
        The wcs used to generate the TODs
    map: 2d array
        the sky map used to generate the amplitude TODs
    hist: 2d array
        the reconstructed sky map given the pointing paths 
    norm: 2d array
        the hitmap
    samples: list
        list of the amplitude timestreams of each detectors
    positions_x: list
        list of RA coordinates timestreams of each detectors
    positions_y: list
        list of DEC coordinates timestreams of each detectors 
    """ 
    
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
        values[valid_mask] = Map[x_pixel_coords_rounded[valid_mask], y_pixel_coords_rounded[valid_mask]]
        samples[detector,:] = np.asarray(values.astype(float))
        positions_x[detector,:] = x_pixel_coords
        positions_y[detector,:] = y_pixel_coords

    norm, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins),  )
    hist, edges = np.histogramdd(sample=(positions_x.ravel(), positions_y.ravel()), bins=(xbins,ybins), weights=samples.ravel())

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)  # Catch all runtime warnings
        hist /= norm  # Perform the division

    return wcs, Map, hist, norm, samples, positions_x, positions_y

def save_scan_path(tod_file, scan_path, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    scan_path_sky: 2d array
        (ra, dec) coordinates timestreams of the center pixel
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('RA', 'DEC'), (scan_path_sky[:,0],scan_path_sky[:,1]))):
        namegrp = name
        if namegrp not in H:
            grp = H.create_group(namegrp)
            grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=spf)
    H.close() 

def save_time_tod(tod_file, T, spf):
    '''
    Save the time tod in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    T: array
        time timestreams
    spf: int
        the number of samples per frame
    Returns
    -------
    '''
    H = h5py.File(tod_file, "a")
    namegrp = f'time'
    if namegrp not in H:
        grp = H.create_group(namegrp)
        grp.create_dataset('data', data=T, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close()

def save_tod_in_hdf5(tod_file, det_names, samples, pixel_offset, pixel_shift, pointing_paths, dect_file, F, spf):
    """
    Save the tod for one array of TIM detectors in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file   
    det_names: list
        list of names for the detectors, same lenght as pixel_offset
    samples: list
        list of amplitude timestreams.  
    pixel_offset: array
        vertical position of each pixel on the array with respect to the center 
    pixel_shift array
        horizontal position of each pixel on the array with respect to the center 
    pointing_paths: list of 2d array
        coordinates of the sky scan path of each pixel.
    spf: int
        the number of samples per frame

    Returns
    -------
    """ 
    
    H = h5py.File(tod_file, "a")

    for detector, (offset, shift, name, pointing) in enumerate(zip(pixel_offset, pixel_shift, det_names, pointing_paths)):
        namegrp = f'kid{name}_roach'
        if namegrp not in H:
            grp = H.create_group(namegrp)
            grp.create_dataset('data', data=samples[detector,:], compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=spf)
            grp.create_dataset('pixel_offset_y', data=offset)
            grp.create_dataset('pixel_offset_x', data=shift)
            grp.create_dataset('frequency', data=shift)

        namegrp = f'kid{name}_RA'
        if namegrp not in H:
            grp = H.create_group(namegrp)    
            grp.create_dataset('data', data=pointing[:,0])
            grp.create_dataset('spf', data=spf)
        namegrp = f'kid{name}_DEC'
        if namegrp not in H:
            grp = H.create_group(namegrp)    
            grp.create_dataset('data', data=pointing[:,1])
            grp.create_dataset('spf', data=spf)

    H.close()

    #Finally, update the detectors file with the central frequency of the detectors
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    mask = det_names_dict["Name"].isin(det_names)
    det_names_dict.loc[mask, 'Frequency'] = F.value   
    det_names_dict.to_csv(P['detectors_name_file'], sep='\t', index=False)

def save_lst_lat(tod_file, lst, lat, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    lst: array
        the local sideral time timestream, for the center of the array. 
    lat: array 
        the lattitude timestream, for the center of the array. 
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('lst', 'lat'), (lst,lat))):
        namegrp = name
        if namegrp not in H:
            grp = H.create_group(namegrp)
            grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=spf)
    H.close() 

def save_az_el(tod_file, azimuths, elevations, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    azimuths: array
        the azimuth timestream of the centre of the array
    elevations: array 
        the lattitude timestream of the centre of the array
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('azimuth', 'elevation'), (azimuths,elevations))):
        namegrp = name
        if namegrp not in H:
            grp = H.create_group(namegrp)
            grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
            grp.create_dataset('spf', data=spf)
    H.close() 

if __name__ == "__main__":
    '''
    Instructions: 

    1st: git clone from TIM_analysis/namap, extragalactic_sky_tod_maker branch or main branch if it exist. 

    2nd: Download a mock sky: scp yournetid@cc-login.campuscluster.illinois.edu:/projects/ncsa/caps/TIM_analysis/sides_angular_cubes/TIM/pySIDES_from_uchuu_tile_0_1.414deg_x_1.414deg_fir_lines_res20arcsec_dnu4.0GHz_full_de_Looze_smoothed_MJy_sr.fits
    and put it in /fits_and_hdf5/
    
    3rd: generate the KIDs file with python gen_det_names.py params_strategy.par

    To run: python strategy.py params_strategy.par

    Left to be done:
        Compute mapping efficiency
        add noise and atmosphere in TOD
        make the simIM version
        LST LAT timestreams, AZ and EL coord timestreams, and test the namap related steps
        implement noise peak / cosmic ray maker. 
        Problem in HA to be fixed. 
        Make sure that TOD are generated with right periode. 
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

    #get params for sim generation

    #sim specs
    nsims = P['nsim_noise']

    #define some detectors
    total_detectors = 2*P["nb_channels_per_array"]
    detector_array = np.arange( total_detectors )

    #The coordinates of the field
    name=P['name_field']
    c=SkyCoord.from_name(name)
    ra = c.ra.value
    dec = c.dec.value
    #The contour of the field
    contours = P['contours']
    x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])

    #load the observer position
    lat = P['latittude']
    lon = P['longitude']# degrees
    mcmurdo = EarthLocation(lat=f'{lat}', lon=f'{lon}', height=37000.)

    #Plot parameter
    f_range = P['f_range']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value

    #Pixel offsets from the center of the field of view in degree. 
    pixel_offset_HW, pixel_shift_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW'], -P['arrays_separation']/2)
    pixel_offset_LW, pixel_shift_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW'], P['arrays_separation']/2) 
    pixel_offset = np.concatenate((pixel_offset_HW, pixel_offset_LW))
    pixel_shift  = np.concatenate((pixel_shift_HW, pixel_shift_LW)) 

    #Angle of the rotation to apply to the detector array. 
    theta = np.radians(P['theta'])

    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    T = np.arange(0,T_duration,dt) * 3600
    HA = 15*np.arange(-T_duration/2,T_duration/2,dt)  # hours angle
    #load the observing date to generate the local sideral time (lst) coordinates. 
    start_time = Time(P['launch_date'], scale="utc")
    #----------------------------------------
    #Generate the scan path for the center of the arrays. 
    if(P['scan']=='loop'):   az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='raster'): az, alt, flag = genLocalPath_cst_el_scan(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='zigzag'): az, alt, flag = genLocalPath_cst_el_scan_zigzag(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    scan_path, scan_flag = genScanPath(T, alt, az, flag)
    scan_path = scan_path #[scan_flag==1] Use the scan flag to keep only constant scan speed part of the pointing. 
    T_trim = T            #[scan_flag==1]
    HA_trim = HA          #[scan_flag==1]
    
    #Generate the pointing on the sky for the center of the arrays
    scan_path_sky = genPointingPath(T_trim, scan_path, HA_trim, lat, dec, ra) 

    #Generate the scan path of each pixel, as a function of their offset to the center of the arrays. 
    pixel_paths  = genPixelPath(scan_path, pixel_offset, pixel_shift, theta)

    #Generate the pointing on the sky of each pixel. 
    pointing_paths = [genPointingPath(T_trim, pixel_path, HA_trim, lat, dec, ra) for pixel_path in pixel_paths]

    #Generate the hitmap, using all the detectors. 
    xedges,yedges,hit_map = binMap(pointing_paths,res=res,f_range=f_range,dec=dec,ra=ra) 

    #----------------------------------------
    #Plot a scan route
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
    fig, axs = plt.subplots(2,2,figsize=(10,10), dpi=160,)# sharey=True, sharex=True)
    #---
    axradec, ax, axr, axp = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'cyan')
    if(contours is not None): axradec.plot(contours[:, 1]-ra, contours[:, 0]-dec, c='g' )
    axradec.set_xlabel('Az [deg]')
    axradec.set_ylabel('El [deg]')
    axradec.set_aspect(aspect=1)
    #---
    heatmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(heatmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    if(contours is not None): ax.plot(contours[:, 1], contours[:, 0], c='g' )
    #---
    img = axr.imshow((hit_map), extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], 
                    interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hit_map), cmap='binary' )
    if(contours is not None): axr.plot(contours[:, 1], contours[:, 0], c= 'r')
    fig.colorbar(img, ax=axr, label='Counts')
    CS = axr.contour(np.roll(np.sqrt(hit_map*np.roll(hit_map, 4, axis=1)),-2,axis=1), levels=[0.5*np.max(hit_map)], lw=0.5, origin='lower', linewidths=0.6,
                extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], colors='magenta')
    axr.clabel(CS, inline=1, fontsize=8, )

    patchs = []
    axp.set_title('Pointing Path per pixel')
    idx = np.arange(len(pixel_paths))[::10]
    n = 11
    for i,c in zip(idx, cm.rainbow(np.linspace(0.,1,len(idx)))):
        axp.scatter(pointing_paths[i][::n,0], pointing_paths[i][::n,1], s=0.1,c=c)
        patch = mpatches.Patch(color=c, label= 'pixel %d'%i); patchs.append(patch); 
    axp.set_aspect(aspect=1)
    axp.legend(handles=patchs,frameon=False, bbox_to_anchor=(1,1))
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plot/'+f"scan_route_{P['scan']}.png")
    plt.close()
    #----------------------------------------
    #Generate the TODs and save Them in hdf5
    spf = int(1/(dt*3600)) #sample per frame defined here as the acquisition rate in Hz. 
    #The path to the sky simulation from which to generate the TODs from
    simu_sky_path =os.getcwd()+'/'+P['path']+P['file']
    #The output hdf5 file containing the generated TODs. 
    tod_file=os.getcwd()+'/'+P['path']+'TOD_'+P['file'][:-5]+'.hdf5'
    #Load the names of the detectors. We assigned to them random names so that we cannot do naive for loop and avoid mistakes.
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    det_names = det_names_dict['Name']
    #----------------------------------------
    #Load the sky simulation to generate the TOD from
    hdr  = fits.getheader(simu_sky_path)
    pix_size = ((hdr['CDELT1']*u.Unit(hdr['CUNIT1']))**2).to(u.sr).value
    hdr['CRVAL1'] = ra 
    hdr['CRVAL2'] = dec
    hdr['CRPIX1'] = hdr['NAXIS1']//2
    hdr['CRPIX2'] = hdr['NAXIS2']//2
    #Create the list of frequency channels of the simulated cube. 
    freqs =( np.arange(hdr['CRVAL3'], hdr['CRVAL3']+hdr['NAXIS3']*hdr['CDELT3'], hdr['CDELT3'])*u.Unit(hdr['CUNIT3']) ).to(u.GHz)
    #Create the binning of the map in pixel coordinates. 
    xbins = np.arange(-0.5, hdr['NAXIS1']+0.5, 1)
    ybins = np.arange(-0.5, hdr['NAXIS2']+0.5, 1)
    #load the angular spectral cube. 
    cube = fits.getdata(simu_sky_path)
    #Remove the mean in each map, to wich we are not sensitive. 
    cubemean = np.mean(cube, axis=(1,2)) 
    cube -= cubemean[:, None, None]
    cube *= pix_size * 1e6 #conversion MJy/sr to Jy/beam
    #Create the world coordinate object. 
    wcs = WCS(hdr, naxis=2) 
    #----------------------------------------
    #Local Sideral Time: 
    # Define the observation start and end time
    print('Generate the lst and lat coordinates')
    start = time.time()
    T_lst = T[::300]/3600 #1Hz sampling
    times = start_time + T_lst * u.h
    scan_sky_trim = scan_path_sky[::300,:]
    # Convert each RA/Dec to Alt/Az at the corresponding time
    altaz_frame = AltAz(obstime=times, location=mcmurdo)
    skycoords = SkyCoord(ra=scan_sky_trim[:,0]*u.deg, dec=scan_sky_trim[:,1]*u.deg, frame="icrs")
    altaz_coords = skycoords.transform_to(altaz_frame)
    # Extract Azimuth and Elevation
    azimuths = altaz_coords.az
    elevations = altaz_coords.alt
    # Compute the Local Sidereal Time (LST) for each time
    lst = [time.sidereal_time("apparent", longitude=lon).value for time in times]
    lat = np.ones(len(lst)) * lat
    print(f"Generate the lst and lat in {np.round((time.time() - start),2)}"+'s')
    save_lst_lat(tod_file, lst, lat, 1)
    save_az_el(tod_file, azimuths, elevations, 1)
    #----------------------------------------------------------------------------  
    
    #example noise model for auto- and cross-power between detectors
    freq, noise_powspec, noise_powspec_one_over_f, noise_powspec_white = sim_tools_tod.detector_noise_model(P["tod_noise_level"], P["fknee"], P["alphaknee"], len(T_trim), 1/(dt*3600))
    cross_noise_powspec = sim_tools_tod.get_correlated_powspec(P["rho_one_over_f"], noise_powspec_one_over_f, noise_powspec_one_over_f)
    noise_powspec_dic = {}
    for i in range(total_detectors):
        for j in range(total_detectors):
            if i == j:  noise_powspec_dic[i, j] = noise_powspec
            else:       noise_powspec_dic[i, j] = cross_noise_powspec

    fsval = 14
    ax = plt.subplot(111, yscale = 'log', xscale = 'log')
    plt.plot( freq, noise_powspec, label = r'Total', color = 'black' )
    plt.plot( freq, noise_powspec_one_over_f, label = r'$1/f$', color = 'orangered' )
    plt.plot( freq, noise_powspec_white, label = r'White', color = 'darkgreen' )
    plt.xlabel(r'Frequency [Hz]', fontsize = fsval)
    plt.ylabel(r'Noise power NET ', fontsize = fsval)
    plt.grid(True, lw = 0.2, alpha = 0.2, which = 'both')
    plt.ylim(1e-1, 1e11)
    plt.legend(loc = 1, fontsize = fsval - 2)
    plt.show()

    '''
    tod_sims_dic = {}
    pspec_dic_sims = {}
    for sim_no in range( nsims ):
        print('Sim = %s of %s' %(sim_no+1, nsims))
        tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq, noise_powspec_dic, (1,len(T_trim)), (dt*3600))

        #get the sim spectra now.
        curr_sim_pspec_dic = {}
        for (cntr1, tod1) in enumerate( tod_sim_arr ):
            for (cntr2, tod2) in enumerate( tod_sim_arr ):
                if cntr2<cntr1: continue
                    
                if (0):
                    ax = plt.subplot(111)
                    plt.plot( tod1 )
                    plt.show(); sys.exit()
                
                freq = np.fft.fftfreq(len(T_trim), (dt*3600)) #TOD frequencies.
                curr_spec = ( np.fft.fft(tod1) * (dt*3600) * np.conj( np.fft.fft(tod2) * (dt*3600) ) / len(T_trim)  ).real

                if (0):
                    inds = np.where(freq>0)
                    curr_noise_powspec = noise_powspec_dic[(cntr1, cntr2)]
                    fsval = 14
                    ax = plt.subplot(111, yscale = 'log', xscale = 'log')
                    plt.plot( freq[inds], curr_noise_powspec[inds], label = r'Input', color = 'black' )
                    plt.plot( freq[inds], curr_spec[inds], color = 'orangered', label = r'Sims' )
                    plt.xlabel(r'Freqeuency [Hz]', fontsize = fsval)
                    plt.ylabel(r'Noise power NET $\mu$K$^{2}$ seconds', fontsize = fsval)
                    plt.grid(True, lw = 0.2, alpha = 0.2, which = 'both')
                    #ylim(5e4, 1e8); 
                    plt.xlim(1e-2, 1e2)
                    plt.legend(loc = 1, fontsize = fsval - 2)
                    plt.show(); ##sys.exit()
                    
                curr_sim_pspec_dic[(cntr1, cntr2)] = [freq, curr_spec]
        pspec_dic_sims[sim_no] = curr_sim_pspec_dic
    '''
    #----------------------------------------
    save_time_tod(tod_file, T_trim, spf)
    save_scan_path(tod_file, scan_path_sky, spf)
    #Finally, we save the TODs of each pixel, depending on their frequency band. 
    #We first save the TODs of pixels in the HW array. 
    start = time.time()
    bar = Bar('Generate the HW TODs', max=len(freqs[:P['nb_channels_per_array']]))
    #Select the pointing path for pixels belonging to the HW array.
    pointing_paths_to_save = pointing_paths[:P['nb_pixel_HW']]
    #Then, for each frequency bandpass: 
    for f, F in enumerate(freqs[:P['nb_channels_per_array']]):
        #Select the frequency channel out of which the TODs will be sampled. 
        Map = cube[f,:,:]
        #Samples the TODs from Map given the pointing paths
        wcs, map, hist, norm, samples, positions_x, positions_y = gen_tod(wcs, Map, ybins, xbins, pixel_offset_HW, pointing_paths_to_save)
        #----------------------------------------
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
        plt.savefig(os.getcwd()+'/plot/'+f'freq{F.value:.0f}GHz_channel_{P["scan"]}_summary_plot.png')
        plt.close()
        #----------------------------------------
        #Select the right names of the detectors in the name list. 
        names = det_names[f * P['nb_pixel_HW'] : (f + 1) * P['nb_pixel_HW']]
        #Save the TODs.
        save_tod_in_hdf5(tod_file, names, samples, pixel_offset_HW, pixel_shift_HW, pointing_paths_to_save, P['detectors_name_file'], F, spf)
        bar.next()
    bar.finish
    print('')
    
          
    #Same block to save the TODs as above, but for the LW array. 
    bar = Bar('Generate the LW TODs', max=len(freqs[ P['nb_channels_per_array']:P['nb_channels_per_array']*2 ]))
    pointing_paths_to_save = pointing_paths[P['nb_pixel_HW']:]
    for f, F in enumerate(freqs[P['nb_channels_per_array']:P['nb_channels_per_array']*2]):
        Map = cube[P['nb_channels_per_array']+f,:,:]
        wcs, map, hist, norm, samples, positions_x, positions_y =  gen_tod(wcs, Map, ybins, xbins, pixel_offset_LW, pointing_paths_to_save)
        #----------------------------------------
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
        plt.savefig(os.getcwd()+'/plot/'+f'freq{F.value:.0f}GHz_channel_{P["scan"]}_summary_plot.png')
        plt.close()
        #----------------------------------------
        #Save the TODs in .hdf5 
        index = P['nb_pixel_HW'] * P['nb_channels_per_array']
        names = det_names[index +f * P['nb_pixel_LW'] :index + (f + 1) * P['nb_pixel_LW']]
        save_tod_in_hdf5(tod_file, names, samples, pixel_offset_LW, pixel_shift_LW, pointing_paths_to_save, P['detectors_name_file'], F, spf)
        bar.next()
    bar.finish
    print(f"Generate the TODs in {np.round((time.time() - start),2)}"+'s')
