import numpy as np
import scipy as sp
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('Agg')  # non-GUI backend

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
#from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import os
import astropy.units as u
import matplotlib
import scipy.constants as cst
# Set plot parameters
import sys

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
from make_namap_timestreams import *
from scipy import interpolate
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
#from astropy.cosmology import FlatLambdaCDM
# Set cosmology to match Bolshoi-Planck simulation
#cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307, Ob0=0.048, Tcmb0=2.7255, Neff=3.04)
from astropy.wcs import WCS
from astropy.coordinates import Angle

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
        coordinates of the sky scan path of each pixel, for pixels seeing the same frequency band
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

def save_scan_path(tod_file, scan_path, spf, lower_spf=False):
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
    lower_spf: bool
        if save the rad dec with an spf lower than the spf of the data, change the name of the group. 
    Returns
    -------
    """ 
    if(lower_spf): names = ('RA_lowerspf', 'DEC_lowerspf')
    else: names = ('RA', 'DEC')
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(names, (scan_path_sky[:,0],scan_path_sky[:,1]))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
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
    if namegrp not in H: grp = H.create_group(namegrp)
    else:                grp = H[namegrp]
    if('data' in grp): del grp['data'] 
    if('spf' in grp): del grp['spf'] 
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
    dect_file: string
        the name of the .csv where the info on each pixel is stored. This function add the frequency band info to the .csv file. 
    spf: int
        the number of samples per frame
    F: astropy quantity
        the frequency band seen by the detectors

    Returns
    -------
    """ 
    
    H = h5py.File(tod_file, "a")

    for detector, (offset, shift, name, pointing) in enumerate(zip(pixel_offset, pixel_shift, det_names, pointing_paths)):
        
        namegrp = f'kid_{name}_roach'
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        if('pixel_offset_y' in grp): del grp['pixel_offset_y'] 
        if('pixel_offset_x' in grp): del grp['pixel_offset_x'] 
        if('frequency' in grp): del grp['frequency'] 
        grp.create_dataset('data', data=samples[detector,:],
                            compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
        grp.create_dataset('pixel_offset_y', data=offset)
        grp.create_dataset('pixel_offset_x', data=shift)
        grp.create_dataset('frequency', data=F)

        namegrp = f'kid_{name}_RA'
        if namegrp not in H: grp = H.create_group(namegrp)   
        else:                grp = H[namegrp] 
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=pointing[:,0])
        grp.create_dataset('spf', data=spf)
        
        namegrp = f'kid_{name}_DEC'
        if namegrp not in H: grp = H.create_group(namegrp)   
        else:                grp = H[namegrp] 
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
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
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
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
    for i, (name, coord) in enumerate(zip(('AZ', 'EL'), (azimuths,elevations))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def save_telescope_coord(tod_file, x_tel, y_tel, spf):
    """
    Save the scan path in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    azimuths: array
    elevations: array 
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('xEL', 'EL'), (x_tel,y_tel))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def save_PA(tod_file, PA, spf):
    """
    Save the parallactic angle in the .hdf5 format. 

    Parameters
    ----------
    tod_file: string 
        name of the output hdf5 file  
    PA: array
        the parallactic angle timestream 
    spf: int
        the number of samples per frame
    Returns
    -------
    """ 
    H = h5py.File(tod_file, "a")
    for i, (name, coord) in enumerate(zip(('PA',), (PA,))):
        namegrp = name
        if namegrp not in H: grp = H.create_group(namegrp)
        else:                grp = H[namegrp]
        if('data' in grp): del grp['data'] 
        if('spf' in grp): del grp['spf'] 
        grp.create_dataset('data', data=coord, compression='gzip', compression_opts=9)
        grp.create_dataset('spf', data=spf)
    H.close() 

def make_a_whitenoise_cube(cube,hdr):


    normpk = hdr['CDELT1'] * hdr['CDELT2'] / (hdr['NAXIS1'] * hdr['NAXIS2'])
    w_freq = np.fft.fftfreq(hdr['NAXIS2'], d=((hdr['CDELT1']*u.deg).to(u.arcmin)).value)
    v_freq = np.fft.fftfreq(hdr['NAXIS1'], d=((hdr['CDELT1']*u.deg).to(u.arcmin)).value)
    k = np.sqrt(w_freq[:,np.newaxis]**2 + v_freq[np.newaxis,:]**2)
    kmin = np.min(k[np.where(k!=0)])
    kmax = np.max(k[np.where(k!=0)])
    k_bin = np.logspace(np.log10(kmin), np.log10(kmax),20)
    histo, e = np.histogram(k, bins = k_bin)
    K, e = np.histogram(k, bins = k_bin, weights=k)
    kmean = K/histo
    cube_wn = []

    for i in range(cube.shape[0]):
        M = cube[i,:,:]
        pow_sqr = np.absolute(np.fft.fftn(M)**2 * normpk )
        pk, e = np.histogram(k, bins = k_bin, weights=pow_sqr)
        pk /= histo
        m,_ = gaussian_random_field(kmean, pk, int(hdr['NAXIS2']*1.1), int(hdr['NAXIS1']*1.1), ((hdr['CDELT1']*u.deg).to(u.arcmin)).value, ny_exit = hdr['NAXIS2'], nx_exit = hdr['NAXIS1'], l_cutoff=kmax)
        cube_wn.append(m)
    
    return np.asarray(cube_wn)

def gaussian_random_field(l, clt, ny, nx, res, ny_exit = None, nx_exit = None, l_cutoff=None,  cl_map = None,sigma = 0, force = True):
    """
    Create a map of a Gaussian random field, given its angular power spectrum or from a power law with a given index 
    
    l (astropy.Quantity): the multipol

    clt (astropy.Quantity): the given power spectrum 

    ny, nx: the size of the map to generate

    res (astropy.Quantity): the resolution of the map

    ny_exit, nx_exit (optional): <ny, nx, size of the map to return, to avoid periodic boundaries

    l_cutoff (optional, astropy.Quantity): the maximum multipol to taken into account in the map generation

    sigma (optional, astropy.Quantity): the size of the instrumental Gaussian beam profile in sigma unit to apply, if any. 

    ampl (optional): the amplitude of the power law to compute the angular power spectrum

    index (optional): the index of the power law to compute the angular power spectrum

    cl_map (optional, astropy.Quantity): map containing the power spectrum values for each l values of the map. 
                                         If provided, not recomputed. 

    force (bool): set the negative values in the power spectrum map to zero. 

    return: 
    
    real_space_map (astropy.Quantity): the generated map in real space.

    cl_map (astropy.Quantity): the angular power spectrum map
    """
    
    lmap_rad_y = ny*res
    lmap_rad_x = nx*res
    #Generate gaussian amplitudes
    norm = np.sqrt( (nx/lmap_rad_x)*(ny/lmap_rad_y)) 

    noise = np.random.normal(loc=0, scale=1, size=(ny,nx))
    dmn1  = np.fft.fft2( noise )

    #Interpolate input power spectrum
    if(cl_map is None):
            
            
        w_freq = np.fft.fftfreq(ny, d=res)
        v_freq = np.fft.fftfreq(nx, d=res)
        lmap = np.sqrt(w_freq[:,np.newaxis]**2 + v_freq[np.newaxis,:]**2)


        if(l_cutoff is not None): lmax = np.minimum(l_cutoff, l.max())
        else: lmax = np.minimum(l.max(), lmap.max()) 
    
        cl_map = np.zeros(lmap.shape)
        w = np.where((lmap>l.min()) & (lmap<=lmax))
        if(not w[0].any()): print("wrong k range")
        else:
            #Power law spectrum
            print("interpolate")
            f = interpolate.interp1d( l, clt,  kind='linear')
            cl_map[w] = f(lmap[w])
            w1 = np.where( cl_map <= 0)
            if(w1[0].shape[0] != 0 and force): cl_map[w1] = 0

    #Fill amn_t
    amn_t = dmn1 * norm * np.sqrt( cl_map )
    
    #Output map
    real_space_map = np.real(np.fft.ifft2( amn_t ))
    
    if(ny_exit is not None and nx_exit is not None):
        scale_y = ny / ny_exit
        iy0 = int(( (scale_y - 1)*ny_exit/2))
        scale_x = nx / nx_exit
        ix0 = int(( (scale_x - 1)*nx_exit/2))
        return real_space_map[iy0:int(iy0+ny_exit), ix0:int(ix0+nx_exit)], cl_map

    else: return real_space_map, cl_map

def format_duration(hours):
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h}h{m}min{s}sec"



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
        make the simIM version
        implement slope maker in tod / , (noise peak / cosmic ray)  maker. 
        implement a function that makes one loop during T_duration instead of several? 
        Load the pixel offsets from the csv instead of computing them. 
    '''
    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    parser = argparse.ArgumentParser(description="strategy parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    parser.add_argument('--non_iteractive', help = "deactivate matplotlib", action="store_true")

    args = parser.parse_args()

    if(args.non_iteractive): 
        import matplotlib
        matplotlib.use("Agg")

    P = load_params(args.params)
    #------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------
    #Initiate the parameters

    #The coordinates of the field
    name='Goods-S Field' #P['name_field']
    c=SkyCoord.from_name(name)
    ra = 0 
    rafield = c.ra.value
    dec = c.dec.value
    #The contour of the field
    contours = P['contours']
    x_cen, y_cen = np.mean(contours[:, 1]), np.mean(contours[:, 0])

    #load the observer position
    lat = P['latittude']

    #Plot parameter
    f_range = P['f_range']

    #Load the resolution. 
    #if not in params, load it from the map used to generate the TOD. 
    res = P['res']
    if(res is None):
        hdr = fits.getheader(P['path']+P['file'])
        res = (hdr['CDELT1'] * u.Unit(hdr['CUNIT1'])).to(u.deg).value

    #Pixel offsets from the center of the field of view in degree.

    #---------------------------    
    #Generate the offset of the pixels with respect to the center of the two arrays, in degrees. 
    pixel_offset_HW, pixel_shift_HW = pixelOffset(P['nb_pixel_HW'], P['offset_HW'], -P['arrays_separation']/2)
    pixel_offset_LW, pixel_shift_LW = pixelOffset(P['nb_pixel_LW'], P['offset_LW'], P['arrays_separation']/2) 
    
    pixel_offset = np.concatenate((pixel_offset_HW, pixel_offset_LW))
    pixel_shift = np.concatenate((pixel_shift_HW, pixel_shift_LW))

    #Angle of the rotation to apply to the detector array. 
    theta = np.radians(P['theta'])

    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    spf = int(1/np.round(dt*3600,3)) #sample per frame defined here as the acquisition rate in Hz. 
    T = np.arange(0,T_duration,dt) * 3600 #s
    #local sideral time
    LST = np.arange(-T_duration/2,T_duration/2,dt) #hours
    #------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------    
    #Generate the scan path for the center of the arrays. 
    if(P['scan']=='loop'):   az, alt, flag = genLocalPath(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='raster'): az, alt, flag = genLocalPath_cst_el_scan(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))
    if(P['scan']=='zigzag'): az, alt, flag = genLocalPath_cst_el_scan_zigzag(az_size=P['az_size'], alt_size=P['alt_size'], alt_step=P['alt_step'], acc=P['acc'], scan_v=P['scan_v'], dt=np.round(dt*3600,3))

    scan_path, scan_flag = genScanPath(T, alt, az, flag)
    scan_path = scan_path[scan_flag==1] #Use the scan flag to keep only the constant scan speed part of the pointing. 
    T_trim = T[scan_flag==1]
    LST_trim = LST[scan_flag==1]

    #Generate the pointing on the sky for the center of the arrays
    scan_path_sky, azel = genPointingPath(T_trim, scan_path, LST_trim, lat, dec, ra, azel=True) 

    #Generate the scan path of each pixel, as a function of their offset to the center of the arrays. 
    pixel_paths  = genPixelPath(scan_path, pixel_offset, pixel_shift, theta)

    #Generate the pointing on the sky of each pixel. 
    start = time.time() 
    pointing_paths = [genPointingPath(T_trim, pixel_path, LST_trim, lat, dec, ra) for pixel_path in pixel_paths]
    #Generate the hitmap, using all the detectors. 
    xedges,yedges,hit_map = binMap(pointing_paths,res=res,f_range=f_range,dec=dec,ra=ra) 
    #----------------------------------------

    #----------------------------------------
    #Plot a scan route
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
    fig, axs = plt.subplots(3,2,figsize=(10,10), dpi=160,)# sharey=True, sharex=True)
    #---
    axradec, ax, axp, axpix, axr = axs[0,0], axs[1,0], axs[0,1], axs[1,1], axs[2,0]
    axs[2,1].axis('off') 
    axradec.plot(az-az.max()/2,alt-alt.max()/2,'cyan')
    axradec.set_xlabel('Az [deg]')
    axradec.set_ylabel('El [deg]')
    axradec.set_aspect(aspect=1)
    #---
    heatmap, xedges, yedges = np.histogram2d(scan_path_sky[:,0], scan_path_sky[:,1], bins=int(2/res))
    im = ax.imshow(heatmap.T, origin='lower', cmap='binary',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im, ax=ax, label='1 detector counts')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    if(contours is not None): ax.plot(contours[:, 1]-rafield, contours[:, 0], c='g' )
    if(contours is not None): axradec.plot(contours[:, 1]-rafield, contours[:, 0]-dec, c='g' )
    #---
    az_unwrapped = (np.radians(azel[:,0]) + np.pi) % (2 * np.pi) - np.pi
    axr.plot(np.degrees(az_unwrapped),azel[:,1],'b')
    axr.set_xlabel('Az [deg]')
    axr.set_ylabel('El [deg]')
    ##---
    img = axp.imshow((hit_map), extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], 
                    interpolation='nearest', origin='lower', vmin=0, vmax=np.max(hit_map), cmap='binary' )
    if(contours is not None): axp.plot(contours[:, 1], contours[:, 0], c= 'g')
    fig.colorbar(img, ax=axp, label='Counts')
    #CS = axp.contour(np.roll(np.sqrt(hit_map*np.roll(hit_map, 4, axis=1)),-2,axis=1), levels=[0.5*np.max(hit_map)], lw=0.5, origin='lower', linewidths=0.6,
    #extent=[x_cen-f_range, x_cen+f_range,y_cen-f_range, y_cen+f_range,], colors='magenta')
    #axp.clabel(CS, inline=1, fontsize=8, )
    axp.set_xlabel('RA [deg]')
    axp.set_ylabel('Dec [deg]')
    #---
    patchs = []
    axpix.set_title('Pointing Path per pixel')
    idx = np.arange(len(pixel_paths))[::10]
    n = 11
    for i,c in zip(idx, cm.rainbow(np.linspace(0.,1,len(idx)))):
        axpix.scatter(pointing_paths[i][::n,0], pointing_paths[i][::n,1], s=0.1,c=c)
        patch = mpatches.Patch(color=c, label= 'pixel %d'%i); patchs.append(patch); 
    axpix.set_aspect(aspect=1)
    axpix.legend(handles=patchs,frameon=False, bbox_to_anchor=(1,1))
    axpix.set_xlabel('RA [deg]')
    axpix.set_ylabel('Dec [deg]')
    fig.tight_layout()
    plt.savefig(os.getcwd()+'/plot/'+f"scan_route_{P['scan']}.png")
    plt.show()
    #----------------------------------------

    #----------------------------------------
    #Generate the TODs and save Them in hdf5
    #The path to the sky simulation from which to generate the TODs from
    simu_sky_path = P['path']+P['file'] #os.getcwd()
    #The output hdf5 file containing the generated TODs. 
    tod_file=P['path']+f'TOD_{format_duration(T_duration)}'+P['file'][:-5]+'.hdf5' #os.getcwd()+'/'+
    hitmap_file=P['path']+f'TOD_{format_duration(T_duration)}'+'.fits' #os.getcwd()+'/'+

    #Load the names of the detectors. We assigned to them random names so that we cannot do naive for loop and avoid mistakes.
    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    det_names = det_names_dict['Name']
    #----------------------------------------

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
    cube *= 1e6
    if(P['whitenoise']): cube = make_a_whitenoise_cube(cube, hdr)
    cube *= pix_size #conversion MJy/sr to Jy/beam
    #Create the world coordinate object and save it. 
    wcs = WCS(hdr) 
    d = {'wcs':wcs}
    pickle.dump(d, open(P['wcs_dict'], 'wb'))
    wcs = WCS(hdr, naxis=2) 
    #----------------------------------------

    #----------------------------------------
    
    #latitude timestream
    lat = np.ones(len(LST_trim)) * lat

    #Generate the telescope coordinates and parallactic angle. 

    #RA and Dec
    coord1 = np.radians(scan_path_sky[:,0])
    coord2 = np.radians(scan_path_sky[:,1])

    cos_lat = np.cos(np.radians(lat))
    sin_lat = np.sin(np.radians(lat))

    hour_angle = (LST_trim - ra / 15)*np.pi/12
    index, = np.where(hour_angle<0)
    hour_angle[index] += 2*np.pi
    
    #Parallactic angle
    y_pa = cos_lat*np.sin(hour_angle)
    x_pa = sin_lat*np.cos(coord2)-np.cos(hour_angle)*cos_lat*np.sin(coord2)
    pa = np.arctan2(y_pa, x_pa)

    #Telescope coordinates
    x_tel = coord1*np.cos(pa)-coord2*np.sin(pa)
    y_tel = coord2*np.cos(pa)+coord1*np.sin(pa)
    #----------------------------------------

    #----------------------------------------
    #Save timestreams in a .hdf5 file 
    save_PA(tod_file, np.degrees(pa), spf)
    save_telescope_coord(tod_file, np.degrees(x_tel), np.degrees(y_tel), spf)
    save_lst_lat(tod_file, LST_trim, lat, spf)
    save_az_el(tod_file, azel[:,0], azel[:,1], spf)
    save_time_tod(tod_file, T_trim, spf)
    save_scan_path(tod_file, scan_path_sky, spf)
    #-------------------------------------------
    
    #-------------------------------------------
    lower_spf = 50
    save_scan_path(tod_file, scan_path_sky[::int(spf/lower_spf)], spf=lower_spf, lower_spf=True)
    #-------------------------------------------

    #Finally, we save the TODs of each pixel, depending on their frequency band. 
    #We first save the TODs of pixels in the HW array, then in the LW array. 
    start = time.time()

    for array, freqs_of_array, pointing_paths_to_save, pixel_offset_array, pixel_shift_array in zip(
                                                           ('HW', 'LW'), 
                                                           (freqs[:P['nb_channels_per_array']], freqs[ P['nb_channels_per_array']:P['nb_channels_per_array']*2 ] ),
                                                           (pointing_paths[:P['nb_pixel_HW']],  pointing_paths[P['nb_pixel_HW']:]),
                                                           (pixel_offset_HW, pixel_offset_LW),
                                                           (pixel_shift_HW, pixel_shift_LW)):
        
        bar = Bar(f'Generate the {array} TODs', max=len(freqs_of_array))
        #Then, for each frequency bandpass: 
        for f, F in enumerate(freqs_of_array):

            #----------------------------------------
            #Select the frequency channel out of which the TODs will be sampled. 
            if(array=='LW'): Map = cube[f+P['nb_channels_per_array'],:,:]
            else: Map = cube[f,:,:]
            #----------------------------------------

            #----------------------------------------
            #Samples the TODs from Map given the pointing paths ofppixels in an array seing the same frequency (but different beams).
            wcs, map, hist, norm, samples, positions_x, positions_y = gen_tod(wcs, Map, ybins, xbins, pixel_offset_array, pointing_paths_to_save)
            #----------------------------------------

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

            #----------------------------------------
            #Select the right names of the detectors in the name list. 
            if(array=='HW'): 
                names = det_names[f * P['nb_pixel_HW'] : (f + 1) * P['nb_pixel_HW']]
            else:
                index = P['nb_pixel_HW'] * P['nb_channels_per_array']
                names = det_names[index +f * P['nb_pixel_LW'] :index + (f + 1) * P['nb_pixel_LW']]
            #----------------------------------------

            #----------------------------------------
            #Save the TODs.
            save_tod_in_hdf5(tod_file, names, samples, pixel_offset_array, pixel_shift_array, pointing_paths_to_save, P['detectors_name_file'], F, spf)
            #----------------------------------------

            bar.next()
        bar.finish
        print('')
    
    print('save '+tod_file)
    print(f"Generate the TODs in {np.round((time.time() - start),2)}"+'s')
    

    if(True):
        det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
        #Each pixel with the same offset sees the same beam, but in different frequency band. 
        same_offset_groups = det_names_dict.groupby(['Frequency'])['Name'].apply(list).reset_index()
        start = time.time()
        #For each group of pixels seeing the same beam: 
        for group in range(len(same_offset_groups)):
            print(f'starting group {group}')
            H = h5py.File(tod_file, "a")    
            for j, name in  enumerate(same_offset_groups.iloc[group]['Name']):
                namegrp = f'kid_{name}_roach'
                f = H[namegrp]
                data = f['data'][()]
                data_with_slope = data + add_polynome_to_timestream(data, T)
                data_with_peak = add_peaks_to_timestream(data_with_slope)
                if('namap_data' in f): del f['namap_data'] 
                f.create_dataset('namap_data', data=data_with_peak,
                                    compression='gzip', compression_opts=9)
            H.close()
            end = time.time()
            timing = end - start
        print(f'Generate the namap in {np.round(timing,2)} sec!')