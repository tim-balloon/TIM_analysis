import pygetdata as gd
import numpy as np
from IPython import embed
import src.detector as det
import src.loaddata as ld
import src.detector as tod
import src.mapmaker as mp
import src.pointing as pt  
import copy
from astropy import wcs 
import os
from src.gui import MapPlotsGroup
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import h5py
import argparse
from scipy.interpolate import interp1d
import scipy.signal as sgn
import matplotlib.pyplot as plt 
import tracemalloc
import astropy.table as tb

def load_params(path, force_pysides_path = ''):

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


if __name__ == "__main__":

    '''
    1st: git clone from TIM_analysis/namap, mathilde branch or main branch if it exist. 

    (Optional, needed for 4th A) 2nd: Download a mock sky: scp yournetid@cc-login.campuscluster.illinois.edu:/projects/ncsa/caps/TIM_analysis/sides_angular_cubes/TIM/pySIDES_from_uchuu_tile_0_1.414deg_x_1.414deg_fir_lines_res20arcsec_dnu4.0GHz_full_de_Looze_smoothed_MJy_sr.fits .
    and put it in namap/fits_and_hdf5/
    
    3rd: generate the KIDs file: python gen_det_names.py params_strategy.par

    4th A: generate the TOD file: python strategy.py params_strategy.par 
    OR
    4th B: Download the TOD file: scp yournetid@cc-login.campuscluster.illinois.edu:/projects/ncsa/caps/TIM_analysis/timestreams/TOD_pySIDES_from_uchuu_tile_0_1.414deg_x_1.414deg_fir_lines_res20arcsec_dnu4.0GHz_full
_de_Looze_smoothed_MJy_sr.hdf5 . , 
    and put it in namap/fits_and_hdf5/

    To run: python namap_main.py params_namap.par

    Left to be done:
        Implement the telemetry option 
        Implement respons correction
        Implement parallactic angle 
        Implement noise detectors
        Implement spectral axis 

    The different par: 
     params_namap.par: generates the coadd map of all the (64 so far) detectors 
     params_namap_5det.par: generates the coadd map of 5 detectors 
     params_namap_5det_ind.par: generates the individual maps of 5 detectors 
     params_namap_1det_ind.par: generates the individual map of the central detector

    '''

    #-------------------------------------------------------------------------------------------------------------------------
    #Iinitialization
    parser = argparse.ArgumentParser(description='NaMap. A naive mapmaker written in Python for BLASTPol and BLAST-TNG', \
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()

    P = load_params(args.params)

    num_frames, first_frame = P['num_frames'], P['first_frame']

    #Lat and lst need to be implemented in strategy.py
    lat, lst = P['lat'], P['lst']
    #Also need to be implemented. 
    telemetry = P['telemetry']

    #So far, only 'RA and DEC' is implemented and working. 
    if P['ctype'] == 'RA and DEC':
        coord1 = str('RA')
        coord2 = str('DEC')
        xystage = False
    elif P['ctype'] == 'AZ and EL':
        coord1 = str('AZ')
        coord2 = str('EL')
        xystage = False
    elif P['ctype'] == 'CROSS-EL and EL':
        coord1 = str('xEL')
        coord2 = str('EL')
        xystage = False
    elif P['ctype'] == 'XY Stage':
        coord1 = str('X')
        coord2 = str('Y')
        xystage = True

    filepath = P['hdf5_file']
    kid_num = P['kid_num']

    if(kid_num=='all'): 
        btable = tb.Table.read(P['detector_table'], format='ascii.tab')
        kid_num = btable['Name']

    #load the table
    dettable = ld.det_table(kid_num, P['detector_table']) 
    det_off, noise_det, resp = dettable.loadtable()
    
    #Cleaning data parameters
    highpassfreq = P['highpassfreq']
    polynomialorder = P['polynomialorder']
    despike_bool = P['despike']
    sigma,prominence = P['sigma'],P['prominence']
    #So far, the cleaning parameters are set such as no cleaning is applied. 
    #Because the simulated TODS dont have noise or noise peaks. 

    #Beam convolution parameters
    convolution, std = P['gaussian_convolution'], P['std'] 

    #-------------------------------------------------------------------------------------------------------------------------
    #Load the data
    dataload = ld.data_value(filepath, kid_num, filepath, coord1, coord2, lst, lat, first_frame, num_frames, xystage, telemetry)
    det_data, coord1_data, coord2_data, lst, lat, spf_data, spf_coord, lat_spf = dataload.values()
    #--------------------
    #Synchronize the data
    #Needs to be modify , nothing to synchronize so far. 
    '''
    zoomsyncdata = ld.frame_zoom_sync(filepath, det_data, spf_data, spf_data, coord1_data, coord2_data, spf_coord, spf_coord, 
                                      first_frame, num_frames+first_frame,  lst,lat,lat_spf, lat_spf, P['time_offset'], xystage,)
    timemap, detslice, coord1slice, coord2slice, lstslice, latslice = zoomsyncdata.sync_data()
    '''
    #--------------------
    #Do some corrections
    #if coord1.lower() == 'xel': coord1slice *= np.cos(np.radians(coord2slice)) 

    #--------------------
    #Needs to be implemented. So far, I just load the coordinates timestreams for each pixel directly corrected of their offset
    '''
    if(P['correction'] and P['pointing_table'] is not None):               
            #--------------------
            #Needs to be modify !
            xsc_file = ld.xsc_offset(P['pointing_table'], first_frame, num_frames+first_frame)
            xsc_offset = xsc_file.read_file()
            #--------------------
    else:
        xsc_offset = np.zeros(2)
    corr = pt.apply_offset(coord1slice, coord2slice, P['ctype'], xsc_offset, det_offset = det_off, lst = lstslice, lat = latslice)
    coord1slice, coord2slice = corr.correction()
    '''
    coord1slice = coord1_data
    coord2slice = coord2_data
    lstslice = lst
    latslice = lat
    detslice = det_data

    #--------------------
    #Need to be implemented ! So far, set parallactic to 0.
    parallactic=[]
    if not P['telescope_coordinate'] and lstslice is not None and latslice is not None:
        #--------------------
        #Needs to be modify !
        for j, (c1, c2, ilst, ilat) in enumerate(zip(coord1slice,coord2slice, lstslice, latslice)): 
            tel = pt.utils(c1/15., c2, ilst, ilat)
            parallactic.append( tel.parallactic_angle() )
        #--------------------
    else:
        for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
            parallactic.append( np.zeros_like(c1) )
    #---------------------------------
    #Clean the TOD by removing smooth polynomial component, replace peaks, and apply a high pass filter
    
    det_tod = tod.data_cleaned(detslice, spf_data,highpassfreq, kid_num, polynomialorder, despike_bool, sigma, prominence)
    cleaned_data = det_tod.data_clean()

    #--------------------
    #Apply detector's response
    cleaned_data = [arr * resp for arr, resp in zip(cleaned_data, resp)]
    #--------------------
    
    #Create the maps
    #!!!! Change the crval if you change the field coordinates used to gen the timestreams. 
    #Pixnum option to limit the size of the map needs implementation. 
    # P['Power_only'] generates only the sqrt(I**2+Q**2) maps. I and Q maps need implementations. 
    #telcoord option needs implementation. 
    maps = mp.maps(P['ctype'], P['crpix'], P['cdelt'], P['crval'], P['pixnum'], cleaned_data, coord1slice, coord2slice, convolution, std, 
                   Ionly=P['Power_only'], coadd=P['coadd'], noise=noise_det, telcoord = P['telescope_coordinate'], parang=parallactic)
    
    maps.wcs_proj()
    #proj = maps.proj
    #w = maps.w
    map_values = maps.map2d()
    #---------------------------------
    #Plot the maps
    maps.map_plot(data_maps = map_values, kid_num=kid_num)
    #---------------------------------
    
