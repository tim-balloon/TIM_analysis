import numpy as np
import src.loaddata as ld
import src.detector as tod
import src.mapmaker as mp
import src.pointing as pt  
import copy
from astropy import wcs 
import astropy.table as tb
import h5py
import argparse

#for debugging purpose only
from IPython import embed

#for profilling only
import tracemalloc
import time


if __name__ == "__main__":

    '''
    If you want to modify this code, please create your own branch. 

    Instructions: 

    1/4: git clone from TIM_analysis/namap

    (Optional, needed for 4/4A) 2/4: Download a mock sky: scp yournetid@cc-login.campuscluster.illinois.edu:/projects/ncsa/caps/TIM_analysis/sides_angular_cubes/TIM/pySIDES_from_uchuu_tile_0_1.414deg_x_1.414deg_fir_lines_res20arcsec_dnu4.0GHz_full_de_Looze_smoothed_MJy_sr.fits .
    and put it in namap/fits_and_hdf5/
    
    3/4: generate the KIDs file: python gen_det_names.py params_strategy.par

    4/4A: generate the TOD file: python strategy.py params_strategy.par 
    OR
    4/4B: Download the TOD file: scp yournetid@cc-login.campuscluster.illinois.edu:/projects/ncsa/caps/TIM_analysis/timestreams/TOD_pySIDES_from_uchuu_tile_0_1.414deg_x_1.414deg_fir_lines_res20arcsec_dnu4.0GHz_full
_de_Looze_smoothed_MJy_sr.hdf5 . , 
    and put it in namap/fits_and_hdf5/

    To run: python namap_main.py params_namap.par

    Left to be done:
        Implement the telemetry option 
        Implement respons correction
        Test parallactic angle 
        Implement noise detectors
        Implement spectral axis 
    '''

    tracemalloc.start()
    start = time.time()

    #------------------------------------------------------------------------------------------
    #load the .par file parameters
    parser = argparse.ArgumentParser(description="NAMAP parameters",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('params', help=".par file with params", default = None)
    args = parser.parse_args()
    P = ld.load_params(args.params)
    #------------------------------------------------------------------------------------------

    #---------------------------------
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

    btable = tb.Table.read(P['detector_table'], format='ascii.tab')
    filtered = btable[np.isin(btable['Frequency'], P['frequencies'])]
    kid_num = filtered['Name']

    #load the table
    dettable = ld.det_table(kid_num, P['detector_table']) 
    det_off, noise_det, resp = dettable.loadtable()
    
    #Cleaning data parameters
    highpassfreq = P['highpassfreq']
    polynomialorder = P['polynomialorder']
    despike_bool = P['despike']
    sigma,prominence = P['sigma'],P['prominence']

    #Beam convolution parameters
    convolution, std = P['gaussian_convolution'], P['std'] 
    #---------------------------------

    #-------------------------------------------------------------------------------------------------------------------------
    #Load the data
    dataload = ld.data_value(filepath, kid_num, coord1, coord2, first_frame, num_frames, telemetry)
    detslice, coord1slice, coord2slice, lstslice, latslice, spf_data, spf_coord, lat_spf = dataload.values()
    #---------------------------------

    #--------------------
    #Do some corrections
    if coord1.lower() == 'xel': coord1slice *= np.cos(np.radians(coord2slice)) 
    #--------------------

    #---------------------------------
    #Offset with respect to star cameras in xEL and EL
    xsc_offset = (P['xsc_offset'],P['det_offset']) #needs to be tested with real offsets. 
    #xsc_file = ld.xsc_offset(P['pointing_table'], first_frame, num_frames+first_frame)
    #xsc_offset = xsc_file.read_file()
    #---------------------------------

    #---------------------------------
    corr = pt.apply_offset(coord1slice, coord2slice, P['ctype'], xsc_offset, det_offset = det_off, lst = lstslice, lat = latslice, )
    coord1slice, coord2slice = corr.correction()
    #---------------------------------

    #--------------------
    #Need to be implemented ! So far, set parallactic angle to 0.
    parallactic=[]
    if P['telescope_coordinate']:
        for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
            tel = pt.utils(c1, c2, lstslice, latslice)
            parallactic.append( tel.parallactic_angle() )
    else:
        for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
            parallactic.append( np.zeros_like(c1) )
    #---------------------------------
            
    #---------------------------------
    #Clean the TOD by removing smooth polynomial component, replace peaks, and apply a high pass filter
    det_tod = tod.data_cleaned(detslice, spf_data, highpassfreq, kid_num, polynomialorder, despike_bool, sigma, prominence)
    cleaned_data = det_tod.data_clean()
    #---------------------------------

    #---------------------------------
    #Apply detector's response
    cleaned_data = [arr * resp for arr, resp in zip(cleaned_data, resp)]
    #---------------------------------

    #--------------------
    #Create the maps
    maps = mp.maps(P['ctype'], P['crpix'], P['cdelt'], P['crval'], P['pixnum'], cleaned_data, coord1slice, coord2slice, convolution, std, 
                   coadd=P['coadd'], noise=noise_det, telcoord = P['telescope_coordinate'], parang=parallactic)
    
    maps.wcs_proj()
    map_values = maps.map2d()
    #--------------------

    #--------------------------------------------------
    #Plot the maps
    maps.map_plot(data_maps = map_values, kid_num=kid_num)
    #--------------------------------------------------

    #------------------------------------------------------
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
    tracemalloc.stop()
    end = time.time()
    timing = end - start
    print(f'Run Namap in {np.round(timing,2)} sec! ')
    #------------------------------------------------------
