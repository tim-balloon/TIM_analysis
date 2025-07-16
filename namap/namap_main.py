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
import ast
import sys

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
    ## bookend mathilde code 
    #------------------------------------------------------------------------------------------

    def load_par_file(filepath):
        """Loads .par file as a dictionary with literal-evaluated values."""
        params = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        val = ast.literal_eval(val)
                    except Exception:
                        pass  # fallback: treat as string
                    params[key] = val
        return params

    # ----------------- ARGPARSE SETUP -----------------
    parser = argparse.ArgumentParser(description='NAMAP Parameters')

    cli = parser.add_argument_group('Command Line Inputs')
    cli.add_argument('--params-file', required = False,  help='.par file containing parameters')

    # command line parameter possibilities:

    cli.add_argument('--output', type=str, default='output.fits', help='Output file name')
    cli.add_argument('--hdf5_file', type=str, help='Path for TOD data (HDF5 format)')
    cli.add_argument('--detector_table', type=str, help='Path to detector table (TSV format)')
    cli.add_argument('--detectors_to_use', type=str, default = None,help='Path to detector to use table (TSV format)')
    cli.add_argument('--frequencies', type=float, default=None, nargs=2, help='Frequency band in GHz, e.g. 715.0 719.0 to make map from')
    cli.add_argument('--num_frames', type=int, help='Integration time in seconds to be loaded')
    cli.add_argument('--first_frame', type=int, help='Starting frame index (in seconds)')
    cli.add_argument('--time_offset', type=int, help='Time offset between detector data and coordinates')
    cli.add_argument('--correction', action='store_true', help='Enable pointing offset correction')
    cli.add_argument('--telemetry', action='store_true', help='Specify if data is from telemetry (e.g. Mole)')
    cli.add_argument('--telescope_coordinate', action='store_true', help='Use telescope coordinates for mapmaking')
    cli.add_argument('--xystage', action='store_true', help='Use XY stage coordinates')
    cli.add_argument('--xsc_offset', type=float, help='Offset w.r.t. star cameras in xEL and EL')
    cli.add_argument('--det_offset', type=float, help='Offset w.r.t. central detector in xEL and EL')
    cli.add_argument('--ctype', type=str, help='Coordinate system to draw the map (e.g. "RA and DEC")')
    cli.add_argument('--input_ctype', type=str, help='Coordinate system for the maps coming in (e.g. "RA and DEC")')
    cli.add_argument('--lat', action='store_true', help='Use latitude flag (currently always True)')
    cli.add_argument('--lst', action='store_true', help='Use LST flag (currently always True)')
    cli.add_argument('--crpix', type=float, nargs=2, help='Reference pixel position (2 floats)')
    cli.add_argument('--cdelt', type=float, nargs=2, help='Pixel resolution along each axis in degrees (2 floats)')
    cli.add_argument('--crval', type=float, nargs=2, help='Sky coordinates at reference pixel (2 floats)')
    cli.add_argument('--pixnum', type=float, nargs=2, help='Number of pixels along each axis (2 floats)')
    cli.add_argument('--highpassfreq', type=float, default = 0.1, help='High-pass filter cutoff frequency (Hz)')
    cli.add_argument('--polynomialorder', type=int, default = 5,help='Polynomial order used to detrend TODs')
    cli.add_argument('--despike', action='store_true', help='Flag to enable despiking of TODs')
    cli.add_argument('--sigma', type=float, help='Sigma threshold for despike detection')
    cli.add_argument('--prominence', type=float, help='Prominence threshold (in sigma units) for despiking')
    cli.add_argument('--coadd', action='store_true', help='Coadd detectors (True) or map each individually')
    cli.add_argument('--gaussian_convolution', action='store_true', help='Apply Gaussian convolution to map')
    cli.add_argument('--std', type=float, help='STD of Gaussian kernel in arcseconds')


    # Step 1: First parse only --params-file
    args_partial, remaining_argv = parser.parse_known_args()

    # Step 2: Load .par values if requested
    defaults = {}
    if args_partial.params_file:
        defaults = load_par_file(args_partial.params_file)

    # Step 3: Set parser defaults from .par
    parser.set_defaults(**defaults)

    # Step 4: Parse full args
    args = parser.parse_args(remaining_argv)

    # Step 5: Convert Namespace to dictionary
    P = vars(args)

    #------------------------------------------------------------------------------------------
    #### start mathilde code 
    #---------------------------------
    num_frames, first_frame = P['num_frames'], P['first_frame']

    #Also need to be implemented. 
    telemetry = P['telemetry']

    #So far, only 'RA and DEC' is implemented and working. 
    if P['input_ctype'] == 'RA and DEC':
        coord1 = str('RA')
        coord2 = str('DEC')
        xystage = False
    elif P['input_ctype'] == 'AZ and EL':
        coord1 = str('AZ')
        coord2 = str('EL')
        xystage = False
    elif P['input_ctype'] == 'CROSS-EL and EL':
        coord1 = str('xEL')
        coord2 = str('EL')
        xystage = False
    elif P['input_ctype'] == 'XY Stage':
        coord1 = str('X')
        coord2 = str('Y')
        xystage = True

    filepath = P['hdf5_file']
    btable = tb.Table.read(P['detector_table'], format='ascii.tab')
    if P['frequencies'] is not None:
        filtered = btable[np.isin(btable['Frequency'], P['frequencies'])]
    if P['detectors_to_use'] is not None:
        good_kid_table = tb.Table.read(P['detectors_to_use'], format='ascii.tab')
        filtered = btable[np.isin(btable['Name'], good_kid_table['Name'])]
    if P['frequencies'] is None and P['detectors_to_use'] is None:
        filtered = btable
    #option in the par file to good kids list
   
    kid_num = filtered['Name']
    print(kid_num)
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
    corr = pt.apply_offset(P['input_ctype'], coord1slice, coord2slice, P['ctype'], xsc_offset, det_offset = det_off, lst = lstslice, lat = latslice, )
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
    maps = mp.maps(P['ctype'], np.asarray([P['crpix'][0],P['crpix'][1]]), np.asarray([P['cdelt'][0],P['cdelt'][1]]), np.asarray([P['crval'][0], P['crval'][1]]), np.asarray([P['pixnum'][0],P['pixnum'][1]]), cleaned_data, coord1slice, coord2slice, convolution, std, 
                   coadd=P['coadd'], noise=noise_det, telcoord = P['telescope_coordinate'], parang=parallactic, params=str(P))
    
    maps.wcs_proj()
    map_values = maps.map2d()
    #--------------------
    print(P)
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
