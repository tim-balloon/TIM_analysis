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

#for profilling purpose only
import tracemalloc
import time


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

def main(P, nbdets=None):

    #-----------------------------------------------------------------------------------------

    _prec = str(P['precision'].lower())
    dtype_map = {
        '16': np.float16, 'float16': np.float16, 'half': np.float16,
        '32': np.float32, 'float32': np.float32, 'single': np.float32,
        '64': np.float64, 'float64': np.float64, 'double': np.float64
    }

    int_map = {
        '16': np.int16, 'float16': np.int16, 'half': np.int16,
        '32': np.int32, 'float32': np.int32, 'single': np.int32,
        '64': np.int64, 'float64': np.int64, 'double': np.int64
    }


    try:
        DT = dtype_map[_prec]
        IT = int_map[_prec]
    except KeyError:
        raise ValueError(f"Unsupported precision '{_prec}'. Choose float16/32/64 or 16/32/64.")
    print(f"Using numeric dtype: {DT}")
    #-----------------------------------------------------------------------------------------
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
    if(nbdets is not None): kid_num = filtered['Name'][:nbdets]
    print('Nb dets: ', len(kid_num))

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
    dataload = ld.data_value(filepath, kid_num, coord1, coord2, first_frame, num_frames,  DT, IT,telemetry)
    det_data, coord1_data, coord2_data, lst_data, lat_data, spf_data, spf_coord, lat_spf, acqfreq_data, acqfreq_coord, acqfreq_lstlat = dataload.values()
    #-------------------------------

    #---------------------------------
    #First remove noise peaks
    det_tod = tod.data_cleaned(det_data, spf_data, kid_num, 0, 0, despike_bool, sigma, prominence)
    cleaned_data = det_tod.data_clean()
    #---------------------------------

    #---------------------------------
    if(len(cleaned_data[0]) != len(coord1_data) or not P['bypass_synch']):
        
        zoomsyncdata = ld.frame_zoom_sync(filepath, cleaned_data, acqfreq_data, spf_data,  coord1_data, 
                                            coord2_data, acqfreq_coord, spf_coord, first_frame, num_frames, 
                                            lst_data, lat_data, acqfreq_lstlat, lat_spf, DT, IT, offset=0)

        timemap, cleaned_data, coord1_data, coord2_data, lst_data, lat_data = zoomsyncdata.sync_data()  
    #---------------------------------

    #---------------------------------
    #Clean the TOD by removing smooth polynomial component and apply a high pass filter
    det_tod = tod.data_cleaned(cleaned_data, spf_data, kid_num, highpassfreq, polynomialorder, False, 0, 0)
    cleaned_data = det_tod.data_clean()

    
    #Apply detector's response
    cleaned_data = [arr * resp for arr, resp in zip(cleaned_data, resp)]
    #---------------------------------

    #---------------------------------
    #Offset with respect to star cameras in xEL and EL
    xsc_offset = (P['xsc_offset'],P['det_offset']) #needs to be tested with real offsets. 
    #xsc_file = ld.xsc_offset(P['pointing_table'], first_frame, num_frames+first_frame)
    #xsc_offset = xsc_file.read_file()
    
    corr = pt.apply_offset(P['input_ctype'], coord1_data, coord2_data, P['ctype'], xsc_offset, DT,IT, det_offset = det_off, lst = lst_data, lat = lat_data, )
    coord1slice, coord2slice = corr.correction()
    #---------------------------------

    #--------------------
    #Need to be implemented ! So far, set parallactic angle to 0.
    parallactic=[]
    if P['telescope_coordinate']:
        for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
            tel = pt.utils(c1, c2, lst_data, lat_data)
            parallactic.append( tel.parallactic_angle() )
    else:
        for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
            parallactic.append(np.zeros_like(c1, dtype=DT))
    
    #---------------------------------

    #--------------------
    #Create the maps
    maps = mp.maps(P['ctype'], 
                   np.asarray([P['crpix'][0],P['crpix'][1]]), 
                   np.asarray([P['cdelt'][0],P['cdelt'][1]]), 
                   np.asarray([P['crval'][0], P['crval'][1]]), 
                   np.asarray([P['pixnum'][0],P['pixnum'][1]]), 
                   cleaned_data, coord1slice, coord2slice, convolution, std, P['output_map'], DT,IT,
                   coadd=P['coadd'], noise=noise_det, telcoord = P['telescope_coordinate'], parang=parallactic, params=str(P))
    
    maps.wcs_proj()
    map_values = maps.map2d()
    #--------------------

    #--------------------------------------------------
    #Plot the maps
    maps.map_plot(data_maps = map_values, kid_num=kid_num)
    #--------------------------------------------------

    return 0

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
        Implement respons correction
        Test parallactic angle 
        Implement noise detectors
        add buffer frames
    '''

    ## bookend mathilde code 
    #------------------------------------------------------------------------------------------

    # ----------------- ARGPARSE SETUP -----------------
    parser = argparse.ArgumentParser(description='NAMAP Parameters')

    cli = parser.add_argument_group('Command Line Inputs')
    cli.add_argument('--params-file', required = False,  help='.par file containing parameters')
    cli.add_argument('--precision', type=str, default='float64',
                    help='Numeric precision: float16, float32, float64 (or 16/32/64)')

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

    main(P)

