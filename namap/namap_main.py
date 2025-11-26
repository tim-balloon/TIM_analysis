import numpy as np
import src.loaddata as ld
import src.detector as tod
import src.mapmaker as mp
import src.pointing as pt  
import src.beam as bm
import copy
from astropy import wcs 
import astropy.table as tb
import h5py
import argparse
import ast
import sys
from astropy.table import Table
from astropy.io import fits
import datetime
import os
import json

#for debugging purpose only
from IPython import embed

#for profilling purpose only
import tracemalloc
import time


def load_par_file(filepath):
    """
    Return as a dictionary the parameters stores in a .par file
    
    Parameters
    ----------
    filepath: str
        path and name of the parameter file
    Returns
    -------
    params: dictionary
        dictionary containing the loaded parameters
    """    

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
    """
    Main script to call Namap. 
    
    Parameters
    ----------
    P: dictionnary
        dictionnary of parameters
    nbdets: int
        Number of detectors max to be loaded. For testing purpose only. 
    Returns
    -------
    """    
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

    #----------------------------------------------------------------
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

    #-------- for profiling purpose only -------------
    if(nbdets is not None):
        result_rows = []
        # Loop over unique frequencies
        for freq in np.unique(filtered['Frequency']):
            sub = filtered[filtered['Frequency'] == freq]
            # take first N rows for this frequency
            result_rows.append(sub[:nbdets])

        # Concatenate back into a single table
        kid_num = Table(np.hstack(result_rows))['Name']

    else: kid_num = filtered['Name']
    #-------------------------------------------------

    print('Nb dets: ', len(kid_num))
    #----------------------------------------------------------------

    #load the table
    dettable = ld.det_table(kid_num, P['detector_table']) 
    det_off, _,_ = dettable.loadtable() #noise_det, resp
    
    
    #Cleaning data parameters
    highpassfreq = P['highpassfreq']
    polynomialorder = P['polynomialorder']
    despike_bool = P['despike']
    sigma,prominence = P['sigma'],P['prominence']
    sigma_clipping_bool = ['sigma_clipping']
    low_thresh, high_thresh = P['low_thresh'], P['high_thresh'] 
    #Beam convolution parameters
    convolution, std = P['gaussian_convolution'], P['std'] 
    #---------------------------------

    #----------------------------------
    #Load the data
    dataload = ld.data_value(filepath, kid_num, coord1, coord2, first_frame, num_frames,  DT, IT)
    det_data, coord1_data, coord2_data, lst_data, lat_data, spf_data, spf_coord, lat_spf = dataload.values()
    #-------------------------------

    #---------------------------------
    #First remove noise peaks and discard TODs with large & low variance. 
    #det_tod = tod.data_cleaned(det_data, kid_num, det_off, spf_data,  0, 0, despike_bool, sigma, prominence, sigma_clipping_bool, low_thresh,high_thresh ,DT)
    #cleaned_data, kid_num, det_off, rejected_detetectors_list = det_tod.data_clean() 
    #P['rejected detectors list'] = rejected_detetectors_list
    cleaned_data=det_data
    #---------------------------------

    #---------------------------------
    if(len(cleaned_data[0]) != len(coord1_data) or True): #<-- for testing purpose only

        
        zoomsyncdata = ld.frame_zoom_sync(filepath, cleaned_data, spf_data, coord1_data, coord2_data, spf_coord, first_frame, num_frames, 
                                            lst_data, lat_data,  lat_spf,  DT, IT, freq_target=P['downsample_frequency'])
        

        timemap, cleaned_data, coord1_data, coord2_data, lst_data, lat_data, turnarounds_flag = zoomsyncdata.sync_data() 
        P['bypass_synch'] = False
    else: 
        print('Bypass Synch')
        P['bypass_synch'] = True

    #---------------------------------

    #---------------------------------
    #Clean the TOD by removing smooth polynomial component and apply a high pass filter
    #det_tod = tod.data_cleaned(cleaned_data, kid_num, det_off, spf_data, highpassfreq, polynomialorder, False, 0, 0, False, 0,0, DT)
    #cleaned_data, _, _, _ = det_tod.data_clean()

    #Filter out the turnarounds
    if(P['remove_turnarounds'] and not P['bypass_synch']):
        for i in range(len(cleaned_data)): cleaned_data[i] = cleaned_data[i][turnarounds_flag==1]
        if P['save_downsampled_TODS']: timemap = timemap[turnarounds_flag==1]
        lst_data = lst_data[turnarounds_flag==1]
        lat_data = lat_data[turnarounds_flag==1]
        coord2_data = coord2_data[turnarounds_flag==1]
        coord1_data = coord1_data[turnarounds_flag==1] 
    #---------------------------------------------------------------

    
    #---------------------------------------------------------------
    #Apply detector's response
    #cleaned_data = [arr * resp for arr, resp in zip(cleaned_data, resp)]
    #---------------------------------

    if(P['save_downsampled_TODS']):

        tods_compressor = ld.compress_tods(P['output_hdf5'], kid_num, cleaned_data, P['downsample_frequency'], timemap, 
                                           coord1, coord2, coord1_data, coord2_data, 
                                           first_frame, num_frames, lst_data, lat_data,P,
                                            DT, IT, int8=P['int8'])
        tods_compressor.save_tods()

    else:
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
                    coadd=P['coadd'],   parang=parallactic, params=str(P)) #noise=noise_det,telcoord = P['telescope_coordinate'],
        
        maps.wcs_proj()
        map_values = maps.map2d()

        #--------------------

        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval
        zscale = ZScaleInterval()
        wcs = maps.w
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(map_values)
        fig, ax = plt.subplots(dpi=150,figsize=(8, 6),subplot_kw={'projection': wcs})
        img = ax.imshow(map_values, origin='lower', cmap='binary', vmin=vmin, vmax=vmax)
        fig.colorbar(img, ax=ax, label='Amplitude')
        ax.coords[0].set_format_unit('deg', decimal=True)  # RA
        ax.coords[1].set_format_unit('deg', decimal=True)  # De
        ax.set_ylabel('Dec [deg]')
        ax.set_xlabel('RA [deg]')
        ax.plot(P['crval'][0], P['crval'][1],'or', transform=ax.get_transform('world'))
        plt.show()
        
        #--------------------------------------------------    
        #Plot the maps
        maps.map_plot(data_maps = map_values, kid_num=kid_num)
        #--------------------------------------------------      
        
        if P['checkBeam'] and P['coadd']:
                
                vmin, vmax = zscale.get_limits(map_values)
                
                beam_value = bm.beam(map_values, )#param = self.beamparam

                beam_map = beam_value.beam_fit()

                param = beam_map[1]

                if isinstance(beam_map[0], str): print(beam_map[0])
                else: 

                    '''
                    plt.figure(figsize=(8, 6))
                    plt.contour(beam_map[0], levels=10, colors='red')
                    plt.imshow(beam_map[0], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                    plt.colorbar(label='Amplitude')
                    plt.title('2D Gaussian Fit Contours')
                    plt.xlabel('X pixel')
                    plt.ylabel('Y pixel')
                    '''

                    f = fits.PrimaryHDU(beam_map[0], header=wcs.to_header())
                    hdu = fits.HDUList([f])
                    hdr = hdu[0].header
                    hdr.set("map")
                    hdr.set("Datas")
                    hdr["BITPIX"] = ("64", "array data type")
                    hdr["BUNIT"] = 'MJy/sr'
                    hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
                    hdr["INFO"] = json.dumps(P, ensure_ascii=True)
                    hdu.writeto( os.getcwd()+'/fits_and_hdf5/'+P['beam_output'], overwrite=True)
                    hdu.close()
                
                plt.show()

    return 0

if __name__ == "__main__":

    #Repogroup.add_argument('-te', '--telemetry', action='store_true', help='For BLAST-TNG, specify if the data are coming from \


    '''
    If you want to modify this code, please create your own branch. 

    Instructions: 

    1/2: git clone from TIM_analysis/namap

    2/2: Download the TOD file: https://drive.google.com/file/d/1BnkEUj_yhPBPJte7ZgwxNHtMI75y8Nj6/view?usp=drive_link
    and put it in fits_and_hdf5/

    To run: python namap_main.py --params-file PAR_FILES/params_namap.par

    Left to be done:
        (I,Q) --> df/f (tod.kidsutils)
        Improve downsampling (ld.frame_zoom_sync)
        Implement respons correction and noise detectors ? Replaced by sigma clipping. 
        Test parallactic angle & telescope coordinates
        Improve TOD compression (ld.compress_tods)
    
        Focus (see src/beam.py)
        Boresight (see src/coordinate.py/rotate)
        Double check every par file key can be passed as argument
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

    cli.add_argument('--precision ',type=str, help='precision required in float64,float32,float16')
    cli.add_argument('--downsample_frequency ',type=float, help='The frequency to downsample the data to')
    cli.add_argument('--remove_turnarounds ',action='store_true', help='if True, remove data acquired during turnarounds')
    cli.add_argument('--save_downsampled_TODS ',action='store_true', help='if True, save the TOD in an .hdf5')
    cli.add_argument('--output_hdf5',type=str, help='name of the hdf5 file in which to store the TODs.')
    cli.add_argument('--output_map',type=str, help='name of the fits file to which save the map. Write name.fits.gz to automatically compress the fits file')


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

