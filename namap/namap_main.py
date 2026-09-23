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

def main(P, nbdets=None):


    """
    Main script to call Namap. 
    
    Parameters
    ----------
    P: dictionnary
        dictionnary of parameters
    nbdets: int
        Number of detectors max to be loaded. For profiling purpose only. 
        
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


    
    #-----------------------------------------------------------------------------------------

    #Frames to be loaded
    num_frames, first_frame, bufferframe = P['num_frames'], P['first_frame'], P['bufferframe']

    telemetry = P['telemetry']

    #Coordinates system for map making
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


    #-----------------------------------------------------------------------------------------

    # List of detectors to load
    filepath = P['input_file']
    btable = tb.Table.read(P['detector_table'], format='ascii.tab')

    # Select detectors from the good-behaving detector list
    if P['detectors_to_use'] is not None:
        good_kid_table = tb.Table.read(P['detectors_to_use'], format='ascii.tab')
        filtered = btable[np.isin(btable['Name'], good_kid_table['Name'])]

    # Select detectors per observed E.M frequency
    if P['frequencies'] is not None: filtered = btable[np.isin(btable['Frequency'], P['frequencies'])]

    if P['frequencies'] is None and P['detectors_to_use'] is None: filtered = btable

    # Select the number of detectors needed by the profiling code
    # for profiling purpose only    
    if(nbdets is not None):
        result_rows = []
        # Loop over unique frequencies
        for freq in np.unique(filtered['Frequency']):
            sub = filtered[filtered['Frequency'] == freq]
            # take first N rows for this frequency
            result_rows.append(sub[nbdets:])

        # Concatenate back into a single table
        kid_num = Table(np.hstack(result_rows))['Name']

    else: kid_num = filtered['Name']

    #-----------------------------------------------------------------------------------------
    

    #Cleaning data parameters
    downsample_frequency, fc = P['downsample_frequency'], P['antialiasing_filter_frequency']
    if(downsample_frequency is not None): downsample_bool = True
    else: downsample_bool = False
    highpassfreq = P['highpassfreq']
    polynomialorder = P['polynomialorder']
    despike_bool = P['despike']
    sigma,prominence = P['sigma'],P['prominence']
    sigma_clipping_bool = ['sigma_clipping']
    low_thresh, high_thresh = P['low_thresh'], P['high_thresh'] 

    #Beam convolution parameters
    convolution, std = P['gaussian_convolution'], P['std'] 
    

    #-----------------------------------------------------------------------------------------

    
    #Load the data
    dataload = ld.data_value(filepath, kid_num, 
                             coord1, coord2, 
                             first_frame, num_frames,
                             despike_bool, sigma, prominence, 
                             downsample_bool, downsample_frequency, fc,
                             DT, IT, bufferframe=bufferframe)
    
    dettime, det_data, ctime,  coord1_data, coord2_data, turnaround_flags, lst_data, lat_data, spf_data, spf_coord, lat_spf = dataload.values()
    

    #-----------------------------------------------------------------------------------------


    
    #Clean the TOD by clipping out of further analysis TODs with low or high variance,
    # then remove smooth polynomial component and apply a high pass filter to remaining TODs.
    
    det_tod = tod.data_cleaned(det_data, kid_num, spf_data, 
                               highpassfreq, polynomialorder, 
                               False, 0, 0, 
                               sigma_clipping_bool, low_thresh,high_thresh, 
                               DT)
    
    cleaned_data, kid_num, rejected_detetectors_list = det_tod.data_clean()
    P['rejected detectors list'] = rejected_detetectors_list
    

    #-----------------------------------------------------------------------------------------


    #Synchronize data with coordinates
    if(not P['bypass_synch']):
        zoomsyncdata = ld.frame_zoom_sync(dettime, cleaned_data, spf_data, 
                                          ctime, coord1_data, coord2_data, spf_coord, 
                                          turnaround_flags, lst_data, lat_data, lat_spf,  
                                          DT, IT)
        timemap, cleaned_data, coord1_data, coord2_data, lst_data, lat_data, turnarounds_flag = zoomsyncdata.sync_data() 


    #-----------------------------------------------------------------------------------------


    #Filter out the turnarounds
    if(P['remove_turnarounds'] and not P['bypass_synch'] ):
        for i in range(len(cleaned_data)): cleaned_data[i] = cleaned_data[i][turnarounds_flag==1]
        if P['save_TODS']: timemap = timemap[turnarounds_flag==1]
        lst_data = lst_data[turnarounds_flag==1]
        lat_data = lat_data[turnarounds_flag==1]
        coord2_data = coord2_data[turnarounds_flag==1]
        coord1_data = coord1_data[turnarounds_flag==1] 


    #-----------------------------------------------------------------------------------------

    if(P['save_downsampled_TODS']):

        #-----------------------------------------------------------------------------------------


        #Save the timestreams
        tods_compressor = ld.save_tods(P['output_tods'], 
                                       kid_num, cleaned_data, spf_data, timemap, 
                                       coord1, coord2, coord1_data, coord2_data, spf_data,timemap,
                                       first_frame, num_frames, lst_data, lat_data,P,            
                                        DT, IT)
        tods_compressor.fct_save_tods()


        #-----------------------------------------------------------------------------------------


    else:        

        #-----------------------------------------------------------------------------------------


        #load the table of detector offsets
        dettable = ld.det_table(kid_num, P['detector_table']) 
        det_off, _,_ = dettable.loadtable() #noise_det, resp

        #Offset with respect to star cameras in xEL and EL
        xsc_offset = (P['xsc_offset'],P['det_offset']) #needs to be tested with real offsets. 


        #-----------------------------------------------------------------------------------------

        
        #Correct telescope coordinates from detector offsets
        corr = pt.apply_offset(P['input_ctype'], coord1_data, coord2_data, P['ctype'], xsc_offset, DT,IT, det_offset = det_off, lst = lst_data, lat = lat_data, )
        coord1slice, coord2slice = corr.correction()
        

        #-----------------------------------------------------------------------------------------

        #Compute parallactic_angle
        parallactic=[]
        if P['telescope_coordinate']:
            for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
                tel = pt.utils(c1, c2, lst_data, lat_data)
                parallactic.append( tel.parallactic_angle() )
        else:
            for j, (c1, c2) in enumerate(zip(coord1slice,coord2slice)): 
                parallactic.append(np.zeros_like(c1, dtype=DT))
        

        #-----------------------------------------------------------------------------------------

        
        #Create the maps
        maps = mp.maps(P['ctype'], 
                    np.asarray([P['crpix'][0],P['crpix'][1]]), 
                    np.asarray([P['cdelt'][0],P['cdelt'][1]]), 
                    np.asarray([P['crval'][0], P['crval'][1]]), 
                    np.asarray([P['pixnum'][0],P['pixnum'][1]]), 
                    cleaned_data, coord1slice, coord2slice, convolution, std, P['output_map'], DT,IT,
                    coadd=P['coadd'],   parang=parallactic, params=P, variance_weighting=P['variance_weighting']) 
        
        maps.wcs_proj()
        map_values = maps.map2d()
        map_values = np.asarray(map_values)
        map_values /= ( P['cdelt'][0] * np.pi / 180 )**2
        wcs = maps.w
        

        #-----------------------------------------------------------------------------------------

            
        #Plot & save the maps
        maps.map_plot(data_maps = map_values, kid_num=kid_num)

        #The end 


        #-----------------------------------------------------------------------------------------
         
        #Fit a gaussian beam model to a map. 
        if P['checkBeam'] and P['coadd']:
                                
                beam_value = bm.beam(map_values, )#param = self.beamparam
                beam_map = beam_value.beam_fit()
                param = beam_map[1]

                if isinstance(beam_map[0], str): print(beam_map[0])
                else: 

                    f = fits.PrimaryHDU(beam_map[0], header=wcs.to_header())
                    hdu = fits.HDUList([f])
                    hdr = hdu[0].header
                    hdr.set("map")
                    hdr.set("Datas")
                    hdr["BITPIX"] = ("64", "array data type")
                    hdr["BUNIT"] = 'MJy/sr'
                    hdr["DATE"] = (str(datetime.datetime.now()), "date of creation")
                    hdr["INFO"] = json.dumps(P, ensure_ascii=True)
                    hdu.writeto( P['beam_output'], overwrite=True)
                    print('save '+P['beam_output'])
                    hdu.close()  


        #-----------------------------------------------------------------------------------------
        

        #Compute angular offsets of detectors from their obsevartion of the same source
        if P['check_offsets'] and not P['coadd']:

            LST_mean = lst_data.mean()
            lat_value = lat_data.mean()              

            corr = pt.apply_offset('RA and DEC', (wcs.wcs.crval[0],), (wcs.wcs.crval[1],), 'AZ and EL', DT,IT, lst = LST_mean, lat = lat_value, )
            azi_ref, alt_ref = corr.correction()
            
            file = P['detectors_output_file']    
            f = open(file, 'w')
            f.write("Name\tEL\tXEL\n")  # Column headers

            for i_det, name_kid in enumerate(kid_num):
   
                beam_value = bm.beam(map_values[i_det] )
                beam_map = beam_value.beam_fit()
                param = beam_map[1]

                if isinstance(beam_map[0], str): 
                    print(name_kid, beam_map[0])
                    f.write(f"{name_kid}\t None \t None \n") 
                else: 
                    params = beam_map[1]
                    cov = beam_map[2]
                    uncertainties = np.sqrt(np.diag(cov))
                    print(f'yo={params[2]:.2f} pm {uncertainties[2]:.2f} | xo={params[1]:.2f} pm {uncertainties[1]:.2f}' )
                    x_peak = params[1]; y_peak = params[2]
                    ra_deg, dec_deg = wcs.pixel_to_world_values(x_peak, y_peak)
                                        
                    conv = pt.apply_offset('RA and DEC',ra_deg,dec_deg,'AZ and EL',lst = LST_mean, lat = lat_value,)
                    AZ_dets_mine, EL_dets_mine = conv.correction()
                    daz2 = AZ_dets_mine - azi_ref
                    daz2 = (daz2 + 180) % 360 - 180
                    xel = daz2 * np.cos(np.radians(alt_ref))  #alt_ref_namap
                    delv = EL_dets_mine - alt_ref

                    f.write(f"{name_kid}\t{delv:3f}\t{xel:3f}\n")  # Tab-separated values
            f.close()                      


    return len(kid_num)

if __name__ == "__main__":

    # ----------------- ARGPARSE SETUP -----------------
    parser = argparse.ArgumentParser(description='NAMAP Parameters')

    cli = parser.add_argument_group('Command Line Inputs')
    cli.add_argument('--params-file', required = False,  help='.par file containing parameters')
    cli.add_argument('--precision', type=str, default='float64',
                    help='Numeric precision: float16, float32, float64 (or 16/32/64)')
    # command line parameter possibilities:
    cli.add_argument('--input_file',           type=str, default='datasets/TOD', help='Input file name')
    cli.add_argument('--detector_table',       type=str, help='Path to detector table (TSV format)')
    cli.add_argument('--detectors_to_use',     type=str, default = None,help='Path to detector to use table (TSV format)')
    cli.add_argument('--frequencies',          type=float, default=None, nargs=2, help='Frequency band in GHz, e.g. 715.0 719.0 to make map from')
    cli.add_argument('--num_frames', type=int, help='Integration time in seconds to be loaded')
    cli.add_argument('--first_frame', type=int, help='Starting frame index (in seconds)')
    cli.add_argument('--bufferframe', type=int, help='Buffer frame used is starting frame is 0 (in seconds)')
    cli.add_argument('--correction',           action='store_true', help='Enable pointing offset correction')
    cli.add_argument('--telemetry',            action='store_true', help='Specify if data is from telemetry (e.g. Mole)')
    cli.add_argument('--telescope_coordinate', action='store_true', help='Use telescope coordinates for mapmaking')
    cli.add_argument('--xystage',              action='store_true', help='Use XY stage coordinates')
    cli.add_argument('--xsc_offset',           type=float, help='Offset w.r.t. star cameras in xEL and EL')
    cli.add_argument('--det_offset',           type=float, help='Offset w.r.t. central detector in xEL and EL')
    cli.add_argument('--ctype',                type=str, help='Coordinate system to draw the map (e.g. "RA and DEC")')
    cli.add_argument('--input_ctype',          type=str, help='Coordinate system of pointing data in (e.g. "RA and DEC")')
    cli.add_argument('--lat',                  action='store_true', help='Use latitude flag (currently always True)')
    cli.add_argument('--lst',                  action='store_true', help='Use LST flag (currently always True)')
    cli.add_argument('--crpix',                type=float, nargs=2, help='Reference pixel position (2 floats)')
    cli.add_argument('--cdelt',                type=float, nargs=2, help='Pixel resolution along each axis in degrees (2 floats)')
    cli.add_argument('--crval',                type=float, nargs=2, help='Sky coordinates at reference pixel (2 floats)')
    cli.add_argument('--pixnum',               type=float, nargs=2, help='Number of pixels along each axis (2 floats)')
    cli.add_argument('--coadd',                action='store_true', help='Coadd detectors (True) or map each individually')
    cli.add_argument('--bypass_synch',         action='store_true', help='Skip data-telescope coords. synchronization')
    cli.add_argument('--save_raw_IQ_TODS',     action='store_true', help='save IQ tods in dirfile and stops')
    cli.add_argument('--save_downsampled_TODS',action='store_true', help='save cleaned timestreams and stops')
    cli.add_argument('--variance_weighting',   action='store_true', help='if True, weights TOD with their variance in the map-making')
    cli.add_argument('--downsample_frequency ',type=float, help='The frequency to downsample the data to')
    cli.add_argument('--antialiasing_filter_frequency ',type=float, help='The LPF frequency for anti-aliasing')
    cli.add_argument('--highpassfreq',         type=float, default = 0.1, help='High-pass filter cutoff frequency (Hz)')
    cli.add_argument('--polynomialorder',      type=int, default = 5,help='Polynomial order used to detrend TODs')
    cli.add_argument('--despike',              action='store_true', help='Flag to enable despiking of TODs')
    cli.add_argument('--sigma',                type=float, help='Sigma threshold for despike detection')
    cli.add_argument('--prominence',           type=float, help='Prominence threshold (in sigma units) for despiking')
    cli.add_argument('--sigma_clipping',       action='store_true', help='if True, remove TODs from further analysis based on their variance')
    cli.add_argument('--low_thresh',           type=float, help='Max value threshold (in sigma units) for clipping')
    cli.add_argument('--high_thresh',          type=float, help='Min value threshold (in sigma units) for clipping')
    cli.add_argument('--remove_turnarounds',action='store_true', help='if True, remove data acquired during turnarounds')
    cli.add_argument('--gaussian_convolution', action='store_true', help='Apply Gaussian convolution to map')
    cli.add_argument('--std', type=float, help='STD of Gaussian kernel in arcseconds')
    cli.add_argument('--output_map', type=str, default='coadd.fits', help='Output map file name')
    cli.add_argument('--output_tods', type=str, default='output.fits', help='Output dirfile name')
    cli.add_argument('--beam_output', type=str, default='beam_output.fits', help='best-fit beam angular map name')
    cli.add_argument('--checkBeam',action='store_true', help='if True, remove data acquired during turnarounds')
    cli.add_argument('--check_offsets',action='store_true', help='if True, remove data acquired during turnarounds')

    # Step 1: First parse only --params-file
    args_partial, remaining_argv = parser.parse_known_args()

    # Step 2: Load .par values if requested
    defaults = {}
    if args_partial.params_file:
        defaults = ld.load_params(args_partial.params_file)

    # Step 3: Set parser defaults from .par
    parser.set_defaults(**defaults)

    # Step 4: Parse full args
    args = parser.parse_args(remaining_argv)

    # Step 5: Convert Namespace to dictionary
    P = vars(args)

    main(P)