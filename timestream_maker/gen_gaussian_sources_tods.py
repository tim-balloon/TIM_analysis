import astropy.units as u
import scipy.constants as cst
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
import datetime
import matplotlib.pyplot as plt
from hitmap_1detector import main_1det
from hitmap_array import main_arrays
from gen_timestreams import main_tod, gen_tod
from src.load_params import load_params, format_duration
from src.hdf5_fcts import save_tod_in_hdf5
from IPython import embed
import numpy as np 
import os
import pickle
from src.scan_fcts import *
from gen_detectors_arrays import gen_detectors_main
import h5py 
import shutil
from astropy.io import fits
from astropy.wcs import WCS

P = load_params('PAR_files/params_strategy.par')    

P['detectors_name_file'] = '/home/mvancuyck/Desktop/TIM_analysis/namap/TIM_kid_table_reduced_number_of_dets.tsv'
P['nb_pixel_SW'] = 2  #Number of pixel per frequency band in the SW array.
P['nb_pixel_LW'] = 2 #Number of pixel per frequency band in the LW array.
P['offset_SW'] =  1/3600 #[deg] separation in angle between 2 consecutive pixels for the SW array
P['offset_LW'] =  1/3600 #[deg] separation in angle between 2 consecutive pixels for the LW array
P['arrays_separation'] = 0 #[deg]

P['nb_channels_per_array'] = 1
P['T_duration'] = 10/60
P['output_path'] = f'fits_and_hdf5/'
P['scan'] ='raster'
P['az_size'] = 0.2
P['alt_step'] = 2/3600
P['alt_size'] = 50/3600
P['acquisition_frequency'] = 100
P['acquisition_frequency_coords'] = 100
P['vertical_steps'] = 60
P['N_scans'] = 1

P['output_path'] = '/home/mvancuyck/Desktop/TIM_analysis/namap/fits_and_hdf5/'
P['path']        = '/home/mvancuyck/Desktop/TIM_analysis/namap/fits_and_hdf5/'

P['output_name'] = f'TOD_on_1_source_with_1xbigger_sigma_PSF.hdf5'
P['file'] = f"cube_1source_with_1xbigger_sigma_PSF.fits" 

gen_detectors_main(P)

if( not os.path.isfile(P['output_path']+P['output_name']) or True ):
    print('')
    main_1det(P)
    #main_arrays(P)
    main_tod(P)
    
if(False):

    for factor_on_Sigma in (1,1.5,2,2.5): 

        for separation_in_arcsecs in (150.8, 226.1, 301.5, 376.9):

            P['output_name'] = f'TOD_on_2_sources_separated_by_{separation_in_arcsecs}_with_{factor_on_Sigma}xbigger_sigma_PSF.hdf5' 
            P['file'] = f"cube_2sources_separated_by_{separation_in_arcsecs}arcsecs_with_{factor_on_Sigma}xbigger_sigma_PSF.fits" 

            if( not os.path.isfile(P['output_path']+P['output_name']) or True):
                main_1det(P)
                #main_arrays(P)
                main_tod(P)
