import sys
sys.path.append('../simulations')
import sim_tools_flatsky, sim_tools_tod
from make_correlated_noisy_TOD import gaussian_random_tod, add_peaks_to_timestream, add_polynome_to_timestream
import argparse
from strategy import load_params
import numpy as np
from pylab import *
from progress.bar import Bar
import pandas as pd
from itertools import islice
from IPython import embed
import h5py
import os
import astropy.units as u 
from scipy import interpolate
from astropy.wcs import WCS
from astropy.io import fits
import pickle
from progress.bar import Bar
import time
from multiprocessing import Pool, cpu_count

def make_uncorrelated_timestreams(total_detectors, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f, plot=False):


    freq_fft = np.fft.fftfreq(tod_len, 1/sample_freq) #TOD frequencies.
    inds = np.where(freq_fft>0) 

    #------------------------------------------------------------------
    #define some detectors
    total_detectors = len(total_detectors)
    detector_array = np.arange( total_detectors )
    detector_arr_to_plot = np.asarray( detector_array )
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    #Generate the noise model for auto- and cross-power between detectors
    freq, noise_powspec, noise_powspec_one_over_f, noise_powspec_white = sim_tools_tod.detector_noise_model(tod_noise_level, fknee, alphaknee, tod_len, sample_freq)
    noise_powspec_dic = {}
    for i in range(total_detectors):
        for j in range(total_detectors):
            if i == j:
                noise_powspec_dic[i, j] = noise_powspec
            else:
                noise_powspec_dic[i, j] = np.zeros_like(noise_powspec)
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    #If the detectors don't all have a noise timestream yet, generate correlated power spectra
    tod_sims_dic = {}
    pspec_dic_sims = {}
    for sim_no in range( nsims ):
        #Generate un-correlated simulated timestreams
        tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq) 
        tod_sims_dic[sim_no] = tod_sim_arr
    
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    #From the power spectra, generate random gaussian TODs and save them. 
    detector_combs_autos   = [[detector, detector] for detector in detector_array]

    noise_tod_list = []
    for d1d2 in detector_combs_autos:
        d1, d2 = d1d2
        tods = []
        for sim_no in range( nsims ):
            tods.append(tod_sims_dic[sim_no][d1])
        tods = np.asarray(tods)
        noise_tod_list.append( np.mean(tods, axis = 1) )

    return noise_tod_list

if __name__ == "__main__":
    '''
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

    #------------------------------------------------------------------------------------------
    #Initiate the parameters
    
    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    T = np.arange(0,T_duration,dt) * 3600

    #TOD specs
    sample_freq = 1 / np.round(dt*3600,3)
    tod_len = len(T)
    tod_shape = [tod_len]
    #get params for sim generation
    fmin = P['fmin']
    fmax = P['fmax']

    #sim specs
    nsims = P['nsim_noise']

    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    groups = det_names_dict.groupby(['Frequency'])['Name'].apply(list).reset_index()
    
    tod_file=os.getcwd()+'/'+P['path']+'TOD_'+P['file'][:-5]+'.hdf5'

    #rough noise specs - similar to SPT (https://arxiv.org/pdf/2106.11202).
    tod_noise_level = P['tod_noise_level'] #in, for example, uK/\sqrt(seconds), units. (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    fknee = P['fknee']  ##0.05 #Hz (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    alphaknee = P['alphaknee'] 
    rho_one_over_f = P['rho_one_over_f']  #some level of 1/f correlation between detectors.
    #------------------------------------------------------------------------------------------

    det_names_dict = pd.read_csv(P['detectors_name_file'], sep='\t')
    #Each pixel with the same offset sees the same beam, but in different frequency band. 
    same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()


    for group in range(len(same_offset_groups)):

        start = time.time()
        total_detectors = len(same_offset_groups.iloc[group]['Name'])
        tod_list = make_uncorrelated_timestreams(total_detectors, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f)

        print('saving')
        H = h5py.File(tod_file, "a")    
        for j, (tod, name) in  enumerate(zip(tod_list, same_offset_groups.iloc[group]['Name'])):
            f = H[f'kid_{name}_roach']
            sky_tod = f['data']
            data_with_slope = add_polynome_to_timestream(sky_tod, T) + tod
            data_with_peaks = add_peaks_to_timestream(data_with_slope)
            if('corr_noise_data' in f): del f['corr_noise_data'] 
            if('corr_noisy_data' in f): del f['corr_noisy_data'] 
            f.create_dataset('corr_noise_data', data=tod, compression='gzip', compression_opts=9)
            f.create_dataset('corr_noisy_data', data=tod+sky_tod, compression='gzip', compression_opts=9)
            if('namap_data' in f): del f['namap_data'] 
            f.create_dataset('namap_data', data=data_with_peaks,   compression='gzip', compression_opts=9)
        H.close()
    
        end = time.time()
        timing = end - start
        
        print(f'Generate the TODs of group {group} in {np.round(timing,2)} sec!')
    #------------------------------------------------------------------