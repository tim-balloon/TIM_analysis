import sys
sys.path.append('../simulations')
import sim_tools_flatsky, sim_tools_tod
from make_correlated_noisy_TOD import gaussian_random_tod
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


if __name__ == "__main__":
    '''
    Generate noise TOD. 
    The noise TOD are frequency, detector, and time independant.
    The noise TODs are gaussian. 
    The noise TODs for pixels seeing the same beam (at different frequency bands) are correlated. 
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

    plot = True
    
    #Load the scan duration and generate the time coordinates with the desired acquisition rate. 
    T_duration = P['T_duration'] 
    dt = P['dt']*np.pi/3.14 #Make the timestep non rational to avoid some stripes in the hitmap. 
    T = np.arange(0,T_duration,dt) * 3600

    #TOD specs
    sample_freq = 1 / np.round(dt*3600,3)
    tod_len = len(T)
    tod_shape = [tod_len]
    freq_fft = np.fft.fftfreq(tod_len, 1/sample_freq) #TOD frequencies.
    inds = np.where(freq_fft>0) 
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

    #For each group of pixels seeing the frequency band: 
    for group in range(len(groups)):

        start = time.time()
        if(plot):
            fig, axs = plt.subplots(2,3,figsize=(9,6), dpi=150)
            axs[0,0].set_xlabel('$\\rm t_{int}$ [h]')
            axs[0,0].set_ylabel('$\\rm S_{\\nu}$ [Jy]')
            axs[0,0].set_title('Sky TOD')
            axs[0,1].set_title('TOD power spectrum')
            axs[0,1].set_xlabel('frequency [Hz]')
            axs[0,1].set_ylabel('Power amplitude $\\rm [Jy^2.s^{-2}$]')
            axs[0,1].set_xlim(1e-2, 1e1)
            axs[0,1].set_ylim(1e-18,1e-5)
        #------------------------------------------------------------------
        sky_tod = []
        fft_sky_tod = []
        #Load the sky timestreams (from strategy.py)
        H = h5py.File(tod_file, "a")
        for id, d in enumerate(groups.iloc[group]['Name']): 
            #if(id>2): continue
            f = H[f'kid_{d}_roach']
            tod = f['data'][()]
            sky_tod.append(tod)
            if(plot): axs[0,0].plot(T/3600,f['data'][()], alpha=0.1)
            curr_spec = ( np.fft.fft(tod) * (1/sample_freq) * np.conj( np.fft.fft(tod) * (1/sample_freq) ) / tod_len  ).real
            if(plot): axs[0,1].loglog(freq_fft[inds],curr_spec[inds], alpha=0.1)
            fft_sky_tod.append(curr_spec)
        H.close()
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        #define some detectors
        total_detectors = len(fft_sky_tod)
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
        if(plot):
            axs[0,1].loglog( freq, noise_powspec, label = r'Total', color = 'black' )
            axs[0,1].loglog( freq, noise_powspec_one_over_f, label = r'$1/f$', color = 'orangered' )
            axs[0,1].loglog( freq, noise_powspec_white, label = r'White', color = 'darkgreen' )
            axs[0,1].grid(True, lw = 0.2, alpha = 0.2, which = 'both')
            axs[0,1].legend(loc = 'lower right',)

            axs[1,0].loglog( freq, noise_powspec, label = r'Total', color = 'black' )
            axs[1,0].loglog( freq, noise_powspec_one_over_f, label = r'$1/f$', color = 'orangered' )
            axs[1,0].loglog( freq, noise_powspec_white, label = r'White', color = 'darkgreen' )
            axs[1,0].grid(True, lw = 0.2, alpha = 0.2, which = 'both')
            axs[1,0].legend(loc = 'lower right',)
            axs[1,0].set_xlabel('frequency [Hz]')
            axs[1,0].set_ylabel('Power amplitude $\\rm [Jy^2.s^{-2}$]')
            axs[1,0].set_title('$\\rm Noise_{\\nu}$ auto-power')
            axs[1,0].set_ylim(1e-18,1e-5)
            axs[1,1].set_ylabel('Cross-power $\\rm [Jy^2.s^{-2}$]')
            axs[1,1].set_title("$\\rm Noise_{\\nu, \\nu.}$ cross-power")
            axs[1,1].set_ylim(1e-18,1e-5)
            axs[1,1].set_xlabel('frequency [Hz]')

            axs[0,2].set_title('noise TOD')
            axs[0,2].set_xlabel('$\\rm t_{int}$ [h]')
            axs[0,2].set_ylabel('$\\rm S_{\\nu}$ [Jy]')
            axs[1,2].set_title('noisy TOD')
            axs[1,2].set_xlabel('$\\rm t_{int}$ [h]')
            axs[1,2].set_ylabel('$\\rm S_{\\nu}$ [Jy]')
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        #If the detectors don't all have a noise timestream yet, generate correlated power spectra

        tod_sims_dic = {}
        pspec_dic_sims = {}

        for sim_no in range( nsims ):

            #Generate un-correlated simulated timestreams
            tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq) 
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        #From the power spectra, generate random gaussian TODs and save them. 
        detector_combs_autos   = [[detector, detector] for detector in detector_array]

        H = h5py.File(tod_file, "a")
        for d1d2 in detector_combs_autos:
            d1, d2 = d1d2
            curr_theory = noise_powspec_dic[(d1, d2)]
            if(d1==d2):
                name = groups.iloc[group]['Name'][d1]
                f = H[f'kid_{name}_roach']
                noise_tod = tod_sim_arr[d1] #gaussian_random_tod(freq_fft, curr_spec_list[d1], res = (1/sample_freq), nx = tod_len)
                tod_tot =  sky_tod[d1] + noise_tod #noise_tod_list[d1] +
                if('uncorr_noise_data' in f): del f['uncorr_noise_data'] 
                f.create_dataset('uncorr_noise_data', data=noise_tod, compression='gzip', compression_opts=9)
                if('uncorr_noisy_data' in f): del f['uncorr_noisy_data'] 
                f.create_dataset('uncorr_noisy_data', data=tod_tot,   compression='gzip', compression_opts=9)

                if(plot): 
                    #axs[1,0].plot(freq_fft[inds], curr_spec_list[d1][inds], alpha=0.1)
                    axs[0,2].plot(T/3600, noise_tod, alpha=0.1)
                    axs[1,2].plot(T/3600, tod_tot, alpha=0.1)

                #Add a slope
                delta = 0.3 * (np.max(sky_tod[d1]) - np.min(sky_tod[d1]))  # 30% of the data range
                slope = delta / (T[-1] - T[0])
                data_with_slope = noise_tod + slope * (T - T[0])        
            
                #add 7-sigma peaks
                sigma = np.std(data_with_slope)
                peak_indices = np.random.choice(len(T), size=3, replace=False)
                data_with_peaks = data_with_slope.copy()
                peak_amplitude = 7 * sigma
                for idx in peak_indices:
                    data_with_peaks[idx] += peak_amplitude
                if('namap_data' in f): del f['namap_data'] 
                f.create_dataset('namap_data', data=data_with_peaks,   compression='gzip', compression_opts=9)
                
                '''
                plt.plot(T, data_with_slope, label='With 30% slope')
                plt.plot(T, data_with_peaks, 'r', label='With 7σ peaks', markersize=8, alpha=0.1)
                plt.xlabel('Time [s]')
                plt.ylabel('Data')
                plt.legend()
                plt.show()
                '''
                
        H.close()
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        if(plot):
            fig.tight_layout()
            fig.savefig(f'plot/uncorr_group_{group}_summary_plot.png')
            plt.close()
        #------------------------------------------------------------------
        end = time.time()
        timing = end - start
        print('')
        print(f'Generate the TODs of group {group} in {np.round(timing,2)} sec!')
        #------------------------------------------------------------------



'''
from scipy.interpolate import interp1d

# Step 1: create interpolation function from data1
interp_func = interp1d(t1, data1, kind='linear', fill_value='extrapolate')

# Step 2: evaluate at t2 points
resampled_data1 = interp_func(t2)
'''