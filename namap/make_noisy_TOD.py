import sys
sys.path.append('../simulations')
import sim_tools_flatsky, sim_tools_tod
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
import matplotlib
matplotlib.use("Agg")
from multiprocessing import Pool, cpu_count

_args = None

def worker_init(*args):
    global _args
    _args = args

def worker_tod(curr_spec_list):
    global _args
    freq_fft, sample_freq, tod_len = _args 
    TOD_list = []
    for pk in curr_spec_list:
        TOD_list.append(gaussian_random_tod(freq_fft, pk, res = (1/sample_freq), nx = tod_len))
    return np.asarray(TOD_list)

def gaussian_tod_pll(freq_fft, curr_spec_list, sample_freq, tod_len, ncpus):
    with Pool(ncpus, initializer=worker_init, initargs=(freq_fft, sample_freq, tod_len )) as p:
        # Transform full cube (nchan, npix, npix) as (npix*npix, nchan)
        tod_list = p.map(worker_tod, np.array_split(curr_spec_list, ncpus) )
    tod_list_final = np.vstack(tod_list) 
    return tod_list_final

def worker_model(curr_sim_pspec_dic_for_worker):
    global _args
    freq_fft, sample_freq, tod_len = _args 
    spec_list = []
    for tod1, tod2 in curr_sim_pspec_dic_for_worker:
        spec_list.append(( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real )
    return np.asarray(spec_list)
    curr_spec = ( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real
    curr_sim_pspec_dic[(cntr1, cntr2)] = [freq_fft, curr_spec]

def model_pll(freq_fft, tod_sim_arr, sample_freq, tod_len, ncpus):
    embed()

    curr_sim_pspec_dic_for_worker = []
    for (cntr1, tod1) in enumerate( tod_sim_arr ):
        for (cntr2, tod2) in enumerate( tod_sim_arr ):
            if cntr2<cntr1: continue  
            else: curr_sim_pspec_dic_for_worker.append((tod1, tod2))
    
    with Pool(ncpus, initializer=worker_init, initargs=(freq_fft, sample_freq, tod_len )) as p:
        curr_sim_pspec = p.map(worker_model, np.array_split(curr_sim_pspec_dic_for_worker, ncpus) )

    curr_sim_pspec_dic = np.vstack(curr_sim_pspec) 
    return curr_sim_pspec_dic

def gaussian_random_tod(l, clt, nx, res, l_cutoff=None):

    """
    Generate a gaussian random noise timestream from a power spectrum.  

    Parameters
    ----------
    l:  array
        wavenumber array
    clt: array
        power spectrum amplitude array
    nx: int
        the number of elements of the timestream
    res: float
        the (time) resolution
    l_cutoff: float
        maximum wavenumber used in the generation of the timestream. 

    Returns
    -------
    real_space_tod: array
        the generated timestream. 
    """ 
    
    lmap_rad_x = nx*res
    #Generate gaussian amplitudes
    norm = 1/res

    np.random.seed()

    noise = np.random.normal(loc=0, scale=1, size=nx)
    dmn1  = np.fft.fft( noise )

    if(l_cutoff is not None): lmax = np.minimum(l_cutoff, l.max()) 
    else: lmax = l.max()

    cl_map = np.zeros(l.shape) 
        
    w = np.where((l>0) & (l<=lmax))

    f = interpolate.interp1d( l, clt,  kind='linear')
    cl_map[w] = f(l[w])
    w1 = np.where( cl_map <= 0)
    if(w1[0].shape[0] != 0): cl_map[w1] = 0
    #Fill amn_t
    amn_t = dmn1 * norm * np.sqrt( cl_map )
    
    #Output map
    real_space_tod = np.real(np.fft.ifft( amn_t ))
    
    return real_space_tod


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

    plot = False
    
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
    #Each pixel with the same offset sees the same beam, but in different frequency band. 
    same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()
    
    tod_file=os.getcwd()+'/'+P['path']+'TOD_'+P['file'][:-5]+'.hdf5'

    #rough noise specs - similar to SPT (https://arxiv.org/pdf/2106.11202).
    tod_noise_level = P['tod_noise_level'] #in, for example, uK/\sqrt(seconds), units. (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    fknee = P['fknee']  ##0.05 #Hz (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    alphaknee = P['alphaknee'] 
    rho_one_over_f = P['rho_one_over_f']  #some level of 1/f correlation between detectors.
    #------------------------------------------------------------------------------------------

    #For each group of pixels seeing the same beam: 
    for group in range(len(same_offset_groups)):

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
        for id, d in enumerate(same_offset_groups.iloc[group]['Name']): 
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
        cross_noise_powspec = sim_tools_tod.get_correlated_powspec(rho_one_over_f, noise_powspec_one_over_f, noise_powspec_one_over_f)
        noise_powspec_dic = {}
        for i in range(total_detectors):
            for j in range(total_detectors):
                if i == j:
                    noise_powspec_dic[i, j] = noise_powspec
                else:
                    noise_powspec_dic[i, j] = cross_noise_powspec
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
        #Check if all the detectors in the group already have a noise timestream. 
        H = h5py.File(tod_file, "a")
        name = same_offset_groups.iloc[group]['Name'][-1]
        f = H[f'kid_{name}_roach']
        B = ('noise_data' not in f or 'noisy_data' not in f)
        H.close() 
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        #If the detectors don't all have a noise timestream yet, generate correlated power spectra
        if(B or True): 

            tod_sims_dic = {}
            pspec_dic_sims = {}

            for sim_no in range( nsims ):

                #Generate un-correlated simulated timestreams
                tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq) 
                '''             
                bar = Bar('Processing Sim = %s of %s' %(sim_no+1, nsims), max=total_detectors)
                #get the correlated power spectra now.
                curr_sim_pspec_dic = {}
                for (cntr1, tod1) in enumerate( tod_sim_arr ):
                    for (cntr2, tod2) in enumerate( tod_sim_arr ):
                        if cntr2<cntr1: continue       
                        curr_spec = ( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real
                        curr_sim_pspec_dic[(cntr1, cntr2)] = [freq_fft, curr_spec]
                    bar.next()
                bar.finish
                '''
                curr_sim_pspec_dic = model_pll(freq_fft, tod_sim_arr, sample_freq, tod_len, 24)
                pspec_dic_sims[sim_no] = curr_sim_pspec_dic
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        #From the power spectra, generate random gaussian TODs and save them. 
        detector_combs_autos   = [[detector, detector] for detector in detector_array]
        detector_combs_crosses = [[detector1, detector2] for detector1 in detector_array for detector2 in detector_array if (detector1!=detector2 and detector1<detector2)]
        detector_combs = detector_combs_autos + detector_combs_crosses

        curr_spec_list = []
        for d1d2 in detector_combs_autos:
            d1, d2 = d1d2
            curr_theory = noise_powspec_dic[(d1, d2)]
            #sims
            curr_spec_arr = []
            for sim_no in pspec_dic_sims:
                curr_freq, curr_spec = pspec_dic_sims[sim_no][(d1, d2)]
                curr_spec_arr.append( curr_spec )
            curr_spec_list.append( np.mean( curr_spec_arr, axis = 0 ) )

        #noise_tod = gaussian_random_tod(freq_fft, curr_spec_mean, res = (1/sample_freq), nx = tod_len)
        noise_tod_list = gaussian_tod_pll(freq_fft, curr_spec_list, sample_freq, tod_len, 24)


        for d1d2 in detector_combs:
            d1, d2 = d1d2
            curr_theory = noise_powspec_dic[(d1, d2)]
            #sims
            if(d1==d2):
                if(plot): 
                    axs[1,0].plot(freq_fft[inds], curr_spec_list[d1][inds], alpha=0.1)
                    axs[0,2].plot(T/3600, noise_tod_list[d1], alpha=0.1)
                    axs[1,2].plot(T/3600, tod_tot, alpha=0.1)
                name = same_offset_groups.iloc[group]['Name'][d1]
                f = H[f'kid_{name}_roach']
                if('noise_data' not in f or 'noisy_data' not in f): 
                    #noise_tod = gaussian_random_tod(freq_fft, curr_spec_mean, res = (1/sample_freq), nx = tod_len)
                    tod_tot = noise_tod_list[d1] + sky_tod[d1]
                    f.create_dataset('noise_data', data=noise_tod_list[d1], compression='gzip', compression_opts=9)
                    f.create_dataset('noisy_data', data=tod_tot, compression='gzip', compression_opts=9)
            else:
                curr_theory = noise_powspec_dic[(d1, d2)]
                if(plot): 
                    axs[1,1].loglog( freq_fft[inds], curr_theory[inds], color = 'black', zorder = 100)
                    axs[1,1].loglog(freq_fft[inds], curr_spec_list[d1][inds], alpha=0.1)
            
            bar.next()
        #------------------------------------------------------------------

        #------------------------------------------------------------------
        if(plot):
            fig.tight_layout()
            fig.savefig(f'plot/group_{group}_summary_plot.png')
            plt.close()
        #------------------------------------------------------------------

        bar.finish

        H.close()
        end = time.time()
        timing = end - start
        print('')
        print(f'Generate the TODs of group{group} in {np.round(timing,2)} sec!')
        #------------------------------------------------------------------

