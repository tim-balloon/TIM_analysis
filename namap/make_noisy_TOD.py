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


def gaussian_random_tod(l, clt, nx, res, l_cutoff=None):
    
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
    #Initiate the parameters
    
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
    det_dict = det_names_dict.groupby('Frequency')['Name'].first().to_dict()
    det_dict = dict(islice(det_dict.items(), 5))
    same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()
    
    tod_file=os.getcwd()+'/'+P['path']+'TOD_'+P['file'][:-5]+'.hdf5'

    #rough noise specs - similar to SPT (https://arxiv.org/pdf/2106.11202).
    tod_noise_level = P['tod_noise_level'] #in, for example, uK/\sqrt(seconds), units. (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    fknee = P['fknee']  ##0.05 #Hz (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    alphaknee = P['alphaknee'] 
    rho_one_over_f = P['rho_one_over_f']  #some level of 1/f correlation between detectors.

    #----------------------
    #Load the sky simulation to generate the TOD from
    dwcs = pickle.load( open(P['wcs_dict'], 'rb'))
    wcs = dwcs['wcs']
    #y_pixel_coords, x_pixel_coords = wcs.world_to_pixel_values(ra,dec)    
    xbins = np.arange(-0.5, wcs.pixel_shape[0]+0.5, 1)
    ybins = np.arange(-0.5, wcs.pixel_shape[1]+0.5, 1)
    #norm, edges = np.histogramdd(sample=(x_pixel_coords, y_pixel_coords), bins=(xbins,ybins),)

    #---------------------------------------------------

    for group in range(len(same_offset_groups)):
        start = time.time()
        
        sky_tod = []
        fft_sky_tod = []
        
        '''
        fig, axs = plt.subplots(2,3,figsize=(9,6), dpi=150)
        axs[0,0].set_xlabel('$\\rm t_{int}$ [h]')
        axs[0,0].set_ylabel('$\\rm S_{\\nu}$ [Jy]')
        axs[0,0].set_title('Sky TOD')
        axs[0,1].set_title('TOD power spectrum')
        axs[0,1].set_xlabel('frequency [Hz]')
        axs[0,1].set_ylabel('Power amplitude $\\rm [Jy^2.s^{-2}$]')
        axs[0,1].set_xlim(1e-2, 1e1)
        axs[0,1].set_ylim(1e-18,1e-5)
        '''

        H = h5py.File(tod_file, "a")
        for id, d in enumerate(same_offset_groups.iloc[group]['Name']): 
            #if(id>2): continue
            f = H[f'kid_{d}_roach']
            tod = f['data'][()]
            sky_tod.append(tod)

            #axs[0,0].plot(T/3600,f['data'][()], alpha=0.1)
            curr_spec = ( np.fft.fft(tod) * (1/sample_freq) * np.conj( np.fft.fft(tod) * (1/sample_freq) ) / tod_len  ).real
            #axs[0,1].loglog(freq_fft[inds],curr_spec[inds], alpha=0.1)
            fft_sky_tod.append(curr_spec)
        H.close()

        #define some detectors
        total_detectors = len(fft_sky_tod)
        detector_array = np.arange( total_detectors )
        detector_arr_to_plot = np.asarray( detector_array )

        #example noise model for auto- and cross-power between detectors
        freq, noise_powspec, noise_powspec_one_over_f, noise_powspec_white = sim_tools_tod.detector_noise_model(tod_noise_level, fknee, alphaknee, tod_len, sample_freq)
        cross_noise_powspec = sim_tools_tod.get_correlated_powspec(rho_one_over_f, noise_powspec_one_over_f, noise_powspec_one_over_f)
        noise_powspec_dic = {}
        for i in range(total_detectors):
            for j in range(total_detectors):
                if i == j:
                    noise_powspec_dic[i, j] = noise_powspec
                else:
                    noise_powspec_dic[i, j] = cross_noise_powspec
        '''
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
        
        '''
        tod_sims_dic = {}
        pspec_dic_sims = {}

        H = h5py.File(tod_file, "a")
        name = same_offset_groups.iloc[group]['Name'][-1]
        f = H[f'kid_{name}_roach']
        B = ('noise_data' not in f or 'noisy_data' not in f)
        H.close()
        if(B): 

            for sim_no in range( nsims ):
                bar = Bar('Processing Sim = %s of %s' %(sim_no+1, nsims), max=total_detectors)
                tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq)
                ###print( tod_sim_arr.shape ); ##sys.exit()
                #get the sim spectra now.
                curr_sim_pspec_dic = {}
                for (cntr1, tod1) in enumerate( tod_sim_arr ):
                    for (cntr2, tod2) in enumerate( tod_sim_arr ):
                        if cntr2<cntr1: continue       
                        curr_spec = ( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real
                        curr_sim_pspec_dic[(cntr1, cntr2)] = [freq_fft, curr_spec]
                    bar.next()
                bar.finish
                pspec_dic_sims[sim_no] = curr_sim_pspec_dic

        detector_combs_autos   = [[detector, detector] for detector in detector_array]
        detector_combs_crosses = [[detector1, detector2] for detector1 in detector_array for detector2 in detector_array if (detector1!=detector2 and detector1<detector2)]
        detector_combs = detector_combs_autos + detector_combs_crosses

        H = h5py.File(tod_file, "a")
        bar = Bar('Processing detectors', max=len(detector_combs))
        for d1d2 in detector_combs:
            d1, d2 = d1d2
            rowval, colval = np.where(detector_arr_to_plot == d1)[0][0], np.where(detector_arr_to_plot == d2)[0][0]
            curr_theory = noise_powspec_dic[(d1, d2)]
            #sims
            curr_spec_arr = []
            for sim_no in pspec_dic_sims:
                curr_freq, curr_spec = pspec_dic_sims[sim_no][(d1, d2)]
                curr_spec_arr.append( curr_spec )
            curr_spec_mean = np.mean( curr_spec_arr, axis = 0 )

            if(d1==d2):
                #axs[1,0].plot(freq_fft[inds], curr_spec_mean[inds], alpha=0.1)
                #axs[0,2].plot(T/3600, noise_tod, alpha=0.1)
                #axs[1,2].plot(T/3600, tod_tot, alpha=0.1)
                name = same_offset_groups.iloc[group]['Name'][d1]
                f = H[f'kid_{name}_roach']
                if('noise_data' not in f or 'noisy_data' not in f): 
                    noise_tod = gaussian_random_tod(freq_fft, curr_spec_mean, res = (1/sample_freq), nx = tod_len)
                    tod_tot = noise_tod + sky_tod[d1]
                    f.create_dataset('noise_data', data=noise_tod, compression='gzip', compression_opts=9)
                    f.create_dataset('noisy_data', data=tod_tot, compression='gzip', compression_opts=9)
            else:
                curr_theory = noise_powspec_dic[(d1, d2)]
                #axs[1,1].loglog( freq_fft[inds], curr_theory[inds], color = 'black', zorder = 100)
                #axs[1,1].loglog(freq_fft[inds], curr_spec_mean[inds], alpha=0.1)
            
            bar.next()
        
        #fig.tight_layout()
        #fig.savefig(f'plot/group_{group}_summary_plot.png')
        #plt.close()

        bar.finish
        #------------------
        H.close()
        end = time.time()
        timing = end - start
        print(f'Generate the TODs of group{group} in {np.round(timing,2)} sec!')


    '''
    
    
    #make plots of the input vs output power spectra.
    if total_detectors > 5:
        detector_arr_to_plot = detector_arr_to_plot[:5]
    detector_combs_for_plotting_autos = [[detector, detector] for detector in detector_arr_to_plot]
    detector_combs_for_plotting_crosses = [[detector1, detector2] for detector1 in detector_arr_to_plot for detector2 in detector_arr_to_plot if (detector1!=detector2 and detector1<detector2)]
    detector_combs_for_plotting = detector_combs_for_plotting_autos + detector_combs_for_plotting_crosses

    fsval = 12
    tr, tc = len(detector_arr_to_plot), len(detector_arr_to_plot)
    sbpl = 1
    figure(figsize = (tr+6, tc+6))
    subplots_adjust( hspace = 0.2, wspace = 0.1)

    for d1d2 in detector_combs_for_plotting:
        d1, d2 = d1d2

        rowval, colval = np.where(detector_arr_to_plot == d1)[0][0], np.where(detector_arr_to_plot == d2)[0][0]
        ax = subplot2grid((tr, tc), (rowval, colval), yscale = 'log', xscale = 'log')

        #plot sky data
        if(d1==d2 and d1<=4):  ax.plot(freq_fft[inds] , fft_sky_tod[d1][inds], c='g', alpha=0.1)


        curr_theory = noise_powspec_dic[(d1, d2)]
        #theory
        plot( freq_fft[inds], curr_theory[inds], color = 'black', label = r'Input', lw = 2., zorder = 100)

        #sims
        colorval = 'orangered'
        curr_spec_arr = []
        for sim_no in pspec_dic_sims:
            curr_freq, curr_spec = pspec_dic_sims[sim_no][(d1, d2)]
            curr_spec_arr.append( curr_spec )
            plot( curr_freq[inds], curr_spec[inds], color = colorval, lw = 0.1, alpha = 0.1)
        curr_spec_mean = np.mean( curr_spec_arr, axis = 0 )
        plot( curr_freq[inds], curr_spec_mean[inds], color = colorval, lw = 0.5, label = r'Sim mean')

        xlim(1e-2, 1e1); ylim(1e-18,1e-5)
        grid(True, lw = 0.2, alpha = 0.2, which = 'both')
        if rowval == colval:
            xlabel(r'Freqeuency [Hz]', fontsize = fsval-2)
            ylabel(r'Power $\mu$K$^{2}$ seconds', fontsize = fsval-2)
        else:
            setp(ax.get_xticklabels(), visible=False);# tick_params(axis='y',left='off')
            setp(ax.get_yticklabels(), visible=False);# tick_params(axis='y',left='off')
        if rowval == colval and rowval == 0: legend(loc = 1, fontsize = fsval - 4, ncol = 1)
        title(r'd%s x d%s' %(d1, d2), fontsize = fsval)
        sbpl += 1
    plt.show()
    show()
    '''