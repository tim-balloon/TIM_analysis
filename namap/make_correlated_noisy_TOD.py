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
from multiprocessing import Pool, cpu_count
#import matplotlib
#matplotlib.use('Agg')
from itertools import chain

#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

_args = None

def worker_init(*args):
    global _args
    _args = args

def worker_model(grps):
    global _args
    same_offset_groups, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f  = _args 

    noise_list = []
    name_list = []
    for group in grps:
        opf.write(f'Generate group {group} \n'); opf.flush()
        total_detectors = 2 #len(same_offset_groups.iloc[group]['Name'])
        name_list.append(same_offset_groups.iloc[group]['Name'])
        noise_list.append(make_correlated_timestreams(total_detectors, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f))
    return (noise_list, name_list)

def make_all_tods_pll(same_offset_groups, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f):
    '''
    Parallelization on 24 nodes of the noise TODs generation. One node now process several group of pixels which see the same beam. 
    Parameters
    ----------
    same_offset_groups: pandas.core.frame.DataFrame
        pixels grouped by the beam they are seeing 
    T: array
        the time timestream
    sample_freq: float
        the qcquisition rate of T and of the sky TODs. 
    tod_len: int
        lenght of the noise TODs
    tod_shape: list
        list of the TODs dimensions
    fmin: float
        minimum frequency from which to generate the noise TOD. NOT IMPLEMENTED 
    fmax: float
        maximum frequency from which to generate the noise TOD. NOT IMPLEMENTED 
    nsims: int
        number of simulation to generate. 
    tod_file: 
        name of the hdf5 cntaining the sky TODs and in which to save the noise TODs. 
    tod_noise_level: float

    '''

    grps = np.arange(len(same_offset_groups))
    opf.write('start // \n'); opf.flush()
    with Pool(ncpus, initializer=worker_init, initargs=(same_offset_groups, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f )) as p:
        results = p.map(worker_model, np.array_split(grps, ncpus) )
    opf.write('done \n'); opf.flush()
    
    final, names = zip(*results)
    opf.write('get tods \n'); opf.flush()
    final = list(chain.from_iterable(final))
    opf.write('get names \n'); opf.flush()
    names = list(chain.from_iterable(names))
    embed()
    opf.write('saving \n'); opf.flush()
    H = h5py.File(tod_file, "a")    
    for i, (tod_list, list_names) in enumerate(zip(final, names)):
        for j, (tod, name) in  enumerate(zip(tod_list, list_names)):
            f = H[f'kid_{name}_roach']
            sky_tod = f['data']
            if('corr_noise_data' in f): del f['corr_noise_data'] 
            f.create_dataset('corr_noise_data', data=tod, compression='gzip', compression_opts=9)
    H.close()
    
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

def make_correlated_timestreams(total_detectors, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f, plot=False):

    '''
    For pixels seeing the same beam, but at different frequency bands, 
    this function generates noise timestreams with the 1/f correlated,
    due to all the pixel seeing the same atmosphere. 

    Parameters
    ----------
    total_detectors: int
        the number of detectors to generate noise for. 
    T: array
        the time timestream
    sample_freq: float
        the qcquisition rate of T and of the sky TODs. 
    tod_len: int
        lenght of the noise TODs
    tod_shape: list
        list of the TODs dimensions
    fmin: float
        minimum frequency from which to generate the noise TOD. NOT IMPLEMENTED 
    fmax: float
        maximum frequency from which to generate the noise TOD. NOT IMPLEMENTED 
    nsims: int
        number of simulation to generate. 
    tod_file: 
        name of the hdf5 cntaining the sky TODs and in which to save the noise TODs. 
    tod_noise_level: float

    fknee: float

    alphaknee: float

    rho_one_over_f: float

    plot: bool
        If to plot a summary plot for the group of pixels or not.

    Returns
    -------
    noise_tods_list: list
        list containing the n=total_detectors noise timestreams.

    '''

    freq_fft = np.fft.fftfreq(tod_len, 1/sample_freq) #TOD frequencies.
    inds = np.where(freq_fft>0) 

    #------------------------------------------------------------------
    #define some detectors
    detector_array = np.arange( total_detectors )
    detector_combs_autos   = [[detector, detector] for detector in detector_array]
    detector_combs_crosses = [[detector1, detector2] for detector1 in detector_array for detector2 in detector_array if (detector1!=detector2 and detector1<detector2)]
    detector_combs = detector_combs_autos + detector_combs_crosses
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
    #If the detectors don't all have a noise timestream yet, generate correlated power spectra
    tod_sims_dic = {}
    pspec_dic_sims = {}
    for sim_no in range( nsims ):
        #Generate un-correlated simulated timestreams
        tod_sim_arr = sim_tools_flatsky.make_gaussian_realisations(freq_fft, noise_powspec_dic, tod_shape, 1./sample_freq) 
        #get the correlated power spectra now.
        curr_sim_pspec_dic = {}
        for (cntr1, tod1) in enumerate( tod_sim_arr ):
            for (cntr2, tod2) in enumerate( tod_sim_arr ):
                if cntr2<cntr1: continue       
                curr_spec = ( np.fft.fft(tod1) * (1/sample_freq) * np.conj( np.fft.fft(tod2) * (1/sample_freq) ) / tod_len  ).real
                curr_sim_pspec_dic[(cntr1, cntr2)] = [freq_fft, curr_spec]
        pspec_dic_sims[sim_no] = curr_sim_pspec_dic
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    #From the power spectra, generate random gaussian TODs and save them. 
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

    noise_tods_list = []

    #opf.write(f'get noise tod'); opf.flush()
    for d1d2 in detector_combs:
        d1, d2 = d1d2
        curr_theory = noise_powspec_dic[(d1, d2)]
        if(d1==d2):
            noise_tod = gaussian_random_tod(freq_fft, curr_spec_list[d1], res = (1/sample_freq), nx = tod_len)
            noise_tods_list.append(noise_tod)
    #------------------------------------------------------------------

    return noise_tods_list

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
    parser.add_argument('--non_iteractive', help = "load directly Mbb", action="store_true")

    args = parser.parse_args()

    P = load_params(args.params)

    if(args.non_iteractive): 
        import matplotlib
        matplotlib.use("Agg")

    #------------------------------------------------------------------------------------------

    file_path = "log_correlated_plll_timestreams.txt"
    if os.path.exists(file_path): os.remove(file_path)
    opf = open(file_path, 'a')

    #------------------------------------------------------------------------------------------
    #Initiate the parameters
    pll = True
    ncpus = 24
    
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
    #Each pixel with the same offset sees the same beam, but in different frequency band. 
    same_offset_groups = det_names_dict.groupby(['XEL', 'EL'])['Name'].apply(list).reset_index()
    
    tod_file= P['path']+'TOD_noise_notpll_'+P['file'][:-5]+'.hdf5'

    #rough noise specs - similar to SPT (https://arxiv.org/pdf/2106.11202).
    tod_noise_level = P['tod_noise_level'] #in, for example, uK/\sqrt(seconds), units. (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    fknee = P['fknee']  ##0.05 #Hz (Fig. 11 of https://arxiv.org/pdf/2106.11202).
    alphaknee = P['alphaknee'] 
    rho_one_over_f = P['rho_one_over_f']  #some level of 1/f correlation between detectors.
    #------------------------------------------------------------------------------------------
    
    #Use the // OR the un// version. 
    #------------------------------------------------------------------------------------------
    #// version 
    
    if(pll):
        start = time.time()
        make_all_tods_pll(same_offset_groups, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f)
        end = time.time()
        timing = end - start
        opf.write(f'Generate all the TODs in {np.round(timing,2)} sec!') ; opf.flush()
    
    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    #un// version
    else:
        #For each group of pixels seeing the same beam: 
        for group in range(len(same_offset_groups)):

            opf.write(f'starting group {group} \n'); opf.flush()
        
            start = time.time()
            total_detectors = len(same_offset_groups.iloc[group]['Name'])
            
            tod_list = make_correlated_timestreams(total_detectors, T, sample_freq, tod_len, tod_shape, fmin, fmax, nsims, tod_file, tod_noise_level, fknee, alphaknee, rho_one_over_f)
            
            opf.write('saving \n'); opf.flush()
            H = h5py.File(tod_file, "a")    
            for j, (tod, name) in  enumerate(zip(tod_list, same_offset_groups.iloc[group]['Name'])):
                namegrp = f'kid_{name}_roach'
                if namegrp not in H: grp = H.create_group(namegrp)
                else:                grp = H[namegrp]
                if('corr_noise_data' in grp): del grp['corr_noise_data'] 
                grp.create_dataset('corr_noise_data', data=tod, compression='gzip', compression_opts=9)
            H.close()
            
            end = time.time()
            timing = end - start
            
            opf.write(f'Generate the TODs of group {group} in {np.round(timing,2)} sec! \n'); opf.flush()
    #------------------------------------------------------------------
    opf.close()