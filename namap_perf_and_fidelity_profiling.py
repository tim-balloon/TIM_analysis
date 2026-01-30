import time
import tracemalloc
import pickle
import cProfile
import pstats
import matplotlib.pyplot as plt
import sysconfig, sys, site, os, io, copy
from astropy.io import fits
import platform
import psutil
from unittest import mock
import pickle
from collections import namedtuple
import glob
from pathlib import Path
import scipy.constants as cst
import numpy as np
import astropy.table as tb
from matplotlib.pyplot import cm

DESKTOP = Path.home() / "/home/mvancuyck/Desktop/TIM_analysis/"

sys.path.insert(0, str(DESKTOP / "namap"))
sys.path.insert(0, str(DESKTOP / "timestream_maker"))

from namap_main import namap_main
from gen_timestreams import *
from hitmap_1detector import *
from gen_detectors_arrays import *
import namap.src.loaddata as ld
import namap.src.detector as det
import namap.src.map_power_spectrum as aps


def test_namap_coadded_map_fidelity(dict_coadded_map_fidelity_file, load_directly = False):

    if(load_directly): dict_tods_fidelity = pickle.load( open(dict_tods_fidelity_file, 'rb'))
    else: 
        if(os.path.isfile(dict_tods_fidelity_file)): dict_tods_fidelity = pickle.load( open(dict_tods_fidelity_file, 'rb'))
        else: dict_tods_fidelity = {}

        dict_tods_fidelity['fft_freq_bounds'] = (freq_min_for_psd, freq_max_for_psd )

        for t_int in t_int_list:

            T_key = f'T = {t_int:.1f} min'
            dict_tods_fidelity.setdefault(T_key, {})  

            for downsample_frequency in downsampled_freq_list:

                df_key = f'downsample_frequency = {downsample_frequency:.1f} Hz'
                dict_tods_fidelity[T_key].setdefault(df_key, {})  

                for prec in precision_list:

                    dict_tods_fidelity[T_key][df_key].setdefault(prec, {})  

                    #-------------------------------------------
                    P_namap['hdf5_file'] = P['output_path']+f'TOD_{t_int:.1f}min.hdf5' 
                    P_namap['remove_turnarounds'] = True
                    P_namap['save_downsampled_TODS'] = False
                    P_namap['downsample_frequency'] = downsample_frequency
                    P_namap['output_hdf5'] = P['output_path']+f'namap_downsampled_TOD_{t_int:.1f}min_{downsample_frequency:.1f}Hz_{prec}.hdf5' 
                    P_namap['num_frames']  = int(t_int*60+1) #integration time in seconds to be loaded. 
                    P_namap['first_frame'] = 0 #Starting time in second to loaded
                    P_namap['precision'] = prec
                    P_namap['output_map'] = P['output_path']+f'namap_downsampled_TOD_{t_int:.1f}min_{100:.1f}Hz_{prec}_coadd_map.fits' 

                    if(not os.path.isfile(P_namap['output_map'])): 
                        print('Run Namap to make maps')
                        print(f'Run Namap on the {t_int}min timestreams.')
                        namap_main(P_namap)

                    created_map = fits.getdata(P_namap['output_map'])
                    hdr = fits.getheader(P_namap['output_map'])

                    from astropy.visualization import ZScaleInterval
                    zscale = ZScaleInterval()
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    vmin, vmax = zscale.get_limits(created_map)
                    fig, (ax, axp) = plt.subplots(1,2,figsize=(6,6))
                    im = ax.imshow(created_map, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)  # size can be a percentage or absolute
                    fig.colorbar(im, cax=cax, label='Amplitude')

                    pk = aps.angular_power_spectrum(created_map,hdr['CDELT1']*60, delta_k_over_k=0.1)
                    pk_mes, k = pk.p2()
                    axp.step(k, pk_mes, where='mid', c='k')
                    axp.set_ylabel('P(k) [$\\rm Jy^2/sr$]')
                    axp.set_xlabel('k [$\\rm arcmin^{-1}$]')
                    axp.set_yscale('log')
                    axp.set_xscale('log')
                    fig.tight_layout()
                    plt.show()

                    embed()
    return 0

def test_namap_tods_fidelity(dict_tods_fidelity_file, load_directly = False):

    if(load_directly): dict_tods_fidelity = pickle.load( open(dict_tods_fidelity_file, 'rb'))
    else: 
        if(os.path.isfile(dict_tods_fidelity_file)): dict_tods_fidelity = pickle.load( open(dict_tods_fidelity_file, 'rb'))
        else: dict_tods_fidelity = {}

        dict_tods_fidelity['fft_freq_bounds'] = (freq_min_for_psd, freq_max_for_psd )

        for t_int in t_int_list:

            T_key = f'T = {t_int:.1f} min'
            dict_tods_fidelity.setdefault(T_key, {})  

            for downsample_frequency in downsampled_freq_list:

                df_key = f'downsample_frequency = {downsample_frequency:.1f} Hz'
                dict_tods_fidelity[T_key].setdefault(df_key, {})  

                for prec in precision_list:

                    dict_tods_fidelity[T_key][df_key].setdefault(prec, {})  

                    #-------------------------------------------
                    P_namap['hdf5_file'] = P['output_path']+f'TOD_{t_int:.1f}min.hdf5' 
                    P_namap['remove_turnarounds'] = True
                    P_namap['save_downsampled_TODS'] = True
                    P_namap['downsample_frequency'] = downsample_frequency
                    P_namap['output_hdf5'] = P['output_path']+f'downsampled_TOD_{t_int:.1f}min_{downsample_frequency:.1f}Hz_{prec}.hdf5' 
                    P_namap['num_frames']  = int(t_int*60+1) #integration time in seconds to be loaded. 
                    P_namap['first_frame'] = 0 #Starting time in second to loaded
                    P_namap['precision'] = prec

                    if(not os.path.isfile(P_namap['output_hdf5'])): 
                        print(f'Run Namap on the {t_int}min timestreams.')
                        namap_main(P_namap)
                    #-------------------------------------------

                    kid_num = load_kids(P_namap)

                    #-------------------------------------------
                    #Load the reference TODs
                    data_load = ld.data_value(P_namap['hdf5_file'], kid_num, 'AZ', 'EL', P_namap['first_frame'], P_namap['num_frames'],  DT, IT)
                    det_data_original, _, _, _, _, _, _, _ = data_load.values()
                    timestamps_original = ld.data_value.loaddata(P_namap['hdf5_file'], f'data_time', DT, P_namap['num_frames'], P_namap['first_frame']) 
                    flags_original = ld.data_value.loaddata(P_namap['hdf5_file'], f'data_turnaround_flags', DT, P_namap['num_frames'], P_namap['first_frame']) 
                    acq_freq_original, chunks_original = make_chunks(timestamps_original,det_data_original,P['acquisition_frequency'],flags_original)
                    #-------------------------------------------
                    
                    #-------------------------------------------
                    #Load the downsampled TODs
                    H = h5py.File(P_namap['output_hdf5'], "a")
                    downsampled_tods = H['TODs']['data'][()]
                    timestamps_downsampled = H['timestamps']['data'][()]
                    acq_freq_downsampled, chunks_downsampled = make_chunks(timestamps_downsampled,downsampled_tods,P_namap['downsample_frequency'])
                    H.close()
                    #-------------------------------------------

                    psds_original = []
                    psds_downsampled = []
                    psds_relative_error = []
                    for i, (chunk, chunk_processed) in enumerate(zip(chunks_original,chunks_downsampled)):

                        for det_id, (det_tod, det_tod_processed) in enumerate(zip(chunk,chunk_processed)):
                            if(len(det_tod) <= 1): continue

                            freq, psd = tod_psd(det_tod, acq_freq_original)
                            band = (freq >= freq_min_for_psd) & (freq <= freq_max_for_psd)
                            psds_original.append((freq, psd))
                            psd_avg_original = np.mean(psd[band])

                            freq, psd = tod_psd(det_tod_processed, acq_freq_downsampled)
                            psd /= ( len(det_tod) / len(det_tod_processed) )
                            band = (freq >= freq_min_for_psd) & (freq <= freq_max_for_psd)
                            psds_downsampled.append((freq, psd /( len(det_tod) / len(det_tod_processed) )))
                            psd_avg_downsampled = np.mean(psd[band])

                            psds_relative_error.append((psd_avg_original - psd_avg_downsampled)/psd_avg_original)

                    dict_tods_fidelity[T_key][df_key][prec]['psds_original'] = psds_original
                    dict_tods_fidelity[T_key][df_key][prec]['psds_downsampled']  = psds_downsampled
                    dict_tods_fidelity[T_key][df_key][prec]['psds_relative_error']  = psds_relative_error

                    pickle.dump(dict_tods_fidelity, open(dict_tods_fidelity_file, 'wb'))

                    if(False): 

                        plt.figure()
                        plt.xscale('log'); plt.yscale('log'); plt.xlabel('f [Hz]'); plt.ylabel('PSD [$\\rm Jy^2/beam.s^2$]')
                        for i, (chunk, chunk_processed) in enumerate(zip(chunks_original,chunks_downsampled)):

                            for det_id, (det_tod, det_tod_processed) in enumerate(zip(chunk,chunk_processed)):
                                if(len(det_tod) <= 1): continue

                                freq, psd = tod_psd(det_tod, acq_freq_original)
                                band = (freq >= freq_min_for_psd) & (freq <= freq_max_for_psd)
                                psd_avg_original = np.mean(psd[band])
                                plt.plot(3.5, psd_avg_original, 'ok')

                                if(False):
                                    aaf = det.AntiAliasingFilter( fs_in=P['acquisition_frequency'], fs_out=100.0, fc=45.0)
                                    tod_resampled = aaf.process(det_tod)
                                    freq, psd = tod_psd(tod_resampled, P['acquisition_frequency'])
                                    plt.step(freq_fft[inds], tod_psd[inds] /( len(det_tod) / len(tod_resampled) ), where='mid', alpha=0.1, c='r')

                                freq, psd = tod_psd(det_tod_processed, acq_freq_downsampled)

                                band = (freq >= freq_min_for_psd) & (freq <= freq_max_for_psd)
                                psd_avg_original = np.mean(psd[band])
                                plt.plot(3.5, psd_avg_original, 'or')
                                plt.step(freq, psd/( len(det_tod) / len(det_tod_processed) ), where='mid', alpha=.1, c='r')


                                freq, psd = tod_psd(det_tod, acq_freq_original)
                                band = (freq >= freq_min_for_psd) & (freq <= freq_max_for_psd)
                                psds_original.append((freq, psd))
                                psd_avg_original = np.mean(psd[band])


                                freq, psd = tod_psd(det_tod_processed, acq_freq_downsampled)
                                band = (freq >= freq_min_for_psd) & (freq <= freq_max_for_psd)
                                psds_downsampled.append((freq, psd /( len(det_tod) / len(det_tod_processed) )))
                                psd_avg_downsampled = np.mean(psd[band])

                                print((psd_avg_original - psd_avg_downsampled)/psd_avg_original)

                                psds_relative_error.append((psd_avg_original - psd_avg_downsampled)/psd_avg_original)
                                            
                        plt.axvline(acq_freq_original,   ymin=1e-20, ymax=1e10, color='grey', ls=':' , label=f'{acq_freq_original:.1f}Hz sampling frequency')
                        plt.axvline(acq_freq_original/2, ymin=1e-20, ymax=1e10, color='grey', label=f'{acq_freq_original:.1f}Hz Nyquist frequency')
                        plt.axvline(P['acquisition_frequency']/2, ymin=1e-20, ymax=1e10, color='grey', label=f'{P["acquisition_frequency"]:.1f}Hz Nyquist frequency')
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.xlabel('f [Hz]')
                        plt.ylabel('PSD [$\\rm Jy^2/beam.s^2$]')
                        plt.legend()
                        plt.show()
        
    #-----------------------------------------------------------------------------------
    BS = 12; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    plt.figure()

    for downsample_frequency, c in zip(downsampled_freq_list, cm.rainbow(np.linspace(0.,1,len(downsampled_freq_list)))):
        y_points = []
        y_err = []
        df_key = f'downsample_frequency = {downsample_frequency:.1f} Hz'    
        for t_int in t_int_list:
            T_key = f'T = {t_int:.1f} min'
            y_points.append(np.mean(dict_tods_fidelity[T_key][df_key]['float64']['psds_relative_error']))
            y_err.append(np.std(dict_tods_fidelity[T_key][df_key]['float64']['psds_relative_error']))
        plt.errorbar(t_int_list, y_points, yerr=y_err, fmt='o', color = c, ecolor=c, label=f'{downsample_frequency:.1f}Hz')

    plt.xlabel('Integration time [min]')
    plt.ylabel(f"Relat.diff in PSD@f=({dict_tods_fidelity['fft_freq_bounds'][0]:.1f}-{dict_tods_fidelity['fft_freq_bounds'][1]:.1f}Hz)\n between native {P['acquisition_frequency']:.1f}Hz\nand downsampled timestreams")
    plt.tight_layout()
    plt.legend()

    plt.figure()
    for prec,c in zip(precision_list, cm.viridis(np.linspace(0.,1,len(precision_list)))):

        y_points = []
        y_err = []
        df_key = f'downsample_frequency = {100:.1f} Hz'

        for t_int in t_int_list:
            T_key = f'T = {t_int:.1f} min'
            y_points.append(np.mean(dict_tods_fidelity[T_key][df_key][prec]['psds_relative_error']))
            y_err.append(np.std(dict_tods_fidelity[T_key][df_key][prec]['psds_relative_error']))
        
        plt.errorbar(t_int_list, y_points, yerr=y_err, fmt='o', color = c, ecolor=c, label=prec)
    
    plt.xlabel('Integration time [min]')
    plt.ylabel(f"Relat.diff in PSD@f=({dict_tods_fidelity['fft_freq_bounds'][0]:.1f}-{dict_tods_fidelity['fft_freq_bounds'][1]:.1f}Hz)\n between native {P['acquisition_frequency']:.1f}Hz\nand 100Hz downsampled timestreams")
    plt.tight_layout()
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------

    return 0

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

def load_kids(P_namap):
    btable = tb.Table.read(P_namap['detector_table'], format='ascii.tab')
    if P_namap['frequencies'] is not None: filtered = btable[np.isin(btable['Frequency'], P_namap['frequencies'])]
    if P_namap['detectors_to_use'] is not None: 
        good_kid_table = tb.Table.read(P_namap['detectors_to_use'], format='ascii.tab')
        filtered = btable[np.isin(btable['Name'], good_kid_table['Name'])]
    if P_namap['frequencies'] is None and P_namap['detectors_to_use'] is None: filtered = btable
    return filtered['Name']

def make_chunks(timestamps, det_data, acq_freq, flags=None): 
    if(flags is not None):
        timestamps = timestamps[flags==1]
        det_data = np.asarray(det_data)[:, flags==1]
    dt = np.diff(timestamps)
    breaks = np.where(np.abs(np.diff(dt)) > 1)[0] + 1 #np.where(dt != dt[0])[0] + 1
    timestamp_chunks = np.split(timestamps, breaks)
    det_chunks = np.split(det_data, breaks, axis=1)
    return acq_freq, det_chunks

def tod_psd(tod, acq_freq):
    freq_fft = np.fft.fftfreq(len(tod), 1/acq_freq) #TOD frequencies.
    inds = np.where(freq_fft>0) 
    tod_psd = ( np.fft.fft(tod) * (1/acq_freq) * np.conj( np.fft.fft(tod) * (1/acq_freq) ) / len(tod)  ).real
    return freq_fft[inds], tod_psd[inds]

DT = dtype_map['float64']
IT = int_map['float64']

#------------------------------------------------------
t_int_list = (5,)#1,2,3,10,15,20,25)# 4, 5, 6, 7, 8, 9, 15, 25) #min
downsampling_frequency = (100,)#50,150)
precision_list = ('float64', )#'float32','float16')
resolution_list = (30,40,50,60) #arcsecs
#pix_num
nb_pixels = (1,2,3,5,6,7,8,9,10,20,30,40,50)
nb_bands = (1,2,3,4,11,21,41,64)
downsampled_freq_list = (50, 100,150)

dict_tods_fidelity_file = 'dict_tods_fidelity.p'
dict_coadded_map_fidelity_file = 'dict_coadded_map_fidelity.p'
freq_min_for_psd, freq_max_for_psd = 1,6
#------------------------------------------------------

#------------------
LW_min= 317e-6  # Hz
D = 2.0             # m
FWHM = 1.22 * LW_min / D * 180 / np.pi  # degrees
res = FWHM / 2  
#------------------

#----------------------------------------------------------------------------------------
P = load_params(f'{DESKTOP}/'+'timestream_maker/PAR_files/params_strategy_profiling.par')
#-----------------------------
P['nb_channels_per_array'] = 1 #!!
#P['acquisition_frequency'] = 110
nbdets = None
#-----------------------------
P_namap = load_params(f'{DESKTOP}/'+'namap/PAR_FILES/params_namap_profiling.par')
P_namap['detector_table'] = P['detectors_name_file']
P_namap['cdelt'] = res, res
#crval output_map save_downsampled_TODS output_hdf5
#----------------------------------------------------------------------------------------

if(not os.path.isfile( P['detectors_name_file']) ): gen_detectors_main(P)

#I) Fidelity tests

for t_int in t_int_list:

    print(f'Generating {t_int}min timestreams.')

    P['T_duration'] = t_int / 60
    P['output_name'] = f'TOD_{t_int:.1f}min.hdf5' 
    P['alt_size'] = 2.5*FWHM
    P['alt_step'] = FWHM*1/3

    if(not os.path.isfile(P['output_path']+P['output_name'] )):
        main_1det(P)
        main_tod(P)

if(False): test_namap_tods_fidelity(dict_tods_fidelity_file)
test_namap_coadded_map_fidelity(dict_coadded_map_fidelity_file)

