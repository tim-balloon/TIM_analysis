import time
import tracemalloc
import cProfile
import pstats
import matplotlib.pyplot as plt
import sysconfig, sys, site, os, io, copy, glob, pickle, psutil, platform
from astropy.io import fits
from unittest import mock
from collections import namedtuple
from pathlib import Path
import scipy.constants as cst
import numpy as np
import astropy.table as tb
from matplotlib.pyplot import cm
import pygetdata as gd
import shutil

#--------------------------------------------------------------
# --- Intel Celeron 4305UE Specs ---
SIM_CPU = "Intel(R) Celeron(R) 4305UE @ 2.00GHz"
SIM_CORES = 2
SIM_THREADS = 2
SIM_MEMORY_GB = 64 # realistic config (max supported = 64 GB)
SIM_MEMORY_BYTES = SIM_MEMORY_GB * 1024**3
#--------------------------------------------------------------

DESKTOP = Path.home() / "/home/mvancuyck/Desktop/TIM_analysis/"

sys.path.insert(0, str(DESKTOP / "namap"))
sys.path.insert(0, str(DESKTOP / "timestream_maker"))

from timestream_maker.gen_timestreams import *
from timestream_maker.hitmap_1detector import *
from timestream_maker.gen_detectors_arrays import *

from namap.namap_main import main as namap_main
import namap.src.loaddata as ld
import namap.src.detector as det
import namap.src.psd_analysis as aps

def profiling_coadded_maps(dict_file_path, profiling_vs_tod_time =True, profiling_vs_nb_bands=True, load_directly = False):

    if(load_directly): results = pickle.load( open(dict_file_path, 'rb'))
    else: 
        if(os.path.isfile(dict_file_path)): results = pickle.load( open(dict_file_path, 'rb'))
        else: results = {}

    if(profiling_vs_tod_time):

            key = 'profiling vs tod time'
            results.setdefault(key, {})

            results[key]["t_int"] = t_int_list
            map_compression_list = ('coadd.fits','coadd.fits.gz')

            for precision in precision_list: 

                results[key].setdefault(precision, {})

                for map_compression in map_compression_list:
                    
                    results[key][precision].setdefault(map_compression, {}) 
                    results[key][precision][map_compression]['peak memory [MB]'] = []
                    results[key][precision][map_compression]['time [s]'] = []
                    results[key][precision][map_compression]['output size [MB]'] = []

                    for t in results[key]["t_int"]: #Nbdets 51

                        P_namap['cdelt'] = 40/3600, 40/3600
                        P_namap['frequencies'] = (715.0,) #GHz        
                        P_namap['precision'] = precision                        
                        P_namap['num_frames']  = int(t*60) #integration time in seconds to be loaded. 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['output_map'] = P['output_path']+map_compression
                        P_namap['coadd'] = True
                        P_namap['save_downsampled_TODS'] = False
                        P_namap['remove_turnarounds'] = True
                        P_namap['downsample_frequency'] = 100

                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        namap_main(P_namap)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start
                        #------------------------------------------------------
                       
                        if os.path.exists(P_namap['output_map']): file_size_mb = os.path.getsize(P_namap['output_map'] ) / 1e6
                        else: file_size_mb = float('nan')  

                        print(f"time {t}min, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={file_size_mb:.2f}MB")

                        # Store results
                        results[key][precision][map_compression]['peak memory [MB]'].append(peak / 1e6)
                        results[key][precision][map_compression]['time [s]'].append(timing)
                        results[key][precision][map_compression]['output size [MB]'].append(file_size_mb)

    if(profiling_vs_nb_bands):

        key = 'profiling vs nb bands'
        results.setdefault(key, {})

        for precision in precision_list: 

            results[key].setdefault(precision, {})

            for map_compression in ('coadd.fits','coadd.fits.gz'):

                results[key][precision].setdefault(map_compression, {})
                results[key][precision][map_compression]['peak memory [MB]'] = []
                results[key][precision][map_compression]['time [s]'] = []
                results[key][precision][map_compression]['output size [MB]'] = []

                results[key]["nb dets"] = []

                for nband in nb_bands:
                    for npix in nb_pixels:
                        val = int(nband * npix)
                        if val in results[key]["nb dets"]: continue
                                    
                        # Skip if val is smaller than max so far
                        if results[key]["nb dets"] and val < max(results[key]["nb dets"]): continue


                        if val > 1000 and len(results[key]["nb dets"]) > 0:
                            if val < 1.3 * max(results[key]["nb dets"]): continue
                            
                        results[key]["nb dets"].append(val)

                        freq_list = 715.0 + 4.0 * np.arange(nband)
                        P_namap['cdelt'] = 40/3600, 40/3600
                        P_namap['frequencies'] = freq_list     
                        P_namap['precision'] = precision                        
                        P_namap['num_frames']  = int(5*60) #integration time in seconds to be loaded. 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['output_map'] = P['output_path']+map_compression
                        P_namap['coadd'] = True
                        P_namap['save_downsampled_TODS'] = False
                        P_namap['remove_turnarounds'] = True
                        P_namap['downsample_frequency'] = 100
                        print('')
                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        namap_main(P_namap, npix)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start
                        #------------------------------------------------------

                        # Measure output file size (adapt this path!)
                        if os.path.exists(P_namap['output_map'] ): file_size_mb = os.path.getsize(P_namap['output_map'] ) / 1e6
                        else: file_size_mb = float('nan')  # file not found → record NaN or 0

                        print(f"Nb bands {nband*npix}, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={file_size_mb:.2f}MB")
                        print('')

                        # Store results
                        results[key][precision][map_compression]['peak memory [MB]'].append(peak / 1e6)
                        results[key][precision][map_compression]['time [s]'].append(timing)
                        results[key][precision][map_compression]['output size [MB]'].append(file_size_mb)

    with open(dict_file_path, 'wb') as f: pickle.dump(results, f)
    
    return 0

def profiling_individual_maps(dict_file_path, profiling_vs_tod_time=True, profiling_vs_nb_bands=True, load_directly=False):

    if(load_directly): results = pickle.load( open(dict_file_path, 'rb'))
    else: 
        if(os.path.isfile(dict_file_path)): results = pickle.load( open(dict_file_path, 'rb'))
        else: results = {}

    if(profiling_vs_tod_time):

        key = 'profiling vs tod time'
        results.setdefault(key, {})

        results[key]["t_int"] = t_int_list

        for precision in precision_list: 

            results[key].setdefault(precision, {})


            for map_compression in ('individual.fits.gz', ): #'individual.fits'
                
                results[key][precision].setdefault(map_compression, {})
                results[key][precision][map_compression]['peak memory [MB]'] = []
                results[key][precision][map_compression]['time [s]'] = []
                results[key][precision][map_compression]['output size [MB]'] = []

                for t in results[key]["t_int"]:
                    
                    P_namap['cdelt'] = 40/3600, 40/3600
                    P_namap['frequencies'] = (715.0,) #GHz       
                    P_namap['precision'] = precision                        
                    P_namap['num_frames']  = t * 60 #seconds 
                    P_namap['first_frame'] = 0 #Starting time in second to loaded
                    P_namap['output_map'] = P['output_path']+map_compression
                    P_namap['coadd'] = False
                    P_namap['save_downsampled_TODS'] = False

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    namap_main(P_namap)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Store results
                    results[key][precision][map_compression]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision][map_compression]['time [s]'].append(timing)

                    # Path to your files (adjust if needed)
                    folder = P['output_path']  # current directory
                    filename = map_compression
                    name_before_fits = filename.rsplit('.fits', 1)[0]
                    fits_and_after = filename[filename.find('.fits'):]  
                    pattern = f'{name_before_fits}_*{fits_and_after}'

                    # Get all matching files
                    files = glob.glob(os.path.join(folder, pattern))
                    # Sum their sizes in bytes
                    total_size_bytes = sum(os.path.getsize(f) for f in files)
                    # Optionally, convert to MB
                    total_size_mb = total_size_bytes / (1024**2)
                    results[key][precision][map_compression]['output size [MB]'].append(total_size_mb)
                    print(f"time {t}min, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={total_size_mb:.2f}MB")

                    # Delete them
                    for f in files:
                        try:
                            os.remove(f)
                        except OSError as e:
                            print(f"Error deleting {f}: {e}")

    if(profiling_vs_nb_bands):

        key = 'profiling vs nb bands'
        results.setdefault(key, {})

        for precision in precision_list: 
                results[key].setdefault(precision, {})

                for map_compression in ('individual.fits.gz',): #'individual.fits',

                    results[key][precision].setdefault(map_compression, {})
                    results[key][precision][map_compression]['peak memory [MB]'] = []
                    results[key][precision][map_compression]['time [s]'] = []
                    results[key][precision][map_compression]['output size [MB]'] = []

                    results[key]["nb dets"] = []

                    for nband in nb_bands:
                        for npix in nb_pixels:
                            val = int(nband * npix)
                            if val in results[key]["nb dets"]: continue         
                            # Skip if val is smaller than max so far
                            if results[key]["nb dets"] and val < max(results[key]["nb dets"]): continue

                            if val > 1000 and len(results[key]["nb dets"]) > 0:
                                if val < 1.3 * max(results[key]["nb dets"]): continue

                            results[key]["nb dets"].append(val)
                            freq_list = 715.0 + 4.0 * np.arange(nband)

                            P_namap['cdelt'] = 40/3600, 40/3600
                            P_namap['frequencies'] = freq_list    
                            P_namap['precision'] = precision                        
                            P_namap['num_frames']  = 5 * 60 #seconds 
                            P_namap['first_frame'] = 0 #Starting time in second to loaded
                            P_namap['output_map'] = P['output_path']+map_compression
                            P_namap['coadd'] = False
                            P_namap['save_downsampled_TODS'] = False
                            

                            #------------------------------------------------------
                            tracemalloc.start()
                            start = time.time()
                            namap_main(P_namap, npix)
                            current, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            end = time.time()
                            timing = end - start
                            #------------------------------------------------------

                            # Store results
                            results[key][precision][map_compression]['peak memory [MB]'].append(peak / 1e6)
                            results[key][precision][map_compression]['time [s]'].append(timing)

                            folder = P['output_path']  # current directory
                            filename = map_compression
                            name_before_fits = filename.rsplit('.fits', 1)[0]
                            fits_and_after = filename[filename.find('.fits'):]  
                            pattern = f'{name_before_fits}_*{fits_and_after}'

                            # Get all matching files
                            files = glob.glob(os.path.join(folder, pattern))
                            # Sum their sizes in bytes
                            total_size_bytes = sum(os.path.getsize(f) for f in files)
                            # Optionally, convert to MB
                            total_size_mb = total_size_bytes / (1024**2)
                            results[key][precision][map_compression]['output size [MB]'].append(total_size_mb)
                            print(f"Nb bands {val}, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={total_size_mb:.2f}MB")

                            # Delete them
                            for f in files:
                                try:
                                    os.remove(f)
                                except OSError as e:
                                    print(f"Error deleting {f}: {e}")

    with open(dict_file_path, 'wb') as f: pickle.dump(results, f)

    return 0

def profiling_tods(dict_file_path, profiling_vs_tod_time = True, profiling_vs_nb_bands=True, load_directly = False):

    
    def get_dir_size(path):
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    if(load_directly): results = pickle.load( open(dict_file_path, 'rb'))
    else: 
        if(os.path.isfile(dict_file_path)): results = pickle.load( open(dict_file_path, 'rb'))
        else: results = {}

    if(profiling_vs_tod_time):

        key = 'profiling vs tod time'
        results.setdefault(key, {})

        results[key]["t_int"] = t_int_list

        for precision in precision_list: 

            results[key].setdefault(precision, {})

            for compression in ('','.hdf5','.zip'):
            
                results[key][precision].setdefault(compression, {})
                results[key][precision][compression]['peak memory [MB]'] = []
                results[key][precision][compression]['time [s]'] = []
                results[key][precision][compression]['output size [MB]'] = []

                for t in results[key]["t_int"]:

                    P_namap['cdelt'] = 40/3600, 40/3600
                    P_namap['frequencies'] = (715.0,) #GHz       
                    P_namap['precision'] = precision                        
                    P_namap['num_frames']  = t * 60 #seconds 
                    P_namap['first_frame'] = 0 #Starting time in second to loaded
                    P_namap['save_downsampled_TODS'] = True
                    P_namap['output_tods'] = P['output_path']+f'tods_{precision}'+compression
                    P_namap['remove_turnarounds'] = False
                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    namap_main(P_namap)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Store results
                    output_file = P_namap['output_tods'] 
                    #if('zip' in compression): output_file += '.zip'
                    if os.path.exists(output_file):
                        if os.path.isfile(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                        else: file_size_mb = get_dir_size(output_file) / 1e6
                    else: file_size_mb = float('nan') 
                    print(f" time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={file_size_mb:.2f}MB")

                    results[key][precision][compression]['output size [MB]'].append(file_size_mb)
                    results[key][precision][compression]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision][compression]['time [s]'].append(timing)

                    try:
                        if os.path.isfile(output_file): os.remove(output_file)
                        else: shutil.rmtree(output_file)
                    except OSError as e:
                        print(f"Error deleting {output_file}: {e}")

    

    if(profiling_vs_nb_bands):

        key = 'profiling vs nb bands'
        results.setdefault(key, {})

        for precision in precision_list: 

            results[key].setdefault(precision, {})

            for compression in ('','.hdf5','.zip'):
            
                results[key][precision].setdefault(compression, {})
                results[key][precision][compression]['peak memory [MB]'] = []
                results[key][precision][compression]['time [s]'] = []
                results[key][precision][compression]['output size [MB]'] = []
                
                results[key]["nb dets"] = []

                for nband in nb_bands:
                    for npix in nb_pixels:
                        val = int(nband * npix)
                        if val in results[key]["nb dets"]: continue
                                    
                        # Skip if val is smaller than max so far
                        if results[key]["nb dets"] and val < max(results[key]["nb dets"]): continue

                        if val > 1000 and len(results[key]["nb dets"]) > 0:
                            if val < 1.3 * max(results[key]["nb dets"]): continue

                        results[key]["nb dets"].append(val)
                        
                        freq_list = 715.0 + 4.0 * np.arange(nband)

                        P_namap['cdelt'] = 40/3600, 40/3600
                        P_namap['frequencies'] = freq_list    
                        P_namap['precision'] = precision                        
                        P_namap['num_frames']  = 5 * 60 #seconds 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['save_downsampled_TODS'] = True
                        P_namap['output_tods'] = P['output_path']+f'tods_{precision}'+compression
                        P_namap['remove_turnarounds'] = False

                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        namap_main(P_namap, npix)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start
                        #------------------------------------------------------

                        # Measure output file size (adapt this path!)
                        output_file = P_namap['output_tods']
                        #if('zip' in compression): output_file += '.zip'

                        if os.path.exists(output_file):
                            if os.path.isfile(output_file):
                                file_size_mb = os.path.getsize(output_file) / 1e6
                            else:
                                file_size_mb = get_dir_size(output_file) / 1e6
                        else: file_size_mb = float('nan') 
                        
                        print(f"Nb bands {val}, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={file_size_mb:.2f}MB")

                        # Store results
                        results[key][precision][compression]['peak memory [MB]'].append(peak / 1e6)
                        results[key][precision][compression]['time [s]'].append(timing)
                        results[key][precision][compression]['output size [MB]'].append(file_size_mb)

                        try:
                            if os.path.isfile(output_file): os.remove(output_file)
                            else: shutil.rmtree(output_file)
                        except OSError as e:
                            print(f"Error deleting {output_file}: {e}")

        with open(dict_file_path, 'wb') as f: pickle.dump(results, f)
        return 0

def profiling_raw_tods(dict_file_path, profiling_vs_tod_time = True, profiling_vs_nb_bands=True, load_directly = False):

    if(load_directly): results = pickle.load( open(dict_file_path, 'rb'))
    else: 
        if(os.path.isfile(dict_file_path)): results = pickle.load( open(dict_file_path, 'rb'))
        else: results = {}

    if(profiling_vs_tod_time):

        key = 'profiling vs tod time'
        results.setdefault(key, {})

        results[key]["t_int"] = t_int_list

        for precision in precision_list: 

            results[key].setdefault(precision, {})

            for compression, frequency in zip(('100Hz','no_compression'), (100,None)):
            
                results[key][precision].setdefault(compression, {})
                results[key][precision][compression]['peak memory [MB]'] = []
                results[key][precision][compression]['time [s]'] = []
                results[key][precision][compression]['output size [MB]'] = []

                for t in results[key]["t_int"]:

                    P_namap['cdelt'] = 40/3600, 40/3600
                    P_namap['frequencies'] = (715.0,) #GHz       
                    P_namap['precision'] = precision                        
                    P_namap['num_frames']  = t * 60 #seconds 
                    P_namap['first_frame'] = 0 #Starting time in second to loaded
                    P_namap['output_tods'] = P['output_path']+f'tods_{precision}_{compression}'
                    P_namap['remove_turnarounds'] = False
                    P_namap['downsample_frequency'] = frequency
                    P_namap['save_raw_TODS'] = True

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    namap_main(P_namap)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    end = time.time()
                    timing = end - start
                    #------------------------------------------------------

                    # Store results
                    output_file = P_namap['output_tods'] + '.zip'
                    
                    if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                    else: file_size_mb = float('nan')  # file not found → record NaN or 0

                    results[key][precision][compression]['output size [MB]'].append(file_size_mb)
                    results[key][precision][compression]['peak memory [MB]'].append(peak / 1e6)
                    results[key][precision][compression]['time [s]'].append(timing)

                    try:
                        os.remove(output_file)
                    except OSError as e:
                        print(f"Error deleting {output_file}: {e}")

    if(profiling_vs_nb_bands):

        key = 'profiling vs nb bands'
        results.setdefault(key, {})

        for precision in precision_list: 

            results[key].setdefault(precision, {})

            for compression, frequency in zip(('100Hz','no_compression'), (100,None)):
            
                results[key][precision].setdefault(compression, {})
                results[key][precision][compression]['peak memory [MB]'] = []
                results[key][precision][compression]['time [s]'] = []
                results[key][precision][compression]['output size [MB]'] = []
                
                results[key]["nb dets"] = []

                for nband in nb_bands:
                    for npix in nb_pixels:
                        val = int(nband * npix)
                        if val in results[key]["nb dets"]: continue
                                    
                        # Skip if val is smaller than max so far
                        if results[key]["nb dets"] and val < max(results[key]["nb dets"]): continue
                        results[key]["nb dets"].append(val)
                        
                        freq_list = 715.0 + 4.0 * np.arange(nband)

                        P_namap['cdelt'] = 40/3600, 40/3600
                        P_namap['frequencies'] = freq_list    
                        P_namap['precision'] = precision                        
                        P_namap['num_frames']  = 5 * 60 #seconds 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['save_downsampled_TODS'] = True
                        P_namap['output_tods'] = P['output_path']+f'tods_{precision}'
                        P_namap['remove_turnarounds'] = False
                        P_namap['downsample_frequency'] = frequency
                        P_namap['save_raw_TODS'] = True

                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        namap_main(P_namap, npix)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start
                        #------------------------------------------------------

                        # Measure output file size (adapt this path!)
                        output_file = P_namap['output_tods'] + '.zip'
                        if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
                        else: file_size_mb = float('nan')  # file not found → record NaN or 0

                        print(f"Nb bands {val}, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={file_size_mb:.2f}MB")

                        # Store results
                        results[key][precision][compression]['peak memory [MB]'].append(peak / 1e6)
                        results[key][precision][compression]['time [s]'].append(timing)
                        results[key][precision][compression]['output size [MB]'].append(file_size_mb)

                        try:
                            os.remove(output_file)

                        except OSError as e:
                            print(f"Error deleting {output_file}: {e}")

        with open(dict_file_path, 'wb') as f: pickle.dump(results, f)
        return 0

def profiling_fcts(dict_file_path, load_directly = False):

    if(load_directly): results = pickle.load( open(dict_file_path, 'rb'))
    else: 
        if(os.path.isfile(dict_file_path)): results = pickle.load( open(dict_file_path, 'rb'))
        else: results = {}

    key = 'profiling fcts'
    results.setdefault(key, {})

    results[key]["t_int"] = 5
    results[key]['#Nbdets'] = 51
    map_compression = 'coadd.fits'
    nrep = 20

    for precision in precision_list: 

        results[key].setdefault(precision, {})

        P_namap['cdelt'] = 40/3600, 40/3600
        P_namap['frequencies'] = (715.0,) #GHz        
        P_namap['precision'] = precision                        
        P_namap['num_frames']  = int(results[key]["t_int"]*60) #integration time in seconds to be loaded. 
        P_namap['first_frame'] = 0 #Starting time in second to loaded
        P_namap['output_map'] = P['output_path']+map_compression
        P_namap['coadd'] = True
        P_namap['save_downsampled_TODS'] = False
        P_namap['remove_turnarounds'] = True
        P_namap['downsample_frequency'] = 100

        results[key][precision]['peak memory [MB] loaddata'] = np.zeros(nrep)
        results[key][precision]['time [s] loaddata'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] clean'] = np.zeros(nrep)
        results[key][precision]['time [s] clean'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] sync'] = np.zeros(nrep)
        results[key][precision]['time [s] sync'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] turns'] = np.zeros(nrep)
        results[key][precision]['time [s] turns'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] corr'] = np.zeros(nrep)
        results[key][precision]['time [s] corr'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] maps'] = np.zeros(nrep)
        results[key][precision]['time [s] maps'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] savemap'] = np.zeros(nrep)
        results[key][precision]['time [s] savemap'] = np.zeros(nrep)
        results[key][precision]['peak memory [MB] savetods'] = np.zeros(nrep)
        results[key][precision]['time [s] savetods'] = np.zeros(nrep)

        for i in range(nrep):
            results[key][precision][map_compression]['peak memory [MB] loaddata'][i],results[key][precision][map_compression]['time [s] loaddata'][i],results[key][precision][map_compression]['peak memory [MB] clean'][i], results[key][precision][map_compression]['time [s] clean'][i], results[key][precision][map_compression]['peak memory [MB] sync'][i], results[key][precision][map_compression]['time [s] sync'][i], results[key][precision][map_compression]['peak memory [MB] turns'][i], results[key][precision][map_compression]['time [s] turns'][i], results[key][precision][map_compression]['peak memory [MB] corr'][i], results[key][precision][map_compression]['time [s] corr'][i], results[key][precision][map_compression]['peak memory [MB] maps'][i], results[key][precision][map_compression]['time [s] maps'][i], results[key][precision][map_compression]['peak memory [MB] savemap'][i], results[key][precision][map_compression]['time [s] savemap'][i] = namap_main(P_namap)

        P_namap['save_downsampled_TODS'] = True
        P_namap['output_tods'] = 'tods'

        for i in range(nrep):
            _,_,_, _, _, _, _, _, results[key][precision][map_compression]['peak memory [MB] savetods'][i], results[key][precision][map_compression]['time [s] savetods'][i] = namap_main(P_namap)

        results[key][precision]['peak memory [MB] loaddata mean'] = np.mean(results[key][precision]['peak memory [MB] loaddata'])
        results[key][precision]['peak memory [MB] loaddata std'] = np.std(results[key][precision]['peak memory [MB] loaddata'])
        results[key][precision]['time [s] loaddata mean'] = np.mean(results[key][precision]['time [s] loaddata'])
        results[key][precision]['time [s] loaddata std'] = np.std(results[key][precision]['time [s] loaddata'])
        results[key][precision]['peak memory [MB] clean mean'] = np.mean(results[key][precision]['peak memory [MB] clean'])
        results[key][precision]['peak memory [MB] clean std'] = np.std(results[key][precision]['peak memory [MB] clean'])
        results[key][precision]['time [s] clean mean'] = np.mean(results[key][precision]['time [s] clean'])
        results[key][precision]['time [s] clean std'] = np.std(results[key][precision]['time [s] clean'])
        results[key][precision]['peak memory [MB] sync mean'] = np.mean(results[key][precision]['peak memory [MB] sync'])
        results[key][precision]['peak memory [MB] sync std'] = np.std(results[key][precision]['peak memory [MB] sync'])
        results[key][precision]['time [s] sync mean'] = np.mean(results[key][precision]['time [s] sync'])
        results[key][precision]['time [s] sync std'] = np.std(results[key][precision]['time [s] sync'])
        results[key][precision]['peak memory [MB] turns mean'] = np.mean(results[key][precision]['peak memory [MB] turns'])
        results[key][precision]['peak memory [MB] turns std'] = np.std(results[key][precision]['peak memory [MB] turns'])
        results[key][precision]['time [s] turns mean'] = np.mean(results[key][precision]['time [s] turns'])
        results[key][precision]['time [s] turns std'] = np.std(results[key][precision]['time [s] turns'])
        results[key][precision]['peak memory [MB] corr mean'] = np.mean(results[key][precision]['peak memory [MB] corr'])
        results[key][precision]['peak memory [MB] corr std'] = np.std(results[key][precision]['peak memory [MB] corr'])
        results[key][precision]['time [s] corr mean'] = np.mean(results[key][precision]['time [s] corr'])
        results[key][precision]['time [s] corr std'] = np.std(results[key][precision]['time [s] corr'])
        results[key][precision]['peak memory [MB] maps mean'] = np.mean(results[key][precision]['peak memory [MB] maps'])
        results[key][precision]['peak memory [MB] maps std'] = np.std(results[key][precision]['peak memory [MB] maps'])
        results[key][precision]['time [s] maps mean'] = np.mean(results[key][precision]['time [s] maps'])
        results[key][precision]['time [s] maps std'] = np.std(results[key][precision]['time [s] maps'])
        results[key][precision]['peak memory [MB] savemap mean'] = np.mean(results[key][precision]['peak memory [MB] savemap'])
        results[key][precision]['peak memory [MB] savemap std'] = np.std(results[key][precision]['peak memory [MB] savemap'])
        results[key][precision]['time [s] savemap mean'] = np.mean(results[key][precision]['time [s] savemap'])
        results[key][precision]['time [s] savemap std'] = np.std(results[key][precision]['time [s] savemap'])
        results[key][precision]['peak memory [MB] savetods mean'] = np.mean(results[key][precision]['peak memory [MB] savetods'])
        results[key][precision]['peak memory [MB] savetods std'] = np.std(results[key][precision]['peak memory [MB] savetods'])
        results[key][precision]['time [s] savetods mean'] = np.mean(results[key][precision]['time [s] savetods'])
        results[key][precision]['time [s] savetods std'] = np.std(results[key][precision]['time [s] savetods'])

    with open(dict_file_path, 'wb') as f: pickle.dump(results, f)
    
    return 0

def test_namap_coadded_map_fidelity(dict_coadded_map_fidelity_file, load_directly = False):

    if(load_directly): dict_maps_fidelity = pickle.load( open(dict_coadded_map_fidelity_file, 'rb'))
    else: 
        if(os.path.isfile(dict_coadded_map_fidelity_file)): dict_maps_fidelity = pickle.load( open(dict_coadded_map_fidelity_file, 'rb'))
        else: dict_maps_fidelity = {}

        dict_maps_fidelity['k_bounds'] = (k_min_for_aps, freq_max_for_aps )

        for t_int in t_int_list:

            T_key = f'T = {t_int:.1f} min'
            dict_maps_fidelity.setdefault(T_key, {})  

            for downsample_frequency in downsampled_freq_list:

                df_key = f'downsample_frequency = {downsample_frequency:.1f} Hz'
                dict_maps_fidelity[T_key].setdefault(df_key, {})  

                for prec in precision_list:

                    dict_maps_fidelity[T_key][df_key].setdefault(prec, {})  

                    for res_value in resolution_list:

                        res_key = f'res={res_value*3600:.2f} arcsecs'
                        dict_maps_fidelity[T_key][df_key][prec].setdefault(res_key, {})  
                        #-------------------------------------------
                        P_namap['hdf5_file'] = P['output_path']+f'TOD_{t_int:.1f}min.hdf5' 
                        P_namap['remove_turnarounds'] = True
                        P_namap['save_downsampled_TODS'] = False
                        P_namap['downsample_frequency'] = downsample_frequency
                        #P_namap['output_hdf5'] = P['output_path']+f'namap_downsampled_TOD_{t_int:.1f}min_{downsample_frequency:.1f}Hz_{prec}.hdf5' 
                        P_namap['num_frames']  = int(t_int*60+1) #integration time in seconds to be loaded. 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['precision'] = prec
                        P_namap['output_map'] = P['output_path']+f'namap_downsampled_TOD_{t_int:.1f}min_{100:.1f}Hz_{prec}_coadd_map_res{res_value*3600:.2f}arcsecs.fits' 
                        P_namap['cdelt'] = res, res

                        if(not os.path.isfile(P_namap['output_map'])): namap_main(P_namap)

                        created_map = fits.getdata(P_namap['output_map'])
                        hdr = fits.getheader(P_namap['output_map'])
                        pk = aps.angular_power_spectrum((created_map,),hdr['CDELT1']*60, delta_k_over_k=delta_k_over_k)
                        pk_mes, k = pk.p2()

                        dict_maps_fidelity[T_key][df_key][prec][res_key]['coadd map'] = created_map
                        dict_maps_fidelity[T_key][df_key][prec][res_key]['hdr'] = hdr
                        dict_maps_fidelity[T_key][df_key][prec][res_key]['k'] = k
                        dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes'] = pk_mes[0]
                        band = (k >= k_min_for_aps) & (k <= freq_max_for_aps)
                        dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes_avg'] = np.mean(pk_mes[0][band])

                        pickle.dump(dict_maps_fidelity, open(dict_coadded_map_fidelity_file, 'wb'))

                        #print(T_key,df_key,prec,res_key)

                        if(False):

                            from astropy.visualization import ZScaleInterval
                            zscale = ZScaleInterval()
                            from mpl_toolkits.axes_grid1 import make_axes_locatable
                            vmin, vmax = zscale.get_limits(created_map)
                            fig, (ax, axp) = plt.subplots(1,2,figsize=(6,6))
                            im = ax.imshow(created_map, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)  # size can be a percentage or absolute
                            fig.colorbar(im, cax=cax, label='Amplitude')

                            pk = aps.angular_power_spectrum((created_map,),hdr['CDELT1']*60, delta_k_over_k=delta_k_over_k)
                            pk_mes, k = pk.p2()

                            axp.step(k, pk_mes[0], where='mid', c='k')
                            axp.set_ylabel('P(k) [$\\rm Jy^2/sr$]')
                            axp.set_xlabel('k [$\\rm arcmin^{-1}$]')
                            axp.set_yscale('log')
                            axp.set_xscale('log')
                            fig.tight_layout()
                        

    
    #-----------------------------------------------------------------------------------

    BS = 12; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3), dpi=200)
    res_key = 'res=19.94 arcsecs'; prec='float64'
    for downsample_frequency, c in zip(downsampled_freq_list, cm.rainbow(np.linspace(0.,1,len(downsampled_freq_list)))):
        df_key = f'downsample_frequency = {downsample_frequency:.1f} Hz'    
        y_points = []
        for t_int in t_int_list:
            T_key = f'T = {t_int:.1f} min'
            y_points.append(dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes_avg'])

            k = dict_maps_fidelity[T_key][df_key][prec][res_key]['k'] 
            pk = dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes']
            ax2.loglog(k, pk, alpha=0.1, c=c)

        ax1.errorbar(t_int_list, y_points, fmt='o', color = c, ecolor=c, label=f'{downsample_frequency:.1f}Hz')
    ax1.set_xlabel('Integration time [min]')
    ax1.set_ylabel(f"P(k={dict_maps_fidelity['k_bounds'][0]:.1f}-{dict_maps_fidelity['k_bounds'][1]:.1f}"+"$\\rm arcmin^{-1}$)")
    ax1.set_yscale('log')
    ax1.legend(fontsize=BS-4)
    ax2.set_ylabel('P(k) [$\\rm Jy^2/sr$]')
    ax2.set_xlabel('k [$\\rm arcmin^{-1}$]')
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3), dpi=200)
    df_key = f'downsample_frequency = {100:.1f} Hz'; res_key = 'res=19.94 arcsecs'
    for prec,c in zip(precision_list, cm.viridis(np.linspace(0.,1,len(precision_list)))):
        y_points = []
        for t_int in t_int_list:
            T_key = f'T = {t_int:.1f} min'
            y_points.append(dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes_avg'])

            k = dict_maps_fidelity[T_key][df_key][prec][res_key]['k'] 
            pk = dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes']
            ax2.loglog(k, pk, alpha=0.1, c=c)

        ax1.errorbar(t_int_list, y_points, fmt='o', color = c, ecolor=c, label=prec)
    ax1.set_xlabel('Integration time [min]')
    ax1.set_ylabel(f"P(k={dict_maps_fidelity['k_bounds'][0]:.1f}-{dict_maps_fidelity['k_bounds'][1]:.1f}"+"$\\rm arcmin^{-1}$)")
    ax1.set_yscale('log')
    ax1.legend(fontsize=BS-4)
    ax2.set_ylabel('P(k) [$\\rm Jy^2/sr$]')
    ax2.set_xlabel('k [$\\rm arcmin^{-1}$]')
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3), dpi=200)
    df_key = f'downsample_frequency = {100:.1f} Hz'; prec='float64'
    for res_value, c in zip(resolution_list, cm.viridis(np.linspace(0.,1,len(resolution_list)))):
        y_points = []
        res_key = f'res={res_value*3600:.2f} arcsecs'
        for t_int in t_int_list:
            T_key = f'T = {t_int:.1f} min'
            y_points.append(dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes_avg'])

            k = dict_maps_fidelity[T_key][df_key][prec][res_key]['k'] 
            pk = dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes']
            ax2.loglog(k, pk, alpha=0.1, c=c)
        ax1.errorbar(t_int_list, y_points, fmt='o', color = c, ecolor=c, label=f'res={res_value*3600:.2f}arcsecs')
    ax1.set_xlabel('Integration time [min]')
    ax1.set_ylabel(f"P(k={dict_maps_fidelity['k_bounds'][0]:.1f}-{dict_maps_fidelity['k_bounds'][1]:.1f}"+" $\\rm arcmin^{-1}$)")
    ax1.set_yscale('log')
    ax2.set_ylabel('P(k) [$\\rm Jy^2/sr$]')
    ax2.set_xlabel('k [$\\rm arcmin^{-1}$]')
    ax1.legend(fontsize=BS-4)
    fig.tight_layout()

    plt.show()


    
    """
    for t_int in t_int_list:

        T_key = f'T = {t_int:.1f} min'

        for downsample_frequency in downsampled_freq_list:

            df_key = f'downsample_frequency = {downsample_frequency:.1f} Hz'

            for prec in precision_list:

                for res_value in resolution_list:

                    res_key = f'res={res_value*3600:.2f} arcsecs'
                    k = dict_maps_fidelity[T_key][df_key][prec][res_key]['k'] 
                    pk = dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes']
                    aps_avg_original = dict_maps_fidelity[T_key][df_key][prec][res_key]['pk_mes_avg']
                    plt.step(k, pk, where='mid', c='k', alpha=0.1)
                    plt.plot(2e-1,aps_avg_original, 'ok' )
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('P(k) [$\\rm Jy^2/sr$]')
    plt.xlabel('k [$\\rm arcmin^{-1}$]')    
    """
    plt.show()

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

def simulate_celeron_system():
    """Return mocks for CPU and memory to simulate Intel Celeron 4305UE."""
    svmem = namedtuple('svmem', [
        'total', 'available', 'percent', 'used', 'free',
        'active', 'inactive', 'buffers', 'cached', 'shared', 'slab'
    ])
    fake_mem = svmem(
        total=SIM_MEMORY_BYTES,
        available=SIM_MEMORY_BYTES,
        percent=0,
        used=0,
        free=SIM_MEMORY_BYTES,
        active=0,
        inactive=0,
        buffers=0,
        cached=0,
        shared=0,
        slab=0
    )

    cpu_mock = mock.patch("os.cpu_count", return_value=SIM_CORES)
    mem_mock = mock.patch("psutil.virtual_memory", return_value=fake_mem)
    return cpu_mock, mem_mock

if __name__ == "__main__":

    # --- Toggle simulation mode ---
    USE_FAKE_SYSTEM = True  # 🔄 Set False to use your real machine

    #-----------------------
    #I: coadded maps
    perfs_coadded_maps = False
    #II: individual mapscoadd
    perfs_individual_maps = True
    #III a TODs 
    perfs_tods =  False
    #
    perf_fct = False
    #III b raw tods
    perfs_raw_tods = False
    #IV
    tod_fidelity = False
    #V
    map_fidelity = False
    #-----------------------

    #------------------
    LW_min= 317e-6  # Hz
    D = 2.0             # m
    FWHM = 1.22 * LW_min / D * 180 / np.pi  # degrees
    res = FWHM / 2  
    #------------------

    #----------------------------------------------------------------------------------------
    P = load_params(f'{DESKTOP}/'+'timestream_maker/PAR_files/params_strategy_profiling.par')
    P_namap = load_params(f'{DESKTOP}/'+'namap/PAR_FILES/params_namap_profiling.par')
    P_namap['detector_table'] = P['detectors_name_file']
    P_namap['cdelt'] = res, res
    #----------------------------------------------------------------------------------------

    #------------------------------------------------------
    dict_tods_fidelity_file = 'dict_tods_fidelity.p'
    dict_coadded_map_fidelity_file = 'dict_coadded_map_fidelity_test.p'
    dict_coadd_perf = 'namap_perf_profiling_coadded_maps.p'
    dict_individual_perf = 'namap_perf_profiling_individual_maps.p'
    dict_tods_perf = 'namap_perf_downsampled_tods.p'
    dict_tods_raw_perf = 'namap_perf_raw_tods.p'
    dict_fcts = 'namap_perf_of_fcts.p'

    freq_min_for_psd, freq_max_for_psd = 1,6
    k_min_for_aps, freq_max_for_aps = 1e-1,3e-1
    delta_k_over_k = 0.1

    t_int_list = (4, 5, 6, 7, 8, 9,10,15, 25) #min
    downsampled_freq_list = (100,50,150)
    precision_list = ('float32', 'float16') #'float64',
    resolution_list = (res,40/3600,50/3600,60/3600) #deg
    #pix_num
    nb_pixels = (1,2,3,5,6,7,8,9,10,20,30,40,50)
    nb_bands  = (1,2,3,4,11,21, 41,  61,  81, 101, 128)
    #------------------------------------------------------

    #------------------------------------------------------
    exist = False
    tod_file = P['output_path']+P['output_name']
    if('.hdf5' in tod_file and os.path.isfile(tod_file)): exist = True
    elif(os.path.exists(P['output_path']+P['output_name'])): exist = True
    if(not exist):
        print(f'Generating {max(t_int_list)}min timestreams.')
        if(not os.path.isfile(P['detectors_name_file'])): gen_detectors_main(P)
        main_1det(P)
        main_tod(P)
    #------------------------------------------------------

    if USE_FAKE_SYSTEM:
        print("⚙️  Simulating Intel Celeron 4305UE environment...")
        os.environ["OMP_NUM_THREADS"] = str(SIM_CORES)  # restrict OpenMP threads
        os.environ["OPENBLAS_NUM_THREADS"] = str(SIM_CORES)
        os.environ["MKL_NUM_THREADS"] = str(SIM_CORES)
        os.environ["NUMEXPR_NUM_THREADS"] = str(SIM_CORES)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(SIM_CORES)

        cpu_mock, mem_mock = simulate_celeron_system()
        with cpu_mock, mem_mock:
            print("Simulated CPU count:", os.cpu_count())
            print("Simulated RAM (GB):", psutil.virtual_memory().total / 1024**3)
            print("Simulated CPU model:", SIM_CPU)
            print("OpenMP threads limited to:", os.environ["OMP_NUM_THREADS"])

            # Place your performance or profiling code here
            if(perfs_coadded_maps): profiling_coadded_maps(dict_coadd_perf)
            if(perfs_individual_maps): profiling_individual_maps(dict_individual_perf)
            if(perfs_tods): profiling_tods(dict_tods_perf)
            if(perfs_raw_tods): profiling_raw_tods(dict_tods_raw_perf)
            if(perf_fct): profiling_fcts(dict_fcts)
            if(tod_fidelity): test_namap_tods_fidelity(dict_tods_fidelity_file)
            if(map_fidelity): test_namap_coadded_map_fidelity(dict_coadded_map_fidelity_file)
    else:
        print("💻 Using your real system:")
        print("Real CPU count:", os.cpu_count())
        print("Real RAM (GB):", psutil.virtual_memory().total / 1024**3)
        print("Real CPU model:", platform.processor())
        print("OMP threads (default):", os.environ.get("OMP_NUM_THREADS", "not set"))

        #if(perfs_coadded_maps): profiling_coadded_maps('mycomputer_'+dict_coadd_perf)
        #if(perfs_individual_maps): profiling_individual_maps('mycomputer_'+dict_individual_perf)
        if(perfs_tods): profiling_tods('mycomputer_'+dict_tods_perf)