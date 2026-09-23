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

from namap_main import main as namap_main
import src.loaddata as ld
import src.detector as det
import src.psd_analysis as aps

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

DT = dtype_map['float64']
IT = int_map['float64']

def profiling_coadded_maps(dict_file_path, profiling_vs_tod_time =True, profiling_vs_nb_bands=True, load_directly = False):

    if(load_directly): results = pickle.load( open(dict_file_path, 'rb'))
    else: 
        if(os.path.isfile(dict_file_path)): results = pickle.load( open(dict_file_path, 'rb'))
        else: results = {}

    if(profiling_vs_tod_time):

            key = 'profiling vs tod time'
            results.setdefault(key, {})

            results[key]["t_int"] = t_int_list
            map_compression_list = ('coadd.fits',)

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
                        P_namap['save_TODS'] = False
                        P_namap['remove_turnarounds'] = False
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

            for map_compression in ('coadd.fits',):

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
                            if val < 1.25 * max(results[key]["nb dets"]): continue
                        
                        freq_list = 715.0 + 4.0 * np.arange(nband)
                        P_namap['cdelt'] = 40/3600, 40/3600
                        P_namap['frequencies'] = freq_list     
                        P_namap['precision'] = precision                        
                        P_namap['num_frames']  = int(5*60) #integration time in seconds to be loaded. 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['output_map'] = P['output_path']+map_compression
                        P_namap['coadd'] = True
                        P_namap['save_TODS'] = False
                        P_namap['remove_turnarounds'] = False
                        P_namap['downsample_frequency'] = 100
                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        pitot = namap_main(P_namap, val)
                        
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start

                        results[key]["nb dets"].append(pitot)
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
                    P_namap['save_TODS'] = False

                    #------------------------------------------------------
                    tracemalloc.start()
                    start = time.time()
                    pitot = namap_main(P_namap)
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
                                if val < 1.25 * max(results[key]["nb dets"]): continue

                            #results[key]["nb dets"].append(val)
                            freq_list = 715.0 + 4.0 * np.arange(nband)

                            P_namap['cdelt'] = 40/3600, 40/3600
                            P_namap['frequencies'] = freq_list    
                            P_namap['precision'] = precision                        
                            P_namap['num_frames']  = 5 * 60 #seconds 
                            P_namap['first_frame'] = 0 #Starting time in second to loaded
                            P_namap['output_map'] = P['output_path']+map_compression
                            P_namap['coadd'] = False
                            P_namap['save_TODS'] = False
                            

                            #------------------------------------------------------
                            tracemalloc.start()
                            start = time.time()
                            pitot = namap_main(P_namap, val)
                            current, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            end = time.time()
                            timing = end - start

                            results[key]["nb dets"].append(pitot)
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

    if(profiling_vs_tod_time and True):

        key = 'profiling vs tod time'
        results.setdefault(key, {})

        results[key]["t_int"] = t_int_list

        for precision in precision_list: 

            results[key].setdefault(precision, {})

            for compression in ('rawzip', '','.hdf5','zip'):
            
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
                    P_namap['save_TODS'] = True
                    P_namap['output_tods'] = P['output_path']+f'tods_{precision}'+compression
                    P_namap['remove_turnarounds'] = False
                    if('raw' in compression): P_namap['downsample_frequency'] = None
                    else: P_namap['downsample_frequency'] = 100
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
                    if('zip' in compression): 
                        output_file += '.zip'
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

            for compression in ('rawzip','','.hdf5','zip'):
            
                results[key][precision].setdefault(compression, {})
                results[key][precision][compression]['peak memory [MB]'] = []
                results[key][precision][compression]['time [s]'] = []
                results[key][precision][compression]['output size [MB]'] = []
                
                results[key]["nb dets"] = []
                for nband in nb_bands:
                    for npix in nb_pixels:
                        val = int(nband * npix)
                                    
                        # Skip if val is smaller than max so far
                        if results[key]["nb dets"] and val < max(results[key]["nb dets"]): continue

                        if val > 1000 and len(results[key]["nb dets"]) > 0:
                            if val < 1.25 * max(results[key]["nb dets"]): continue

                        
                        freq_list = 715.0 + 4.0 * np.arange(nband)

                        P_namap['cdelt'] = 40/3600, 40/3600
                        P_namap['frequencies'] = freq_list    
                        P_namap['precision'] = precision                        
                        P_namap['num_frames']  = 5 * 60 #seconds 
                        P_namap['first_frame'] = 0 #Starting time in second to loaded
                        P_namap['save_TODS'] = True
                        P_namap['output_tods'] = P['output_path']+f'tods_{precision}'+compression
                        P_namap['remove_turnarounds'] = False
                        if('raw' in compression): P_namap['downsample_frequency'] = None
                        else: P_namap['downsample_frequency'] = 100

                        #------------------------------------------------------
                        tracemalloc.start()
                        start = time.time()
                        pixtot = namap_main(P_namap, val)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        end = time.time()
                        timing = end - start
                        results[key]["nb dets"].append(pixtot)
                        #------------------------------------------------------

                        # Measure output file size (adapt this path!)
                        output_file = P_namap['output_tods']
                        if('zip' in compression): 
                            output_file += '.zip'
                        #if('zip' in compression): output_file += '.zip'

                        if os.path.exists(output_file):
                            if os.path.isfile(output_file):
                                file_size_mb = os.path.getsize(output_file) / 1e6
                            else:
                                file_size_mb = get_dir_size(output_file) / 1e6
                        else: file_size_mb = float('nan') 
                        
                        print(f"Nb pix {val}, time={timing:.2f}s , peak={peak/1e6:.2f}MB , output={file_size_mb:.2f}MB")

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
    perfs_coadded_maps = True
    perfs_individual_maps = True
    perfs_tods = True
    #-----------------------

    #------------------
    LW_min= 317e-6  # Hz
    D = 2.0             # m
    FWHM = 1.22 * LW_min / D * 180 / np.pi  # degrees
    res = FWHM / 2  
    
    P_namap = ld.load_params(f'PAR_FILES/params_namap_profiling.par')
    #------------------

    #------------------------------------------------------
    dict_coadd_perf = 'namap_perf_profiling_coadded_maps.p'
    dict_individual_perf = 'namap_perf_profiling_individual_maps.p'
    dict_tods_perf = 'namap_perf_downsampled_tods.p'

    freq_min_for_psd, freq_max_for_psd = 1,6
    k_min_for_aps, freq_max_for_aps = 1e-1,3e-1
    delta_k_over_k = 0.1

    t_int_list = (4, 5, 6, 7, 8, 9,10,15, 24) #min
    downsampled_freq_list = (100,50,150)
    precision_list = ('float32', 'float16') #'float64',
    resolution_list = (res,40/3600,50/3600,60/3600) #deg
    #pix_num
    nb_pixels = (1,2,3,5,6,7,8,9,10,20,30,40,50)
    nb_bands  = (1,2,3,4,11,21, 41,  61,  81, 101, 128)
    #------------------------------------------------------

    #------------------------------------------------------
    exist = False
    tod_file = P_namap['input_file']
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
    else:
        print("💻 Using your real system:")
        print("Real CPU count:", os.cpu_count())
        print("Real RAM (GB):", psutil.virtual_memory().total / 1024**3)
        print("Real CPU model:", platform.processor())
        print("OMP threads (default):", os.environ.get("OMP_NUM_THREADS", "not set"))

        # Place your performance or profiling code here
        if(perfs_coadded_maps): profiling_coadded_maps(dict_coadd_perf)
        if(perfs_individual_maps): profiling_individual_maps(dict_individual_perf)
        if(perfs_tods): profiling_tods(dict_tods_perf)