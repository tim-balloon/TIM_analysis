
import os
import time
import tracemalloc
from namap_main import *

results = {}

if(True): 
        
    results['profiling vs resolution'] = {}
    results['profiling vs resolution']["resolution ['']"] = (10, 20, 30, 40, 50, 60)
    results['profiling vs resolution']['peak memory [MB]'] = []
    results['profiling vs resolution']['time [s]'] = []
    results['profiling vs resolution']['output size [MB]'] = []

    for res in results['profiling vs resolution']["resolution ['']"]:

        par = load_par_file(f'params_namap.par')
        par['hdf5_file'] = '/home/mvancuyck/Desktop/TODs_profiling/TOD_0h3min0sec.hdf5' 
        par['cdelt'] = res/3600, res/3600
        par['frequencies'] = (715.0,) #GHz

        #------------------------------------------------------
        tracemalloc.start()
        start = time.time()
        main(par)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        timing = end - start
        #------------------------------------------------------

        # Measure output file size (adapt this path!)
        output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
        if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
        else: file_size_mb = float('nan')  # file not found → record NaN or 0

        print(f"Resolution={res}'' | time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

        # Store results
        results['profiling vs resolution']['peak memory [MB]'].append(peak / 1e6)
        results['profiling vs resolution']['time [s]'].append(timing)
        results['profiling vs resolution']['output size [MB]'].append(file_size_mb)

if(False): 

    results['profiling vs nb of detectors'] = {}
    results['profiling vs nb of detectors']["nb detectors"] = (1,5, 10, 20, 30, 40, 50, 63)
    results['profiling vs nb of detectors']['peak memory [MB]'] = []
    results['profiling vs nb of detectors']['time [s]'] = []
    results['profiling vs nb of detectors']['output size [MB]'] = []

    for nb in results['profiling vs nb of detectors']["nb detectors"]:

        par = load_par_file(f'params_namap.par')
        par['hdf5_file'] = '/home/mvancuyck/Desktop/TODs_profiling/TOD_0h3min0sec.hdf5' 
        par['cdelt'] = 40/3600, 40/3600
        par['frequencies'] = (715.0,) #GHz

        #------------------------------------------------------
        tracemalloc.start()
        start = time.time()
        main(par, nb)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        timing = end - start
        #------------------------------------------------------

        # Measure output file size (adapt this path!)
        output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
        if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
        else: file_size_mb = float('nan')  # file not found → record NaN or 0

        print(f"Nb dets {nb}| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

        # Store results
        results['profiling vs nb of detectors']['peak memory [MB]'].append(peak / 1e6)
        results['profiling vs nb of detectors']['time [s]'].append(timing)
        results['profiling vs nb of detectors']['output size [MB]'].append(file_size_mb)

if(False): 
    key = 'profiling vs nb bands'

    results[key] = {}
    results[key]["nb bands"] = (1,5, 10, 20, 30, 40, 50, 60)
    results[key]['peak memory [MB]'] = []
    results[key]['time [s]'] = []
    results[key]['output size [MB]'] = []

    for nb in results[key]["nb bands"]:

        freq_list = 715.0 + 4.0 * np.arange(nb)
        print('nb: ',nb)

        par = load_par_file(f'params_namap.par')
        par['hdf5_file'] = '/home/mvancuyck/Desktop/TODs_profiling/TOD_0h3min0sec.hdf5' 
        par['cdelt'] = 40/3600, 40/3600
        par['frequencies'] = freq_list

        #------------------------------------------------------
        tracemalloc.start()
        start = time.time()
        main(par)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        timing = end - start
        #------------------------------------------------------

        # Measure output file size (adapt this path!)
        output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
        if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
        else: file_size_mb = float('nan')  # file not found → record NaN or 0

        print(f"Nb dets {nb}| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

        # Store results
        results[key]['peak memory [MB]'].append(peak / 1e6)
        results[key]['time [s]'].append(timing)
        results[key]['output size [MB]'].append(file_size_mb)

if(False): 
    key = 'profiling vs tod time'

    results[key] = {}
    results[key]["t_int"] = (1, 2, 3, 4, 5, 10)
    results[key]['peak memory [MB]'] = []
    results[key]['time [s]'] = []
    results[key]['output size [MB]'] = []

    for t in results[key]["t_int"]:

        par = load_par_file(f'params_namap.par')
        par['hdf5_file'] = f'/home/mvancuyck/Desktop/TODs_profiling/TOD_0h{t}min0sec.hdf5' 
        par['cdelt'] = 40/3600, 40/3600
        par['frequencies'] = (715.0,) #GHz

        #------------------------------------------------------
        tracemalloc.start()
        start = time.time()
        main(par)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        timing = end - start
        #------------------------------------------------------

        # Measure output file size (adapt this path!)
        output_file = 'fits_and_hdf5/coadd.fits'  # or whatever your naming convention is
        if os.path.exists(output_file): file_size_mb = os.path.getsize(output_file) / 1e6
        else: file_size_mb = float('nan')  # file not found → record NaN or 0

        print(f"Nb dets {nb}| time={timing:.2f}s | peak={peak/1e6:.2f}MB | output={file_size_mb:.2f}MB")

        # Store results
        results[key]['peak memory [MB]'].append(peak / 1e6)
        results[key]['time [s]'].append(timing)
        results[key]['output size [MB]'].append(file_size_mb)


