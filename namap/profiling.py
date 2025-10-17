
import os
import time
import tracemalloc
from namap_main import *
import pickle

results = {}

if(False): 
        
    results['profiling vs resolution'] = {}
    results['profiling vs resolution']["resolution ['']"] = (10, 20, 30, 40, 50, 60)

    for precision in ('float64', 'float32','float16'): 

        results['profiling vs resolution'][precision] = {}
        results['profiling vs resolution'][precision]['peak memory [MB]'] = []
        results['profiling vs resolution'][precision]['time [s]'] = []
        results['profiling vs resolution'][precision]['output size [MB]'] = []
            
        for res in results['profiling vs resolution']["resolution ['']"]:

            par = load_par_file(f'params_namap.par')
            par['hdf5_file'] = '/home/mvancuyck/Desktop/TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
            par['cdelt'] = res/3600, res/3600
            par['frequencies'] = (715.0,) #GHz
            par['precision'] = precision

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
            results['profiling vs resolution'][precision]['peak memory [MB]'].append(peak / 1e6)
            results['profiling vs resolution'][precision]['time [s]'].append(timing)
            results['profiling vs resolution'][precision]['output size [MB]'].append(file_size_mb)
    

if(False): 

    results['profiling vs nb of detectors'] = {}
    results['profiling vs nb of detectors']["nb detectors"] = (1,5, 10, 20, 30, 40, 50, 64)

    for precision in ('float64', 'float32','float16'): 

        results['profiling vs nb of detectors'][precision] = {}
        results['profiling vs nb of detectors'][precision]['peak memory [MB]'] = []
        results['profiling vs nb of detectors'][precision]['time [s]'] = []
        results['profiling vs nb of detectors'][precision]['output size [MB]'] = []

        for nb in results['profiling vs nb of detectors']["nb detectors"]:

            par = load_par_file(f'params_namap.par')
            par['hdf5_file'] = '/home/mvancuyck/Desktop/TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
            par['cdelt'] = 40/3600, 40/3600
            par['frequencies'] = (715.0,) #GHz
            par['precision'] = precision

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
            results['profiling vs nb of detectors'][precision]['peak memory [MB]'].append(peak / 1e6)
            results['profiling vs nb of detectors'][precision]['time [s]'].append(timing)
            results['profiling vs nb of detectors'][precision]['output size [MB]'].append(file_size_mb)



if(True): 

    key = 'profiling vs nb bands'
    results[key] = {}
    results[key]["nb bands"] = (1,5, 10, 20, 30, 40, 50, 60)

    for precision in ('float64', 'float32','float16'): 

        results[key][precision] = {}
        results[key][precision]['peak memory [MB]'] = []
        results[key][precision]['time [s]'] = []
        results[key][precision]['output size [MB]'] = []


        for nb in results[key]["nb bands"]:

            freq_list = 715.0 + 4.0 * np.arange(nb)

            par = load_par_file(f'params_namap.par')
            par['hdf5_file'] = '/home/mvancuyck/Desktop/TODs_profiling/TOD_0h3min0sec_64bits_488Hz.hdf5' 
            par['cdelt'] = 40/3600, 40/3600
            par['frequencies'] = freq_list
            par['precision'] = precision

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
            results[key][precision]['peak memory [MB]'].append(peak / 1e6)
            results[key][precision]['time [s]'].append(timing)
            results[key][precision]['output size [MB]'].append(file_size_mb)

if(False): 
    key = 'profiling vs tod time'

    results[key] = {}
    results[key]["t_int"] = (1, 2, 3, 4, 5)

    for precision in ('float64', 'float32','float16'): 

        results[key][precision] = {}
        results[key][precision]['peak memory [MB]'] = []
        results[key][precision]['time [s]'] = []
        results[key][precision]['output size [MB]'] = []

        for t in results[key]["t_int"]:

            par = load_par_file(f'params_namap.par')
            par['hdf5_file'] = f'/home/mvancuyck/Desktop/TODs_profiling/TOD_0h{t}min0sec_64bits_488Hz.hdf5' 
            par['cdelt'] = 40/3600, 40/3600
            par['frequencies'] = (715.0,) #GHz        
            par['precision'] = precision

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
            results[key][precision]['peak memory [MB]'].append(peak / 1e6)
            results[key][precision]['time [s]'].append(timing)
            results[key][precision]['output size [MB]'].append(file_size_mb)

pickle.dump(results, open('/home/mvancuyck/Desktop/TODs_profiling/namap_perf_profiling.p', 'wb'))
